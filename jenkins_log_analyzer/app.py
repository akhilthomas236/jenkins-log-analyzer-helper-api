import os
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Any
import requests
from sklearn.ensemble import IsolationForest
from dotenv import load_dotenv
import psutil
import platform
import gc
from fastapi.openapi.utils import get_openapi
from fastapi.responses import RedirectResponse
import re

load_dotenv()

app = FastAPI(title="Jenkins Log Analyzer API")

JENKINS_URL = os.getenv("JENKINS_URL", "http://localhost:8080")
JENKINS_USER = os.getenv("JENKINS_USER", "user")
JENKINS_TOKEN = os.getenv("JENKINS_TOKEN", "token")
JENKINS_METRICS_TOKEN = os.getenv("JENKINS_METRICS_TOKEN", "metrics_token")

class LogComparisonResult(BaseModel):
    latest_log: str
    last_success_log: str
    diff_lines: int
    insights: Dict[str, Any]
    anomalies: Dict[str, Any]

class NodeMetrics(BaseModel):
    node: str
    cpu_percent: float
    memory_percent: float
    total_memory: float
    used_memory: float
    gc_collections: dict

class JenkinsNodeMetrics(BaseModel):
    node: str
    offline: bool
    executors: int
    architecture: str
    os: str
    gc_collections: dict | None = None
    memory: dict | None = None

def fetch_jenkins_log(build_type: str = "latest", job_name: str = None) -> str:
    if not job_name:
        raise HTTPException(status_code=400, detail="job_name parameter is required")
    if build_type == "latest":
        url = f"{JENKINS_URL}/job/{job_name}/lastBuild/consoleText"
    elif build_type == "last-success":
        url = f"{JENKINS_URL}/job/{job_name}/lastSuccessfulBuild/consoleText"
    else:
        raise ValueError("Invalid build_type")
    resp = requests.get(url, auth=(JENKINS_USER, JENKINS_TOKEN))
    if resp.status_code != 200:
        raise HTTPException(status_code=404, detail=f"Log not found: {url}")
    return resp.text

def extract_insights(latest: str, last_success: str) -> Dict[str, Any]:
    latest_lines = set(latest.splitlines())
    last_success_lines = set(last_success.splitlines())
    new_lines = latest_lines - last_success_lines
    removed_lines = last_success_lines - latest_lines
    return {
        "new_lines_count": len(new_lines),
        "removed_lines_count": len(removed_lines),
        "new_lines_sample": list(new_lines)[:5],
        "removed_lines_sample": list(removed_lines)[:5],
    }

def detect_anomalies(log: str) -> Dict[str, Any]:
    # Simple anomaly detection: line length as feature
    lines = log.splitlines()
    if not lines:
        return {"anomaly_score": [], "anomalies": []}
    X = [[len(line)] for line in lines]
    clf = IsolationForest(contamination=0.05, random_state=42)
    preds = clf.fit_predict(X)
    anomaly_lines = [lines[i] for i, p in enumerate(preds) if p == -1]
    return {
        "anomaly_count": len(anomaly_lines),
        "anomaly_lines_sample": anomaly_lines[:5],
    }

# Helper to get system metrics
def get_system_metrics(node_name: str) -> dict:
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    gc_stats = gc.get_stats() if hasattr(gc, 'get_stats') else {}
    return {
        "node": node_name,
        "cpu_percent": cpu,
        "memory_percent": mem.percent,
        "total_memory": mem.total / (1024 ** 3),
        "used_memory": mem.used / (1024 ** 3),
        "gc_collections": gc_stats if gc_stats else {"collected": gc.get_count()}
    }

def fetch_jenkins_node_metrics(node_name: str) -> dict:
    url = f"{JENKINS_URL}/computer/{node_name}/api/json"
    resp = requests.get(url, auth=(JENKINS_USER, JENKINS_METRICS_TOKEN))
    if resp.status_code != 200:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_name}")
    data = resp.json()
    node_info = {
        "node": node_name,
        "offline": data.get("offline", False),
        "executors": data.get("numExecutors", 0),
        "architecture": data.get("monitorData", {}).get("hudson.node_monitors.ArchitectureMonitor", "unknown"),
        "os": data.get("displayName", "unknown"),
        "gc_collections": None,
        "memory": data.get("monitorData", {}).get("hudson.node_monitors.SwapSpaceMonitor", None)
    }
    return node_info

@app.get("/", include_in_schema=False)
def docs_redirect():
    """
    Redirect root URL to Swagger UI docs.
    """
    return RedirectResponse(url="/docs")

# Optionally, you can customize the OpenAPI schema if you want a custom title/description:
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Jenkins Log Analyzer API",
        version="1.0.0",
        description="API for Jenkins log analysis, comparison, insights, anomaly detection, and Jenkins node metrics.",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

@app.get("/logs/latest")
def get_latest_log(job_name: str):
    return {"log": fetch_jenkins_log("latest", job_name)}

@app.get("/logs/last-success")
def get_last_success_log(job_name: str):
    return {"log": fetch_jenkins_log("last-success", job_name)}

@app.get("/logs/compare", response_model=LogComparisonResult)
def compare_logs(job_name: str):
    latest = fetch_jenkins_log("latest", job_name)
    last_success = fetch_jenkins_log("last-success", job_name)
    insights = extract_insights(latest, last_success)
    anomalies = detect_anomalies(latest)
    diff_lines = abs(len(latest.splitlines()) - len(last_success.splitlines()))
    return LogComparisonResult(
        latest_log=latest,
        last_success_log=last_success,
        diff_lines=diff_lines,
        insights=insights,
        anomalies=anomalies
    )

@app.get("/metrics/node", response_model=NodeMetrics)
def get_node_metrics(node: str = "master"):
    """
    Returns system metrics for the given node (default: master).
    If running on Jenkins slave, pass node name as query param.
    """
    metrics = get_system_metrics(node)
    return NodeMetrics(**metrics)

@app.get("/metrics/jenkins-node", response_model=JenkinsNodeMetrics)
def get_jenkins_node_metrics(node: str = Query("master", description="Jenkins node name (e.g., master, agent name)")):
    metrics = fetch_jenkins_node_metrics(node)
    return JenkinsNodeMetrics(**metrics)

@app.get("/metrics/jenkins-nodes")
def get_all_jenkins_nodes_metrics():
    """
    Returns metrics for all Jenkins nodes (master and all agents).
    """
    url = f"{JENKINS_URL}/computer/api/json"
    resp = requests.get(url, auth=(JENKINS_USER, JENKINS_METRICS_TOKEN))
    if resp.status_code != 200:
        raise HTTPException(status_code=404, detail="Could not fetch Jenkins nodes")
    data = resp.json()
    nodes = data.get("computer", [])
    result = []
    for node in nodes:
        node_name = node.get("displayName", "unknown")
        node_info = {
            "node": node_name,
            "offline": node.get("offline", False),
            "executors": node.get("numExecutors", 0),
            "architecture": node.get("monitorData", {}).get("hudson.node_monitors.ArchitectureMonitor", "unknown"),
            "os": node.get("displayName", "unknown"),
            "gc_collections": None,
            "memory": node.get("monitorData", {}).get("hudson.node_monitors.SwapSpaceMonitor", None)
        }
        result.append(node_info)
    return result

@app.get("/jobs/search")
def search_jobs(query: str = Query(..., description="Search string for Jenkins jobs")):
    """
    Search Jenkins jobs by name substring.
    """
    url = f"{JENKINS_URL}/api/json?tree=jobs[name]"
    resp = requests.get(url, auth=(JENKINS_USER, JENKINS_TOKEN))
    if resp.status_code != 200:
        raise HTTPException(status_code=404, detail="Could not fetch jobs from Jenkins")
    jobs = resp.json().get("jobs", [])
    matched = [job["name"] for job in jobs if query.lower() in job["name"].lower()]
    return {"matched_jobs": matched}

@app.get("/users/list")
def list_jenkins_users():
    """
    List all Jenkins users (requires appropriate permissions).
    """
    url = f"{JENKINS_URL}/asynchPeople/api/json"
    resp = requests.get(url, auth=(JENKINS_USER, JENKINS_TOKEN))
    if resp.status_code != 200:
        raise HTTPException(status_code=404, detail="Could not fetch users from Jenkins")
    users = resp.json().get("users", [])
    user_list = [user["user"].get("fullName", user["user"].get("id", "unknown")) for user in users if user.get("user")]
    return {"users": user_list}

@app.get("/users/jobs/latest")
def get_latest_jobs_by_user(user: str = Query(..., description="Jenkins user id or fullName")):
    """
    List the latest jobs executed by a given user (best effort, based on build data).
    """
    url = f"{JENKINS_URL}/api/json?tree=jobs[name,builds[fullDisplayName,building,result,timestamp,actions[causes[userId,userName]]]]"
    resp = requests.get(url, auth=(JENKINS_USER, JENKINS_TOKEN))
    if resp.status_code != 200:
        raise HTTPException(status_code=404, detail="Could not fetch jobs/builds from Jenkins")
    jobs = resp.json().get("jobs", [])
    user_jobs = []
    for job in jobs:
        job_name = job.get("name")
        for build in job.get("builds", []):
            for action in build.get("actions", []):
                for cause in action.get("causes", []):
                    if (cause.get("userId") == user) or (cause.get("userName") == user):
                        user_jobs.append({
                            "job": job_name,
                            "build": build.get("fullDisplayName"),
                            "result": build.get("result"),
                            "timestamp": build.get("timestamp")
                        })
    # Sort by timestamp descending and return latest per job
    latest_per_job = {}
    for entry in sorted(user_jobs, key=lambda x: x["timestamp"], reverse=True):
        if entry["job"] not in latest_per_job:
            latest_per_job[entry["job"]] = entry
    return {"latest_jobs": list(latest_per_job.values())}

@app.get("/plugins/health")
def check_plugins_health():
    """
    Check the health of all installed Jenkins plugins.
    """
    url = f"{JENKINS_URL}/pluginManager/api/json?depth=1"
    resp = requests.get(url, auth=(JENKINS_USER, JENKINS_TOKEN))
    if resp.status_code != 200:
        raise HTTPException(status_code=404, detail="Could not fetch plugins from Jenkins")
    plugins = resp.json().get("plugins", [])
    plugin_health = []
    for plugin in plugins:
        health = {
            "shortName": plugin.get("shortName"),
            "version": plugin.get("version"),
            "active": plugin.get("active"),
            "enabled": plugin.get("enabled"),
            "hasUpdate": plugin.get("hasUpdate", False),
            "pinned": plugin.get("pinned", False),
            "deleted": plugin.get("deleted", False),
            "downgradable": plugin.get("downgradable", False),
            "supportsDynamicLoad": plugin.get("supportsDynamicLoad", False),
            "health": plugin.get("healthReport", []),
        }
        plugin_health.append(health)
    return {"plugins": plugin_health}

def extract_insights_no_success(log: str) -> dict:
    # Use regex to extract error/warning/info lines and counts
    error_lines = []
    probable_error = None
    # Find all lines containing error/failed/exception and extract the full line
    for line in log.splitlines():
        if re.search(r"(error|failed|exception)", line, re.IGNORECASE):
            error_lines.append(line.strip())
    warning_lines = [line.strip() for line in log.splitlines() if re.search(r"(warn|deprecated)", line, re.IGNORECASE)]
    info_lines = [line.strip() for line in log.splitlines() if re.search(r"(info|started|completed)", line, re.IGNORECASE)]
    # Heuristic: last error line is often the root cause, but skip lines that are just 'ERROR' or similar
    for err in reversed(error_lines):
        if err.strip().lower() not in ["error", "failed", "exception"] and len(err.strip()) > 8:
            probable_error = err.strip()
            break
    if not probable_error and error_lines:
        probable_error = error_lines[-1].strip()
    return {
        "error_count": len(error_lines),
        "warning_count": len(warning_lines),
        "info_count": len(info_lines),
        "error_samples": error_lines[:5],
        "warning_samples": warning_lines[:5],
        "info_samples": info_lines[:5],
        "all_error_messages": error_lines,
        "most_probable_error": probable_error
    }

@app.get("/logs/analyze", response_model=dict)
def analyze_log_when_no_success(job_name: str):
    """
    Analyze the latest Jenkins log for a job when no successful build is available. Uses regex to extract errors, warnings, and info.
    """
    latest_log = fetch_jenkins_log("latest", job_name)
    # Try to fetch last-success, if not found, do regex-based analysis
    try:
        last_success_log = fetch_jenkins_log("last-success", job_name)
        if last_success_log:
            return {"message": "A successful build exists. Use /logs/compare for detailed analysis."}
    except HTTPException:
        pass
    insights = extract_insights_no_success(latest_log)
    return {"insights": insights}

@app.get("/metrics/cloudbees")
def get_cloudbees_metrics():
    """
    Fetch CloudBees/Jenkins metrics using the CloudBees Metrics API.
    """
    url = f"{JENKINS_URL}/metrics/api/json"
    resp = requests.get(url, auth=(JENKINS_USER, JENKINS_METRICS_TOKEN))
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="Could not fetch CloudBees metrics")
    return resp.json()

@app.get("/currentUser/metrics")
def get_current_user_metrics():
    """
    Fetch CloudBees/Jenkins metrics for the current user using the CloudBees Metrics API.
    """
    url = f"{JENKINS_URL}/metrics/{JENKINS_METRICS_TOKEN}/metrics"
    resp = requests.get(url)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="Could not fetch CloudBees metrics")
    return resp.json()

@app.get("/currentUser/metrics/summary")
def get_current_user_metrics_summary():
    """
    Fetch and summarize key JVM, system, and Jenkins-specific metrics from the CloudBees/Jenkins metrics API for the current user.
    """
    url = f"{JENKINS_URL}/metrics/{JENKINS_METRICS_TOKEN}/metrics"
    resp = requests.get(url)
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="Could not fetch CloudBees metrics")
    data = resp.json()
    gauges = data.get("gauges", {})
    summary_keys = [
        # JVM/system metrics
        "system.cpu.load",
        "vm.uptime.milliseconds",
        "vm.count",
        "vm.new.count",
        "vm.timed_waiting.count",
        "vm.blocked.count",
        "vm.deadlocks",
        "vm.memory.heap.init",
        "vm.memory.heap.committed",
        "vm.memory.heap.max",
        "vm.memory.heap.usage",
        "vm.memory.non-heap.init",
        "vm.memory.non-heap.committed",
        "vm.memory.non-heap.max",
        "vm.memory.total.committed",
        "vm.memory.total.max",
        "vm.daemon.count",
        # Jenkins-specific metrics
        "jenkins.executor.count.value",
        "jenkins.executor.free.value",
        "jenkins.executor.in-use.value",
        "jenkins.job.blocked.duration",
        "jenkins.job.building.duration",
        "jenkins.job.queuing.duration",
        "jenkins.job.buildable.duration",
        "jenkins.job.waiting.duration",
        "jenkins.job.total.duration",
        "jenkins.job.count.value",
        "jenkins.job.scheduled",
        "jenkins.node.count.value",
        "jenkins.node.offline.value",
        "jenkins.plugins.active",
        "jenkins.plugins.failed",
        "jenkins.plugins.withUpdate",
        "jenkins.queue.size.value",
        "jenkins.queue.stuck.value"
    ]
    # Also include all vm.gc.X.count metrics
    gc_counts = {k: v.get("value") for k, v in gauges.items() if k.startswith("vm.gc.") and k.endswith(".count")}
    summary = {k: gauges[k]["value"] for k in summary_keys if k in gauges and "value" in gauges[k]}
    summary.update(gc_counts)
    return {"metrics_summary": summary}

@app.get("/metrics/currentUser/healthcheck")
def get_current_user_metrics_healthcheck():
    """
    Healthcheck for CloudBees/Jenkins metrics API for the current user.
    Returns status and a minimal metric if available.
    """
    url = f"{JENKINS_URL}/metrics/{JENKINS_METRICS_TOKEN}/healthcheck"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            # Return a minimal healthcheck, e.g., number of metrics returned
            return {"status": "ok", "metrics_count": len(data) if isinstance(data, dict) else 0}
        else:
            return {"status": "error", "code": resp.status_code, "detail": resp.text}
    except Exception as e:
        return {"status": "error", "detail": str(e)}