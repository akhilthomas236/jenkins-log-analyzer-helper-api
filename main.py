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

load_dotenv()

app = FastAPI(title="Jenkins Log Analyzer API")

JENKINS_URL = os.getenv("JENKINS_URL", "http://localhost:8080")
JENKINS_USER = os.getenv("JENKINS_USER", "user")
JENKINS_TOKEN = os.getenv("JENKINS_TOKEN", "token")
JOB_NAME = os.getenv("JENKINS_JOB", "example-job")

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

def fetch_jenkins_log(build_type: str = "latest") -> str:
    if build_type == "latest":
        url = f"{JENKINS_URL}/job/{JOB_NAME}/lastBuild/consoleText"
    elif build_type == "last-success":
        url = f"{JENKINS_URL}/job/{JOB_NAME}/lastSuccessfulBuild/consoleText"
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
    resp = requests.get(url, auth=(JENKINS_USER, JENKINS_TOKEN))
    if resp.status_code != 200:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_name}")
    data = resp.json()
    # Jenkins node info
    node_info = {
        "node": node_name,
        "offline": data.get("offline", False),
        "executors": data.get("numExecutors", 0),
        "architecture": data.get("monitorData", {}).get("hudson.node_monitors.ArchitectureMonitor", "unknown"),
        "os": data.get("displayName", "unknown"),
        # Jenkins does not expose GC/memory by default; placeholder for plugins
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
def get_latest_log():
    return {"log": fetch_jenkins_log("latest")}

@app.get("/logs/last-success")
def get_last_success_log():
    return {"log": fetch_jenkins_log("last-success")}

@app.get("/logs/compare", response_model=LogComparisonResult)
def compare_logs():
    latest = fetch_jenkins_log("latest")
    last_success = fetch_jenkins_log("last-success")
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
    """
    Returns Jenkins node metrics using Jenkins API. For master or any agent node.
    """
    metrics = fetch_jenkins_node_metrics(node)
    return JenkinsNodeMetrics(**metrics)

@app.get("/metrics/jenkins-nodes")
def get_all_jenkins_nodes_metrics():
    """
    Returns metrics for all Jenkins nodes (master and all agents).
    """
    url = f"{JENKINS_URL}/computer/api/json"
    resp = requests.get(url, auth=(JENKINS_USER, JENKINS_TOKEN))
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
