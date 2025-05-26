# Jenkins Log Analyzer API

This project provides a FastAPI-based API to fetch Jenkins log files, compare the latest log with the last successful log, extract insights, and detect anomalies using machine learning algorithms.

## Features
- Fetch Jenkins log files via API
- Compare latest and last successful logs
- Extract insights from log differences
- Detect anomalies using ML algorithms (e.g., Isolation Forest)

## How to Run

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

2. **Configure Jenkins connection:**
   - Create a `.env` file in the project root with the following content:
     ```env
     JENKINS_URL=http://your-jenkins-url:8080
     JENKINS_USER=your-jenkins-username
     JENKINS_TOKEN=your-jenkins-api-token
     JENKINS_JOB=your-job-name
     ```

3. **Start the FastAPI server:**
   - Using VS Code task: Press `Cmd+Shift+P` (or `Ctrl+Shift+P`), type `Run Task`, and select `Run FastAPI server`.
   - Or run manually:
     ```sh
     uvicorn jenkins_log_analyzer.app:app --reload
     ```

4. **Open the Swagger UI:**
   - Visit [http://localhost:8000/docs](http://localhost:8000/docs) in your browser for the interactive API documentation.

## Endpoints
- `/logs/latest?job_name=...` - Get the latest Jenkins log for a job
- `/logs/last-success?job_name=...` - Get the last successful Jenkins log for a job
- `/logs/compare?job_name=...` - Compare latest and last successful logs for a job, extract insights, and detect anomalies
- `/logs/analyze?job_name=...` - Analyze the latest Jenkins log for a job using regex-based insights if no successful build is available. Returns all error messages and the most probable error message that caused the failure.
- `/metrics/jenkins-node?node=master` - Jenkins master node metrics
- `/metrics/jenkins-node?node=<agent_name>` - Jenkins agent/slave node metrics
- `/metrics/jenkins-nodes` - All Jenkins nodes metrics
- `/jobs/search?query=...` - Search Jenkins jobs by name substring
- `/users/list` - List all Jenkins users
- `/users/jobs/latest?user=...` - List the latest jobs executed by a user
- `/plugins/health` - Check the health of all installed Jenkins plugins
- `/metrics/cloudbees` - Fetch CloudBees/Jenkins metrics using the CloudBees Metrics API
- `/currentUser/metrics` - Fetch CloudBees/Jenkins metrics for the current user
- `/metrics/currentUser/healthcheck` - Healthcheck for CloudBees/Jenkins metrics API for the current user

## Requirements
- Python 3.8+
- FastAPI
- scikit-learn
- requests
- python-dotenv
- psutil

---

Replace Jenkins log fetching logic with your Jenkins server details as needed.

**Note:**
- The Jenkins job name is now a required query parameter for log endpoints (not in `.env`).
- Add `JENKINS_METRICS_TOKEN` to your `.env` for metrics endpoints.
- To run the app from the new package structure, use:
  ```sh
  uvicorn jenkins_log_analyzer.app:app --reload
  ```
