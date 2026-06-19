#!/bin/bash
# deploy.sh - Self-hosted CI/CD webhook handler
#
# Triggered by GitHub webhook or manual invocation.
# Pulls latest code, rebuilds frontend, refreshes Python deps, restarts services.
#
# Usage:  ./deploy.sh          (full deploy)
#         ./deploy.sh backend  (backend only)
#         ./deploy.sh frontend (frontend only)

set -euo pipefail

PROJECT_ROOT="/opt/infrastructure-predictor"
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
PYTHON="C:/Users/Adrian/AppData/Local/Programs/Python/Python312/python.exe"
LOG_FILE="$PROJECT_ROOT/deploy.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

deploy_backend() {
    log "--- Backend Deploy ---"

    cd "$BACKEND_DIR"

    # Pull latest code
    log "Pulling from git..."
    git pull origin master 2>&1 | tee -a "$LOG_FILE"

    # Update Python dependencies
    log "Updating Python dependencies..."
    "$PYTHON" -m pip install -r requirements.txt --quiet 2>&1 | tee -a "$LOG_FILE"

    # Re-fetch data & retrain model
    log "Retraining model..."
    "$PYTHON" -X utf8 fetch_data.py 2>&1 | tee -a "$LOG_FILE"
    "$PYTHON" -X utf8 clean_data.py 2>&1 | tee -a "$LOG_FILE"
    "$PYTHON" -X utf8 train.py 2>&1 | tee -a "$LOG_FILE"

    # Restart backend service
    log "Restarting fastapi-backend service..."
    systemctl restart fastapi-backend 2>&1 | tee -a "$LOG_FILE" || \
        log "WARNING: systemctl not available (WSL/Windows). Restart manually."

    log "Backend deploy complete."
}

deploy_frontend() {
    log "--- Frontend Deploy ---"

    cd "$FRONTEND_DIR"

    log "Pulling from git..."
    git pull origin master 2>&1 | tee -a "$LOG_FILE"

    log "Installing npm dependencies..."
    npm ci --silent 2>&1 | tee -a "$LOG_FILE"

    log "Building frontend..."
    npm run build 2>&1 | tee -a "$LOG_FILE"

    log "Frontend deploy complete (dist/ updated)."
}

# Main
log "============================================"
log "Deploy triggered: ${1:-full}"

case "${1:-full}" in
    backend)
        deploy_backend
        ;;
    frontend)
        deploy_frontend
        ;;
    full|*)
        deploy_backend
        deploy_frontend
        ;;
esac

log "============================================"
log "Deploy finished successfully."
