"""
Retraining Trigger
Monitors drift scores and initiates a retraining pipeline if thresholds are exceeded.
Queries Prometheus for drift metrics and fires a GitHub Actions repository_dispatch
event when drift exceeds the configured threshold.
"""

import requests
import json
import time
import os
from loguru import logger
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# ── Configuration ──────────────────────────────────────────────────────────────
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.5"))
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL_SECONDS", "60"))


def check_drift_and_trigger() -> None:
    """Queries Prometheus for drift share and triggers GHA if above threshold."""
    try:
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": "ml_drift_share"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        if data["status"] == "success" and data["data"]["result"]:
            drift_value = float(data["data"]["result"][0]["value"][1])
            logger.info(f"Current Drift Share: {drift_value:.4f} (threshold={DRIFT_THRESHOLD})")

            if drift_value > DRIFT_THRESHOLD:
                logger.warning(
                    f"DRIFT DETECTED ({drift_value:.4f} > {DRIFT_THRESHOLD})! "
                    "Triggering retraining pipeline..."
                )
                trigger_retraining_workflow(drift_value)
            else:
                logger.info("Drift is within acceptable range. No action needed.")
        else:
            logger.info("No drift metrics found in Prometheus yet.")

    except requests.exceptions.RequestException as e:
        logger.error(f"Error querying Prometheus: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in drift check: {e}")


def trigger_retraining_workflow(drift_value: float = 0.0) -> None:
    """
    Triggers a GitHub Action via repository_dispatch.
    Requires GITHUB_TOKEN and GITHUB_REPO (format: owner/repo) env vars.
    """
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPO")  # format: owner/repo-name

    if not token or not repo:
        logger.warning(
            "GITHUB_TOKEN or GITHUB_REPO not set — skipping live dispatch. "
            "Set these env vars to enable automated retraining."
        )
        return

    url = f"https://api.github.com/repos/{repo}/dispatches"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
    }
    payload = {
        "event_type": "drift_retrain",
        "client_payload": {
            "drift_value": round(drift_value, 4),
            "threshold": DRIFT_THRESHOLD,
        },
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code == 204:
            logger.success(
                f"GitHub Actions retraining workflow triggered successfully! "
                f"(drift={drift_value:.4f})"
            )
        else:
            logger.error(
                f"Failed to trigger GitHub Action: "
                f"HTTP {response.status_code} — {response.text}"
            )
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error when triggering GitHub Action: {e}")


if __name__ == "__main__":
    logger.info("Starting Retraining Trigger service...")
    logger.info(
        f"Config: PROMETHEUS={PROMETHEUS_URL}, THRESHOLD={DRIFT_THRESHOLD}, "
        f"INTERVAL={CHECK_INTERVAL}s"
    )
    while True:
        check_drift_and_trigger()
        time.sleep(CHECK_INTERVAL)
