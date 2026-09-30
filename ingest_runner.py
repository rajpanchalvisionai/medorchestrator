import asyncio
import os
import sys
import json
import logging
import argparse
import urllib.request
import urllib.error

from schemas.inputs import PatientReportPayload, DoctorQueryPayload
from engine.mmash_loader import MMASHDatasetLoader
from main import stream_mmash_csv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("IngestRunner")

DEFAULT_MMASH_DIR = os.path.join(os.path.dirname(__file__), "DataPaper", "user_1")

async def send_http_post(url: str, payload_dict: dict):
    """Sends an HTTP POST JSON payload to the API server."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload_dict).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
            return resp_data
    except urllib.error.HTTPError as e:
        logger.error(f"HTTP Error {e.code} posting to {url}: {e.read().decode('utf-8')}")
        return None
    except Exception as e:
        logger.error(f"Error posting to {url}: {e}")
        return None

async def run_pipeline(
    user_dir: str = DEFAULT_MMASH_DIR,
    api_base_url: str = "http://127.0.0.1:8000",
    ws_base_url: str = "ws://127.0.0.1:8000",
    limit: int = 30,
    interval: float = 0.5,
    start_row: int = 0
):
    if not os.path.isdir(user_dir):
        logger.error(f"User directory does not exist: {user_dir}")
        return

    patient_id = os.path.basename(os.path.normpath(user_dir))
    logger.info(f"=== Starting Multi-Channel Ingestion for MMASH Patient: {patient_id} ===")
    logger.info(f"Source Directory: {user_dir}")

    loader = MMASHDatasetLoader()

    # 1. CHANNEL 2: Ingest Patient Self-Reports (Activity, Sleep, Questionnaire)
    logger.info("\n--- Ingesting Channel 2: Patient Self-Reports ---")
    reports = loader.load_patient_reports(user_dir, patient_id=patient_id)
    report_url = f"{api_base_url}/api/patient/report"
    # Send first 3 diverse reports
    for rep in reports[:3]:
        logger.info(f"Posting Patient Report: '{rep.raw_text}' (Severity: {rep.perceived_severity})")
        resp = await send_http_post(report_url, rep.model_dump(mode="json"))
        if resp:
            logger.info(f"-> Ingested Event {resp.get('event_id')}: Requires LLM = {resp.get('requires_llm_processing')}")

    # 2. CHANNEL 3: Ingest Doctor Queries
    logger.info("\n--- Ingesting Channel 3: Doctor Queries ---")
    queries = loader.generate_doctor_queries(user_dir, patient_id=patient_id)
    query_url = f"{api_base_url}/api/doctor/query"
    for q in queries:
        logger.info(f"Posting Doctor Query: '{q.query_text}'")
        resp = await send_http_post(query_url, q.model_dump(mode="json"))
        if resp:
            logger.info(f"-> Ingested Event {resp.get('event_id')}: Requires LLM = {resp.get('requires_llm_processing')}")

    # 3. CHANNEL 1: Stream Real Sensor Telemetry over WebSocket
    logger.info(f"\n--- Ingesting Channel 1: Telemetry Stream ({limit} rows) ---")
    ws_url = f"{ws_base_url}/ws/telemetry/{patient_id}"
    await stream_mmash_csv(
        file_or_dir_path=user_dir,
        ws_url=ws_url,
        interval_seconds=interval,
        limit=limit,
        start_row=start_row
    )
    logger.info("=== Multi-Channel Ingestion Run Completed Successfully ===")

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Channel MMASH Dataset Ingestion Runner")
    parser.add_argument("--user-dir", default=DEFAULT_MMASH_DIR, help="Path to MMASH user directory (e.g. DataPaper/user_1)")
    parser.add_argument("--limit", type=int, default=30, help="Number of telemetry rows to stream")
    parser.add_argument("--interval", type=float, default=0.2, help="Seconds between telemetry packets")
    parser.add_argument("--start-row", type=int, default=0, help="Start row offset in Actigraph.csv")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_pipeline(
        user_dir=args.user_dir,
        limit=args.limit,
        interval=args.interval,
        start_row=args.start_row
    ))
