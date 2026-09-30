import asyncio
import csv
import json
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
import uvicorn
import websockets

from schemas.inputs import (
    TelemetryPayload,
    PatientReportPayload,
    DoctorQueryPayload,
    IngestionEvent
)
from engine.ingestion import IngestionRouter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HealthOrchestrator")

app = FastAPI(title="Multi-Channel Clinical Telemetry & Message Ingestion Engine")
router = IngestionRouter()

@app.post("/api/patient/report", response_model=IngestionEvent)
async def post_patient_report(payload: PatientReportPayload):
    try:
        event = router.process_event(payload)
        logger.info(f"Processed Patient Report: {event.event_id} (Requires LLM: {event.requires_llm_processing})")
        return event
    except Exception as e:
        logger.error(f"Error processing patient report: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/doctor/query", response_model=IngestionEvent)
async def post_doctor_query(payload: DoctorQueryPayload):
    try:
        event = router.process_event(payload)
        logger.info(f"Processed Doctor Query: {event.event_id} (Requires LLM: {event.requires_llm_processing})")
        return event
    except Exception as e:
        logger.error(f"Error processing doctor query: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.websocket("/ws/telemetry/{patient_id}")
async def ws_telemetry(websocket: WebSocket, patient_id: str):
    await websocket.accept()
    logger.info(f"WebSocket telemetry connection accepted for patient: {patient_id}")
    try:
        while True:
            data = await websocket.receive_text()
            try:
                raw_payload = json.loads(data)
                # Inject patient_id if not present or ensure it matches
                raw_payload["patient_id"] = patient_id
                
                payload = TelemetryPayload.model_validate(raw_payload)
                event = router.process_event(payload)
                
                await websocket.send_json(event.model_dump(mode="json"))
                
                if event.requires_llm_processing:
                    logger.warning(f"ANOMALY DETECTED for patient {patient_id}! Event {event.event_id}")
                
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON format"})
            except Exception as e:
                logger.error(f"Error handling telemetry item: {e}")
                await websocket.send_json({"error": str(e)})
    except WebSocketDisconnect:
        logger.info(f"WebSocket telemetry connection closed for patient: {patient_id}")

async def stream_mmash_csv(
    file_or_dir_path: str, 
    ws_url: str = "ws://localhost:8000/ws/telemetry/user_1",
    interval_seconds: float = 1.0,
    limit: Optional[int] = None,
    start_row: int = 0
):
    """
    Simulates MMASH dataset stream by reading from a MMASH user directory or CSV file
    and pushing rows into the WebSocket at a controlled rate.
    """
    import os
    logger.info(f"Starting MMASH stream simulation from {file_or_dir_path} to {ws_url}")
    
    # Check if path is a MMASH user directory (or contains Actigraph.csv)
    is_user_dir = os.path.isdir(file_or_dir_path)
    user_dir = file_or_dir_path if is_user_dir else os.path.dirname(file_or_dir_path)
    has_actigraph = os.path.exists(os.path.join(user_dir, "Actigraph.csv")) if user_dir else False

    try:
        async with websockets.connect(ws_url) as websocket:
            if has_actigraph:
                from engine.mmash_loader import MMASHDatasetLoader
                loader = MMASHDatasetLoader()
                patient_id = os.path.basename(os.path.normpath(user_dir))
                logger.info(f"Streaming authentic MMASH telemetry using MMASHDatasetLoader for {patient_id}...")
                for payload in loader.stream_telemetry(user_dir, patient_id=patient_id, limit=limit, start_row=start_row):
                    payload_dict = payload.model_dump(mode="json")
                    await websocket.send(json.dumps(payload_dict))
                    response = await websocket.recv()
                    logger.info(f"Received from WebSocket: {response}")
                    if interval_seconds > 0:
                        await asyncio.sleep(interval_seconds)
            else:
                with open(file_or_dir_path, mode="r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    count = 0
                    for row in reader:
                        if limit is not None and count >= limit:
                            break
                        try:
                            timestamp_str = row.get("timestamp", datetime.utcnow().isoformat())
                            hr = float(row.get("heart_rate") or row.get("HR") or row.get("Heart Rate") or 0.0)
                            hrv = float(row.get("hrv_rmssd") or row.get("RMSSD") or 50.0)
                            accel = float(row.get("accelerometer_magnitude") or row.get("Vector Magnitude") or row.get("Axis1") or row.get("Activity") or 0.0)
                            
                            payload = {
                                "timestamp": timestamp_str,
                                "heart_rate": hr,
                                "hrv_rmssd": hrv,
                                "accelerometer_magnitude": accel,
                                "channel_source": "wearable_stream"
                            }
                            
                            await websocket.send(json.dumps(payload))
                            response = await websocket.recv()
                            logger.info(f"Received from WebSocket: {response}")
                            count += 1
                        except Exception as row_err:
                            logger.error(f"Error parsing or sending row {row}: {row_err}")
                            
                        if interval_seconds > 0:
                            await asyncio.sleep(interval_seconds)
    except Exception as e:
        logger.error(f"MMASH streaming simulation error: {e}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
