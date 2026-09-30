import os
import csv
from datetime import datetime, timedelta, timezone
from typing import Iterator, List, Optional, Dict
import pandas as pd
import numpy as np

from schemas.inputs import (
    TelemetryPayload,
    PatientReportPayload,
    DoctorQueryPayload
)

# Mapping of MMASH Activity IDs to human-readable descriptions
ACTIVITY_MAP = {
    1: "Sleeping",
    2: "Laying down",
    3: "Sitting",
    4: "Light movement",
    5: "Medium movement",
    6: "Heavy movement",
    7: "Eating",
    8: "Small screen usage",
    9: "Large screen usage",
    10: "Coffee consumption",
    11: "Smoking",
    12: "Alcohol consumption",
    13: "Other"
}

class MMASHDatasetLoader:
    """
    Parser and synchronizer for the MMASH dataset.
    Loads real Actigraph, RR, Activity, Sleep, and Questionnaire data into
    Pydantic schemas for multi-channel ingestion.
    """
    def __init__(self, base_date: Optional[datetime] = None):
        # Default base anchor date for synchronizing day 1 and day 2 timestamps
        self.base_date = base_date or datetime(2026, 8, 23, 0, 0, 0, tzinfo=timezone.utc)

    def _parse_time(self, day: int, time_str: str) -> datetime:
        """Converts day (1 or 2) and 'HH:MM:SS' into an absolute datetime."""
        parts = time_str.strip().split(":")
        h = int(parts[0])
        m = int(parts[1])
        s = int(parts[2]) if len(parts) > 2 else 0
        return self.base_date + timedelta(days=int(day) - 1, hours=h, minutes=m, seconds=s)

    def compute_user_rmssd_map(self, user_dir: str) -> Dict[tuple, float]:
        """
        Calculates HRV (RMSSD in ms) from RR.csv grouped by (day, time).
        Falls back to user average RMSSD when point-in-time calculation has < 2 beats.
        """
        rr_path = os.path.join(user_dir, "RR.csv")
        if not os.path.exists(rr_path):
            return {}

        df_rr = pd.read_csv(rr_path)
        # Calculate successive differences in milliseconds
        df_rr['diff_ms'] = df_rr['ibi_s'].diff() * 1000.0
        df_rr['sq_diff'] = df_rr['diff_ms'] ** 2

        # Group by (day, time)
        grouped = df_rr.groupby(['day', 'time'])['sq_diff'].mean()
        rmssd_map = np.sqrt(grouped).to_dict()
        
        # User global mean RMSSD as fallback
        global_mean = float(np.sqrt(df_rr['sq_diff'].mean()))
        rmssd_map['__default__'] = global_mean if not np.isnan(global_mean) else 50.0
        return rmssd_map

    def stream_telemetry(
        self, 
        user_dir: str, 
        patient_id: Optional[str] = None,
        limit: Optional[int] = None,
        start_row: int = 0
    ) -> Iterator[TelemetryPayload]:
        """
        Iterates over Actigraph.csv in user_dir, joins with computed RMSSD from RR.csv,
        and yields TelemetryPayload instances.
        """
        if patient_id is None:
            patient_id = os.path.basename(os.path.normpath(user_dir))

        act_path = os.path.join(user_dir, "Actigraph.csv")
        if not os.path.exists(act_path):
            raise FileNotFoundError(f"Actigraph.csv not found in {user_dir}")

        # Compute HRV RMSSD mapping from RR.csv
        rmssd_map = self.compute_user_rmssd_map(user_dir)
        default_rmssd = rmssd_map.get('__default__', 50.0)

        with open(act_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            count = 0
            for idx, row in enumerate(reader):
                if idx < start_row:
                    continue
                if limit is not None and count >= limit:
                    break

                try:
                    day = int(row.get("day", 1))
                    time_str = row.get("time", "00:00:00")
                    ts = self._parse_time(day, time_str)

                    hr = float(row.get("HR") or 0.0)
                    
                    # Inclinometer/acceleration magnitude:
                    # In Actigraph.csv, Vector Magnitude is sqrt(Axis1^2 + Axis2^2 + Axis3^2)
                    accel_raw = float(row.get("Vector Magnitude") or row.get("Axis1") or 0.0)
                    
                    # For physiological interpretation in g-force:
                    # An inactive/resting magnitude is 0.0, normal movement is higher
                    accel_mag = accel_raw

                    # Lookup RMSSD for this (day, time)
                    rmssd = float(rmssd_map.get((day, time_str), default_rmssd))
                    if np.isnan(rmssd):
                        rmssd = default_rmssd

                    payload = TelemetryPayload(
                        patient_id=patient_id,
                        timestamp=ts,
                        heart_rate=round(hr, 2),
                        hrv_rmssd=round(rmssd, 2),
                        accelerometer_magnitude=round(accel_mag, 2),
                        channel_source="wearable_stream"
                    )
                    count += 1
                    yield payload
                except Exception as err:
                    continue

    def load_patient_reports(self, user_dir: str, patient_id: Optional[str] = None) -> List[PatientReportPayload]:
        """
        Parses Activity.csv, sleep.csv, and questionnaire.csv into Channel 2 PatientReportPayload events.
        """
        if patient_id is None:
            patient_id = os.path.basename(os.path.normpath(user_dir))

        reports: List[PatientReportPayload] = []

        # 1. Parse Activity.csv
        activity_path = os.path.join(user_dir, "Activity.csv")
        if os.path.exists(activity_path):
            with open(activity_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        act_code = int(row.get("Activity", 0))
                        start_time = row.get("Start", "00:00")
                        end_time = row.get("End", "00:00")
                        day = int(row.get("Day", 1))
                        
                        act_name = ACTIVITY_MAP.get(act_code, f"Activity code {act_code}")
                        ts = self._parse_time(day, f"{start_time}:00")
                        
                        # Severity rating heuristic based on activity exertion / potential trigger
                        if act_code in (10, 11, 12):  # Coffee, smoke, alcohol
                            severity = 6
                            text = f"Patient logged consumption: {act_name} between {start_time} and {end_time}."
                        elif act_code == 6:  # Heavy movement
                            severity = 7
                            text = f"Patient engaged in strenuous physical exertion: {act_name} from {start_time} to {end_time}."
                        else:
                            severity = 4
                            text = f"Patient activity logged: {act_name} ({start_time} - {end_time})."

                        reports.append(PatientReportPayload(
                            patient_id=patient_id,
                            timestamp=ts,
                            raw_text=text,
                            perceived_severity=severity,
                            channel_source="patient_chat"
                        ))
                    except Exception:
                        continue

        # 2. Parse questionnaire.csv
        quest_path = os.path.join(user_dir, "questionnaire.csv")
        if os.path.exists(quest_path):
            with open(quest_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        daily_stress = float(row.get("Daily_stress", 0))
                        stai1 = float(row.get("STAI1", 0))
                        pittsburgh = float(row.get("Pittsburgh", 0))
                        ts = self._parse_time(1, "20:00:00")
                        
                        # Normalized 1-10 severity scale from daily stress (range ~0-30)
                        severity = min(10, max(1, int(round(daily_stress / 3.0))))
                        text = (f"Patient self-reported daily stress level: {daily_stress:.0f}/30. "
                                f"Anxiety STAI1 score: {stai1:.0f}, Pittsburgh sleep quality: {pittsburgh:.0f}.")
                        
                        reports.append(PatientReportPayload(
                            patient_id=patient_id,
                            timestamp=ts,
                            raw_text=text,
                            perceived_severity=severity,
                            channel_source="patient_chat"
                        ))
                    except Exception:
                        continue

        # 3. Parse sleep.csv
        sleep_path = os.path.join(user_dir, "sleep.csv")
        if os.path.exists(sleep_path):
            with open(sleep_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        in_bed_day = int(row.get("In Bed Date", 2))
                        in_bed_time = row.get("In Bed Time", "00:00")
                        out_bed_time = row.get("Out Bed Time", "07:00")
                        efficiency = float(row.get("Efficiency", 0))
                        frag_index = float(row.get("Sleep Fragmentation Index", 0))
                        awakenings = int(float(row.get("Number of Awakenings", 0)))
                        
                        ts = self._parse_time(in_bed_day, f"{out_bed_time}:00")
                        severity = 7 if (efficiency < 85 or frag_index > 15) else 3
                        text = (f"Sleep diary logged: Slept from {in_bed_time} to {out_bed_time}. "
                                f"Sleep efficiency: {efficiency:.1f}%, awakenings: {awakenings}, "
                                f"fragmentation index: {frag_index:.1f}.")

                        reports.append(PatientReportPayload(
                            patient_id=patient_id,
                            timestamp=ts,
                            raw_text=text,
                            perceived_severity=severity,
                            channel_source="patient_chat"
                        ))
                    except Exception:
                        continue

        # Sort all reports chronologically
        reports.sort(key=lambda r: r.timestamp)
        return reports

    def generate_doctor_queries(self, user_dir: str, patient_id: Optional[str] = None) -> List[DoctorQueryPayload]:
        """
        Generates clinical queries for Channel 3 based on user metadata.
        """
        if patient_id is None:
            patient_id = os.path.basename(os.path.normpath(user_dir))

        user_info_path = os.path.join(user_dir, "user_info.csv")
        age, gender = "unknown", "unknown"
        if os.path.exists(user_info_path):
            with open(user_info_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    age = row.get("Age", "unknown")
                    gender = row.get("Gender", "unknown")
                    break

        base_ts = self._parse_time(1, "12:00:00")

        queries = [
            DoctorQueryPayload(
                doctor_id="dr_cardio_01",
                patient_id=patient_id,
                timestamp=base_ts,
                query_text=(f"Audit telemetry for patient {patient_id} (Age: {age}, Gender: {gender}): "
                            f"Did the patient experience resting tachycardia spikes (HR > 120 with low motion) on Day 1?"),
                channel_source="doctor_portal"
            ),
            DoctorQueryPayload(
                doctor_id="dr_sleep_02",
                patient_id=patient_id,
                timestamp=self._parse_time(2, "08:30:00"),
                query_text=(f"Review sleep quality and nocturnal HRV dynamics for patient {patient_id}. "
                            f"Correlate awakenings with heart rate elevation."),
                channel_source="doctor_portal"
            )
        ]
        return queries
