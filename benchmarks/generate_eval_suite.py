"""
Synthesis and Validation Engine for MMASH Grounded Clinical Benchmark Suite.
Generates `benchmarks/dataset/mmash_grounded_eval_suite.json` with 66 grounded sessions
across 22 MMASH subjects (user_1 through user_22) and 5 distinct evaluation categories.
"""

import os
import sys
import json
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.dataset_helper import extract_mmash_user_profiles

def build_iso_timestamp(base_date: datetime, day: int, time_str: str, offset_seconds: int = 0) -> str:
    """Calculates ISO timestamp with timezone."""
    parts = time_str.split(":")
    h = int(parts[0])
    m = int(parts[1])
    s = int(parts[2]) if len(parts) > 2 else 0
    dt = base_date + timedelta(days=day - 1, hours=h, minutes=m, seconds=s + offset_seconds)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

def generate_benchmark_suite():
    base_date = datetime(2026, 8, 23, 0, 0, 0, tzinfo=timezone.utc)
    profiles = extract_mmash_user_profiles()

    sessions = []

    # -------------------------------------------------------------------------
    # Category 1: implicit_reference (20 sessions: user_1 to user_20)
    # -------------------------------------------------------------------------
    implicit_specs = [
        # (user, pronoun_med, canonical_med, rx_code, pronoun_sym, can_sym, sym_code, focus)
        ("user_1", "that pill", "Metoprolol", "RxNorm:6918", "my usual fluttering", "Palpitations", "SNOMED:80313002", "medication_and_symptom"),
        ("user_2", "the morning dose", "Lisinopril", "RxNorm:29046", "this lightheaded spell", "Lightheadedness", "SNOMED:386705008", "adherence_and_hypotension"),
        ("user_3", "my little white tablet", "Metoprolol", "RxNorm:6918", "the side effect", "Dizziness", "SNOMED:25064002", "symptom_inquiry"),
        ("user_4", "the blood pressure medicine", "Amlodipine", "RxNorm:17767", "that ankle swelling", "Edema", "SNOMED:267038008", "adverse_effect_referral"),
        ("user_5", "the beta blocker", "Propranolol", "RxNorm:8787", "this racing sensation", "Palpitations", "SNOMED:80313002", "situational_anxiety"),
        ("user_6", "that nighttime pill", "Diltiazem", "RxNorm:3443", "the irregular beat", "Atrial Fibrillation", "SNOMED:49436004", "rhythm_management"),
        ("user_7", "my morning tablet", "Metoprolol", "RxNorm:6918", "that dizzy feeling", "Dizziness", "SNOMED:25064002", "positional_tachycardia"),
        ("user_8", "that cardiac capsule", "Carvedilol", "RxNorm:20352", "this chest tightness", "Chest Tightness", "SNOMED:29857009", "hemodynamic_status"),
        ("user_9", "the morning dose", "Metoprolol", "RxNorm:6918", "that rapid rhythm", "Tachycardia", "SNOMED:3424008", "rate_control"),
        ("user_10", "the daily blocker", "Atenolol", "RxNorm:1202", "that constant fatigue", "Fatigue", "SNOMED:84229001", "fatigue_correlation"),
        ("user_11", "my rhythm medication", "Verapamil", "RxNorm:11170", "that sudden flutter", "Palpitations", "SNOMED:80313002", "antiarrhythmic_check"),
        ("user_12", "that little pill", "Amlodipine", "RxNorm:17767", "my blood pressure spikes", "Hypertension", "SNOMED:59621000", "vasodilator_effect"),
        ("user_13", "the night dose", "Propranolol", "RxNorm:8787", "that sleep disruption", "Sleep Disorder", "SNOMED:39898005", "nocturnal_adherence"),
        ("user_14", "my anxiety pill", "Buspirone", "RxNorm:1827", "this heart pounding", "Palpitations", "SNOMED:80313002", "somatic_anxiety"),
        ("user_15", "the daily tablet", "Bisoprolol", "RxNorm:197379", "that lightheadedness", "Lightheadedness", "SNOMED:386705008", "autonomic_regulation"),
        ("user_16", "the morning heart pill", "Metoprolol", "RxNorm:6918", "that breathing trouble", "Dyspnea", "SNOMED:267036007", "sleep_apnea_hypoxia"),
        ("user_17", "my blood thinner", "Warfarin", "RxNorm:11289", "the rhythm flutter", "Palpitations", "SNOMED:80313002", "anticoagulation_adherence"),
        ("user_18", "that heart pill", "Carvedilol", "RxNorm:20352", "this extreme exhaustion", "Fatigue", "SNOMED:84229001", "hemodynamic_tolerance"),
        ("user_19", "the evening blocker", "Propranolol", "RxNorm:8787", "that sudden jump in heart rate", "Tachycardia", "SNOMED:3424008", "orthostatic_surge"),
        ("user_20", "my combined tablet", "Labetalol", "RxNorm:6130", "that pounding in my ears", "Palpitations", "SNOMED:80313002", "blood_pressure_spike")
    ]

    # 7 distinct dialogue templates representing diverse clinical interaction archetypes
    dialogue_templates = [
        # Template 1: Standard Temporal Ingestion & Symptom Onset
        {
            "id": "template_01",
            "subtype": "standard_anaphora",
            "turns_fn": lambda p_med, c_med, r_code, p_sym, c_sym, s_code, t0, t1, t2, t3: [
                {
                    "turn_index": 1,
                    "channel": "patient_chat",
                    "speaker": "patient",
                    "text": f"Hello, I took {p_med} around breakfast today, but about thirty minutes later I noticed {p_sym}.",
                    "timestamp": t0
                },
                {
                    "turn_index": 2,
                    "channel": "patient_chat",
                    "speaker": "agent",
                    "text": f"Understood. Reviewing your profile, you are referencing your prescribed {c_med} ({r_code}). To assist you accurately, is the {c_sym.lower()} accompanied by dizziness, chest tightness, or shortness of breath?",
                    "timestamp": t1
                },
                {
                    "turn_index": 3,
                    "channel": "patient_chat",
                    "speaker": "patient",
                    "text": f"No chest pain, but {p_sym} feels identical to previous episodes when my rate went up.",
                    "timestamp": t2
                },
                {
                    "turn_index": 4,
                    "channel": "patient_chat",
                    "speaker": "agent",
                    "text": f"Thank you for clarifying. I have linked this report of {c_sym} to your active {c_med} baseline in your clinical graph. Please remain seated and monitor your vitals.",
                    "timestamp": t3
                }
            ]
        },
        # Template 2: Temporal Chain & Delay Effect
        {
            "id": "template_02",
            "subtype": "temporal_chain",
            "turns_fn": lambda p_med, c_med, r_code, p_sym, c_sym, s_code, t0, t1, t2, t3: [
                {
                    "turn_index": 1,
                    "channel": "patient_chat",
                    "speaker": "patient",
                    "text": f"Since I took {p_med} earlier this morning, I've had {p_sym} persisting for over an hour.",
                    "timestamp": t0
                },
                {
                    "turn_index": 2,
                    "channel": "patient_chat",
                    "speaker": "agent",
                    "text": f"I have retrieved your active prescription for {c_med} ({r_code}). How severe is {c_sym.lower()} on a scale of 1 to 10, and do you feel short of breath?",
                    "timestamp": t1
                },
                {
                    "turn_index": 3,
                    "channel": "patient_chat",
                    "speaker": "patient",
                    "text": f"It's around a 5 out of 10. {p_sym} seems to flare up mainly when standing.",
                    "timestamp": t2
                },
                {
                    "turn_index": 4,
                    "channel": "patient_chat",
                    "speaker": "agent",
                    "text": f"Recorded. Linking {c_sym} to {c_med} with temporal onset marker in graph memory. Please rest in a seated position while we monitor your wearable vitals.",
                    "timestamp": t3
                }
            ]
        },
        # Template 3: Adherence Timing & Recurrence
        {
            "id": "template_03",
            "subtype": "adherence_inquiry",
            "turns_fn": lambda p_med, c_med, r_code, p_sym, c_sym, s_code, t0, t1, t2, t3: [
                {
                    "turn_index": 1,
                    "channel": "patient_chat",
                    "speaker": "patient",
                    "text": f"I swallowed {p_med} with water about twenty minutes ago, but now {p_sym} has started up again.",
                    "timestamp": t0
                },
                {
                    "turn_index": 2,
                    "channel": "patient_chat",
                    "speaker": "agent",
                    "text": f"Cross-referencing your profile: your {c_med} ({r_code}) dose was scheduled for this morning. Are you experiencing any vision changes or radiating discomfort alongside the {c_sym.lower()}?",
                    "timestamp": t1
                },
                {
                    "turn_index": 3,
                    "channel": "patient_chat",
                    "speaker": "patient",
                    "text": f"No vision changes, just {p_sym} in my chest like last week.",
                    "timestamp": t2
                },
                {
                    "turn_index": 4,
                    "channel": "patient_chat",
                    "speaker": "agent",
                    "text": f"Understood. Updated your clinical graph associating {c_sym} with your current {c_med} regimen. Please keep your telemetry sensor on.",
                    "timestamp": t3
                }
            ]
        },
        # Template 4: Dose Verification & Symptom Attribution
        {
            "id": "template_04",
            "subtype": "symptom_attribution",
            "turns_fn": lambda p_med, c_med, r_code, p_sym, c_sym, s_code, t0, t1, t2, t3: [
                {
                    "turn_index": 1,
                    "channel": "patient_chat",
                    "speaker": "patient",
                    "text": f"Just following up on {p_med} from earlier—could it be responsible for {p_sym} that I am feeling right now?",
                    "timestamp": t0
                },
                {
                    "turn_index": 2,
                    "channel": "patient_chat",
                    "speaker": "agent",
                    "text": f"Analyzing your medication profile for {c_med} ({r_code}). {c_sym} is a documented clinical finding. Did {p_sym} begin immediately after taking it?",
                    "timestamp": t1
                },
                {
                    "turn_index": 3,
                    "channel": "patient_chat",
                    "speaker": "patient",
                    "text": f"Yes, {p_sym} started approximately forty minutes after taking {p_med}.",
                    "timestamp": t2
                },
                {
                    "turn_index": 4,
                    "channel": "patient_chat",
                    "speaker": "agent",
                    "text": f"Documenting this temporal correlation between {c_med} ingestion and {c_sym} in your personal graph memory for physician review.",
                    "timestamp": t3
                }
            ]
        },
        # Template 5: Negation & Breakthrough Symptom
        {
            "id": "template_05",
            "subtype": "negation_adherence",
            "turns_fn": lambda p_med, c_med, r_code, p_sym, c_sym, s_code, t0, t1, t2, t3: [
                {
                    "turn_index": 1,
                    "channel": "patient_chat",
                    "speaker": "patient",
                    "text": f"I didn't skip {p_med} today, but I still ended up getting {p_sym} this afternoon.",
                    "timestamp": t0
                },
                {
                    "turn_index": 2,
                    "channel": "patient_chat",
                    "speaker": "agent",
                    "text": f"Acknowledged. Despite verified compliance with {c_med} ({r_code}), you are reporting breakthrough {c_sym.lower()}. Are your vitals elevated?",
                    "timestamp": t1
                },
                {
                    "turn_index": 3,
                    "channel": "patient_chat",
                    "speaker": "patient",
                    "text": f"My pulse feels fast and {p_sym} is making it hard to concentrate.",
                    "timestamp": t2
                },
                {
                    "turn_index": 4,
                    "channel": "patient_chat",
                    "speaker": "agent",
                    "text": f"Logging breakthrough {c_sym} episode on {c_med} therapy into your graph context. Monitoring your live heart rate stream now.",
                    "timestamp": t3
                }
            ]
        },
        # Template 6: Recurrent Anaphora & Multi-turn Tracking
        {
            "id": "template_06",
            "subtype": "multi_session_anaphora",
            "turns_fn": lambda p_med, c_med, r_code, p_sym, c_sym, s_code, t0, t1, t2, t3: [
                {
                    "turn_index": 1,
                    "channel": "patient_chat",
                    "speaker": "patient",
                    "text": f"Checking in about {p_med}. Every time I take it, {p_sym} seems to follow.",
                    "timestamp": t0
                },
                {
                    "turn_index": 2,
                    "channel": "patient_chat",
                    "speaker": "agent",
                    "text": f"Looking up your graph memory: you are referring to {c_med} ({r_code}). Is this recurrent pattern of {c_sym.lower()} happening with every daily dose?",
                    "timestamp": t1
                },
                {
                    "turn_index": 3,
                    "channel": "patient_chat",
                    "speaker": "patient",
                    "text": f"Yes, exactly. {p_sym} usually peaks within an hour after {p_med}.",
                    "timestamp": t2
                },
                {
                    "turn_index": 4,
                    "channel": "patient_chat",
                    "speaker": "agent",
                    "text": f"Graph node updated with recurrent adverse correlation between {c_med} and {c_sym}. Flagging pattern for your clinical care team.",
                    "timestamp": t3
                }
            ]
        },
        # Template 7: Physical Exertion Correlation
        {
            "id": "template_07",
            "subtype": "activity_exertion_correlation",
            "turns_fn": lambda p_med, c_med, r_code, p_sym, c_sym, s_code, t0, t1, t2, t3: [
                {
                    "turn_index": 1,
                    "channel": "patient_chat",
                    "speaker": "patient",
                    "text": f"After taking {p_med} this morning, I walked up the stairs and immediately noticed {p_sym}.",
                    "timestamp": t0
                },
                {
                    "turn_index": 2,
                    "channel": "patient_chat",
                    "speaker": "agent",
                    "text": f"Checking your graph baseline: you are taking {c_med} ({r_code}). Did {p_sym} resolve once you rested, or is it continuing?",
                    "timestamp": t1
                },
                {
                    "turn_index": 3,
                    "channel": "patient_chat",
                    "speaker": "patient",
                    "text": f"It calmed down a bit with rest, but {p_sym} is still noticeable.",
                    "timestamp": t2
                },
                {
                    "turn_index": 4,
                    "channel": "patient_chat",
                    "speaker": "agent",
                    "text": f"Logged exertional correlation for {c_sym} under active {c_med} protocol. Continuing background telemetry evaluation.",
                    "timestamp": t3
                }
            ]
        }
    ]

    for idx, spec in enumerate(implicit_specs, start=1):
        u, pronoun_med, can_med, rx_code, pronoun_sym, can_sym, sym_code, focus = spec
        p = profiles[u]
        u_clean = u.replace("_", "")
        sess_id = f"eval_{u_clean}_sess_01"
        base_t = "11:00:00"
        t0 = build_iso_timestamp(base_date, 1, base_t, 0)
        t1 = build_iso_timestamp(base_date, 1, base_t, 18)
        t2 = build_iso_timestamp(base_date, 1, base_t, 42)
        t3 = build_iso_timestamp(base_date, 1, base_t, 65)

        # Cycle through the 7 templates deterministically
        tpl = dialogue_templates[(idx - 1) % len(dialogue_templates)]
        turns = tpl["turns_fn"](pronoun_med, can_med, rx_code, pronoun_sym, can_sym, sym_code, t0, t1, t2, t3)

        session_obj = {
            "session_id": sess_id,
            "patient_id": u,
            "category": "implicit_reference",
            "dialogue_template_id": tpl["id"],
            "case_subtype": tpl["subtype"],
            "ehr_baseline": {
                "demographics": p["demographics"],
                "conditions": [p["ehr_baseline"]["condition"]],
                "prescriptions": [p["ehr_baseline"]["prescription"], p["ehr_baseline"]["secondary_rx"]],
                "threshold_protocols": {
                    "tachycardia_threshold_hr": p["ehr_baseline"]["tachy_threshold_hr"],
                    "inactivity_accel_max_g": 0.1
                },
                "psychophysiological_baseline": p["psychophysiological"]
            },
            "ground_truth_entities": [
                {"raw_text": pronoun_med, "canonical_name": can_med, "code": rx_code},
                {"raw_text": pronoun_sym, "canonical_name": can_sym, "code": sym_code}
            ],
            "ground_truth_discrepancy": {
                "detected": False
            },
            "turns": turns
        }
        sessions.append(session_obj)

    # -------------------------------------------------------------------------
    # Category 2: epistemic_discrepancy (16 sessions: user_1 to user_16)
    # -------------------------------------------------------------------------
    discrepancy_specs = [
        ("user_1", "blood pressure pill", "Metoprolol", "RxNorm:6918", "UNVERIFIED_ADHERENCE_STOP", "I stopped taking my blood pressure pill yesterday because it made me feel sluggish."),
        ("user_2", "morning ACE inhibitor", "Lisinopril", "RxNorm:29046", "UNVERIFIED_ADHERENCE_STOP", "I quit taking my morning ACE inhibitor two days ago because it gave me a dry cough."),
        ("user_3", "daily beta blocker", "Metoprolol", "RxNorm:6918", "UNVERIFIED_DOSAGE_CHANGE", "I cut my daily beta blocker dose in half last night because my pulse felt too slow."),
        ("user_4", "calcium channel pill", "Amlodipine", "RxNorm:17767", "UNVERIFIED_ADHERENCE_STOP", "I stopped taking my calcium channel pill over the weekend after my feet swelled up."),
        ("user_5", "propranolol tablets", "Propranolol", "RxNorm:8787", "UNVERIFIED_ADHERENCE_STOP", "I decided to stop taking the propranolol tablets three days ago since I felt calmer."),
        ("user_6", "diltiazem medication", "Diltiazem", "RxNorm:3443", "UNVERIFIED_DOSAGE_CHANGE", "I skipped my diltiazem medication yesterday and only took a quarter dose this morning."),
        ("user_7", "morning metoprolol", "Metoprolol", "RxNorm:6918", "UNVERIFIED_ADHERENCE_STOP", "I halted my morning metoprolol on Tuesday because I felt unsteady standing up."),
        ("user_8", "carvedilol prescription", "Carvedilol", "RxNorm:20352", "UNVERIFIED_ADHERENCE_STOP", "I stopped taking my carvedilol prescription two days ago because of persistent fatigue."),
        ("user_9", "extended-release beta blocker", "Metoprolol", "RxNorm:6918", "UNVERIFIED_DOSAGE_CHANGE", "I decreased my extended-release beta blocker from 100mg to 50mg without a new prescription."),
        ("user_10", "atenolol pills", "Atenolol", "RxNorm:1202", "UNVERIFIED_ADHERENCE_STOP", "I completely stopped my atenolol pills yesterday after reading about side effects."),
        ("user_11", "verapamil capsules", "Verapamil", "RxNorm:11170", "UNVERIFIED_ADHERENCE_STOP", "I ceased taking my verapamil capsules three days ago as my rhythm felt normal."),
        ("user_12", "amlodipine tablet", "Amlodipine", "RxNorm:17767", "UNVERIFIED_DOSAGE_CHANGE", "I have been taking my amlodipine tablet only every other day to make the refill last."),
        ("user_13", "nighttime propranolol", "Propranolol", "RxNorm:8787", "UNVERIFIED_ADHERENCE_STOP", "I discontinued the nighttime propranolol on Monday because it gave me vivid dreams."),
        ("user_14", "metoprolol tartrate", "Metoprolol", "RxNorm:6918", "UNVERIFIED_ADHERENCE_STOP", "I stopped taking my metoprolol tartrate doses entirely over the last forty-eight hours."),
        ("user_15", "bisoprolol medication", "Bisoprolol", "RxNorm:197379", "UNVERIFIED_DOSAGE_CHANGE", "I halved my bisoprolol medication starting yesterday because I was feeling drowsy."),
        ("user_16", "morning metoprolol succinate", "Metoprolol", "RxNorm:6918", "UNVERIFIED_ADHERENCE_STOP", "I stopped taking my morning metoprolol succinate three days ago.")
    ]

    for idx, spec in enumerate(discrepancy_specs, start=1):
        u, raw_med, can_med, rx_code, disc_type, patient_statement = spec
        p = profiles[u]
        u_clean = u.replace("_", "")
        sess_id = f"eval_{u_clean}_sess_02"
        base_t = "14:20:00"
        t0 = build_iso_timestamp(base_date, 1, base_t, 0)
        t1 = build_iso_timestamp(base_date, 1, base_t, 15)
        t2 = build_iso_timestamp(base_date, 1, base_t, 45)
        t3 = build_iso_timestamp(base_date, 1, base_t, 70)

        turns = [
            {
                "turn_index": 1,
                "channel": "patient_chat",
                "speaker": "patient",
                "text": patient_statement,
                "timestamp": t0
            },
            {
                "turn_index": 2,
                "channel": "patient_chat",
                "speaker": "agent",
                "text": f"Thank you for informing me. I have logged an epistemic discrepancy ({disc_type}) regarding your {can_med} ({rx_code}) for your physician's review. Your active EHR prescription baseline remains recorded. Please consult your physician before altering dosage.",
                "timestamp": t1
            },
            {
                "turn_index": 3,
                "channel": "patient_chat",
                "speaker": "patient",
                "text": f"Understood. Will this change my prescription record in the hospital system automatically?",
                "timestamp": t2
            },
            {
                "turn_index": 4,
                "channel": "patient_chat",
                "speaker": "agent",
                "text": f"No, your official EHR prescription for {can_med} remains active. The system preserves verified clinical baselines while logging your self-reported cessation as an unverified discrepancy edge for clinician audit.",
                "timestamp": t3
            }
        ]

        session_obj = {
            "session_id": sess_id,
            "patient_id": u,
            "category": "epistemic_discrepancy",
            "case_subtype": "unverified_cessation" if "STOP" in disc_type else "unverified_dosage_modification",
            "ehr_baseline": {
                "demographics": p["demographics"],
                "conditions": [p["ehr_baseline"]["condition"]],
                "prescriptions": [p["ehr_baseline"]["prescription"], p["ehr_baseline"]["secondary_rx"]],
                "threshold_protocols": {
                    "tachycardia_threshold_hr": p["ehr_baseline"]["tachy_threshold_hr"],
                    "inactivity_accel_max_g": 0.1
                },
                "psychophysiological_baseline": p["psychophysiological"]
            },
            "ground_truth_entities": [
                {"raw_text": raw_med, "canonical_name": can_med, "code": rx_code}
            ],
            "ground_truth_discrepancy": {
                "detected": True,
                "type": disc_type,
                "target_node_id": "rx_01",
                "target_entity": can_med,
                "ehr_baseline_status": "ACTIVE"
            },
            "turns": turns
        }
        sessions.append(session_obj)

    # -------------------------------------------------------------------------
    # Category 3: telemetry_fusion (13 sessions: user_1..user_7 session 03, user_17..user_22 session 02)
    # -------------------------------------------------------------------------
    fusion_users = ["user_1", "user_2", "user_3", "user_4", "user_5", "user_6", "user_7",
                    "user_17", "user_18", "user_19", "user_20", "user_21", "user_22"]

    for idx, u in enumerate(fusion_users, start=1):
        p = profiles[u]
        anchor = p["telemetry_anchor"]
        hr_val = anchor["hr"]
        accel_val = anchor["accel_g"]
        day_val = anchor["day"]
        time_val = anchor["time"]
        
        anomaly_iso = build_iso_timestamp(base_date, day_val, time_val, 0)
        report_iso = build_iso_timestamp(base_date, day_val, time_val, 120)
        agent_iso = build_iso_timestamp(base_date, day_val, time_val, 135)
        followup_iso = build_iso_timestamp(base_date, day_val, time_val, 180)
        agent_resp_iso = build_iso_timestamp(base_date, day_val, time_val, 195)

        event_id = f"anomaly_{u}_{time_val.replace(':', '')}"
        sess_num = "03" if int(u.split("_")[1]) <= 7 else "02"
        u_clean = u.replace("_", "")
        sess_id = f"eval_{u_clean}_sess_{sess_num}"

        can_cond = p["ehr_baseline"]["condition"]["name"]
        can_rx = p["ehr_baseline"]["prescription"]["name"]

        turns = [
            {
                "turn_index": 1,
                "channel": "wearable_anomaly",
                "speaker": "wearable_stream",
                "text": f"GATED_TELEMETRY_ANOMALY: Patient {u} exhibited resting tachycardia spike (HR={hr_val:.1f} BPM, Accel={accel_val:.2f}g) exceeding threshold {p['ehr_baseline']['tachy_threshold_hr']:.0f} BPM during low physical exertion.",
                "timestamp": anomaly_iso
            },
            {
                "turn_index": 2,
                "channel": "patient_chat",
                "speaker": "patient",
                "text": f"I just felt my heart suddenly fluttering and beating very hard in my chest while sitting down reading. It started a couple minutes ago.",
                "timestamp": report_iso
            },
            {
                "turn_index": 3,
                "channel": "patient_chat",
                "speaker": "agent",
                "text": f"I have correlated your symptom report directly with the telemetry alert registered two minutes prior (Heart Rate: {hr_val:.1f} BPM, resting acceleration: {accel_val:.2f}g). Creating a TRIGGERED_BY link between your report and telemetry anomaly in clinical graph memory. Please sit upright and rest.",
                "timestamp": agent_iso
            },
            {
                "turn_index": 4,
                "channel": "patient_chat",
                "speaker": "patient",
                "text": f"The fluttering is slowing down slightly now. Should I take an extra dose of {can_rx}?",
                "timestamp": followup_iso
            },
            {
                "turn_index": 5,
                "channel": "patient_chat",
                "speaker": "agent",
                "text": f"Do not take unauthorized additional doses of {can_rx}. I am continuing to monitor your vitals stream. If you develop chest pressure, radiation of pain, or severe shortness of breath, emergency escalation will trigger immediately.",
                "timestamp": agent_resp_iso
            }
        ]

        session_obj = {
            "session_id": sess_id,
            "patient_id": u,
            "category": "telemetry_fusion",
            "case_subtype": "resting_tachycardia_crossmodal_fusion",
            "ehr_baseline": {
                "demographics": p["demographics"],
                "conditions": [p["ehr_baseline"]["condition"]],
                "prescriptions": [p["ehr_baseline"]["prescription"], p["ehr_baseline"]["secondary_rx"]],
                "threshold_protocols": {
                    "tachycardia_threshold_hr": p["ehr_baseline"]["tachy_threshold_hr"],
                    "inactivity_accel_max_g": 0.1
                },
                "psychophysiological_baseline": p["psychophysiological"]
            },
            "ground_truth_entities": [
                {"raw_text": "heart suddenly fluttering", "canonical_name": "Palpitations", "code": "SNOMED:80313002"},
                {"raw_text": can_rx, "canonical_name": can_rx, "code": p["ehr_baseline"]["prescription"]["code"]}
            ],
            "ground_truth_discrepancy": {
                "detected": False
            },
            "ground_truth_telemetry_link": {
                "anomaly_event_id": event_id,
                "relation": "TRIGGERED_BY",
                "heart_rate": hr_val,
                "accelerometer_magnitude": accel_val,
                "timestamp": anomaly_iso,
                "time_delta_seconds": 120
            },
            "turns": turns
        }
        sessions.append(session_obj)

    # -------------------------------------------------------------------------
    # Category 4: distractor_noise (10 sessions: user_8..user_16 session 03, user_21 session 01)
    # -------------------------------------------------------------------------
    distractor_users = ["user_8", "user_9", "user_10", "user_11", "user_12", "user_13", "user_14", "user_15", "user_16", "user_21"]

    for idx, u in enumerate(distractor_users, start=1):
        p = profiles[u]
        sess_num = "01" if u == "user_21" else "03"
        u_clean = u.replace("_", "")
        sess_id = f"eval_{u_clean}_sess_{sess_num}"
        base_t = "15:00:00"

        can_rx = p["ehr_baseline"]["prescription"]["name"]
        rx_code = p["ehr_baseline"]["prescription"]["code"]
        can_cond = p["ehr_baseline"]["condition"]["name"]

        turns = [
            {
                "turn_index": 1,
                "channel": "patient_chat",
                "speaker": "patient",
                "text": f"I took my prescribed {can_rx} on time at 08:00 AM today.",
                "timestamp": build_iso_timestamp(base_date, 1, base_t, 0)
            },
            {
                "turn_index": 2,
                "channel": "patient_chat",
                "speaker": "patient",
                "text": "The weather looks pretty sunny today, might take a short walk in the afternoon.",
                "timestamp": build_iso_timestamp(base_date, 1, base_t, 60)
            },
            {
                "turn_index": 3,
                "channel": "patient_chat",
                "speaker": "agent",
                "text": "A light walk is great for general cardiovascular wellness, provided you avoid overexertion.",
                "timestamp": build_iso_timestamp(base_date, 1, base_t, 75)
            },
            {
                "turn_index": 4,
                "channel": "patient_chat",
                "speaker": "patient",
                "text": "I had oatmeal and a cup of decaf tea for breakfast around 8:30.",
                "timestamp": build_iso_timestamp(base_date, 1, base_t, 140)
            },
            {
                "turn_index": 5,
                "channel": "patient_chat",
                "speaker": "agent",
                "text": "Decaffeinated options are ideal to prevent stimulant-induced tachycardia.",
                "timestamp": build_iso_timestamp(base_date, 1, base_t, 155)
            },
            {
                "turn_index": 6,
                "channel": "patient_chat",
                "speaker": "patient",
                "text": "Also finished watching a documentary on gardening earlier.",
                "timestamp": build_iso_timestamp(base_date, 1, base_t, 220)
            },
            {
                "turn_index": 7,
                "channel": "patient_chat",
                "speaker": "patient",
                "text": "I need to remember to buy groceries later this evening, mainly vegetables and olive oil.",
                "timestamp": build_iso_timestamp(base_date, 1, base_t, 280)
            },
            {
                "turn_index": 8,
                "channel": "patient_chat",
                "speaker": "agent",
                "text": "A heart-healthy Mediterranean diet supports blood pressure regulation.",
                "timestamp": build_iso_timestamp(base_date, 1, base_t, 295)
            },
            {
                "turn_index": 9,
                "channel": "patient_chat",
                "speaker": "patient",
                "text": "My friend called to ask about plans for the weekend soccer match.",
                "timestamp": build_iso_timestamp(base_date, 1, base_t, 360)
            },
            {
                "turn_index": 10,
                "channel": "patient_chat",
                "speaker": "patient",
                "text": f"Wait, I am suddenly experiencing acute dizziness and chest tightness right now.",
                "timestamp": build_iso_timestamp(base_date, 1, base_t, 420)
            },
            {
                "turn_index": 11,
                "channel": "patient_chat",
                "speaker": "agent",
                "text": f"Logging acute symptoms: Dizziness and Chest Tightness in the clinical graph for {u}. Pruning casual conversational filler from active working context. Recommending vitals check and alerting clinical monitor.",
                "timestamp": build_iso_timestamp(base_date, 1, base_t, 435)
            }
        ]

        session_obj = {
            "session_id": sess_id,
            "patient_id": u,
            "category": "distractor_noise",
            "case_subtype": "conversational_filler_pruning",
            "ehr_baseline": {
                "demographics": p["demographics"],
                "conditions": [p["ehr_baseline"]["condition"]],
                "prescriptions": [p["ehr_baseline"]["prescription"], p["ehr_baseline"]["secondary_rx"]],
                "threshold_protocols": {
                    "tachycardia_threshold_hr": p["ehr_baseline"]["tachy_threshold_hr"],
                    "inactivity_accel_max_g": 0.1
                },
                "psychophysiological_baseline": p["psychophysiological"]
            },
            "ground_truth_entities": [
                {"raw_text": can_rx, "canonical_name": can_rx, "code": rx_code},
                {"raw_text": "acute dizziness", "canonical_name": "Dizziness", "code": "SNOMED:25064002"},
                {"raw_text": "chest tightness", "canonical_name": "Chest Tightness", "code": "SNOMED:29857009"}
            ],
            "ground_truth_discrepancy": {
                "detected": False
            },
            "subgraph_evaluation_targets": {
                "max_target_context_tokens": 400,
                "expected_pruned_turns_count": 8,
                "relevant_turn_indices": [1, 10, 11]
            },
            "turns": turns
        }
        sessions.append(session_obj)

    # -------------------------------------------------------------------------
    # Category 5: doctor_audit (7 sessions: user_17..user_20 session 03, user_21 session 03, user_22 session 01 & session 03)
    # -------------------------------------------------------------------------
    doctor_specs = [
        ("user_17", "eval_user17_sess_03", "dr_cardio_01", "Audit telemetry and patient reports for user_17 over past 24h: Did the patient experience resting tachycardia spikes (HR > 120 with low motion) on Day 1, and were there any unverified medication changes?"),
        ("user_18", "eval_user18_sess_03", "dr_sleep_02", "Review sleep quality, nocturnal HRV dynamics, and daytime tachycardia episodes for user_18. Correlate awakenings with heart rate elevation."),
        ("user_19", "eval_user19_sess_03", "dr_cardio_01", "Provide clinical audit for user_19: Summarize autonomic stability, daily stress index correlation with palpitations, and active prescription adherence."),
        ("user_20", "eval_user20_sess_03", "dr_hypertension_03", "Audit blood pressure symptoms and tachycardia protocol violations for user_20 across the 24-hour observation window."),
        ("user_21", "eval_user21_sess_03", "dr_psych_04", "Evaluate psychophysiological strain for user_21: Correlate STAI-Y1 anxiety score with resting tachycardia occurrences and reported somatic flutters."),
        ("user_22", "eval_user22_sess_01", "dr_cardio_01", "Audit cardiometabolic risk and resting heart rate variability for user_22. Summarize all graph-recorded anomalies and medication consistency."),
        ("user_22", "eval_user22_sess_03", "dr_audit_chief", "Executive audit for user_22: Verify if any epistemic discrepancies or threshold breaches were logged in the past 24 hours requiring clinical follow-up.")
    ]

    for spec in doctor_specs:
        u, sess_id, doc_id, query_text = spec
        p = profiles[u]
        anchor = p["telemetry_anchor"]
        can_rx = p["ehr_baseline"]["prescription"]["name"]
        rx_code = p["ehr_baseline"]["prescription"]["code"]
        can_cond = p["ehr_baseline"]["condition"]["name"]
        cond_code = p["ehr_baseline"]["condition"]["code"]

        t0 = build_iso_timestamp(base_date, 2, "09:00:00", 0)
        t1 = build_iso_timestamp(base_date, 2, "09:00:12", 0)

        turns = [
            {
                "turn_index": 1,
                "channel": "doctor_portal",
                "speaker": "doctor",
                "text": query_text,
                "timestamp": t0
            },
            {
                "turn_index": 2,
                "channel": "doctor_portal",
                "speaker": "agent",
                "text": f"Grounded Audit Summary for {u} (Age: {p['demographics']['age']}, Gender: {p['demographics']['gender']}): Active Condition: {can_cond} ({cond_code}), Prescription: {can_rx} ({rx_code}). Telemetry analysis reveals peak resting HR={anchor['hr']:.1f} BPM (Accel={anchor['accel_g']:.2f}g) on Day {anchor['day']} at {anchor['time']}. Sleep efficiency recorded at {p['psychophysiological']['sleep_efficiency_pct']:.1f}% with fragmentation index {p['psychophysiological']['sleep_fragmentation_index']:.1f} and {p['psychophysiological']['number_of_awakenings']} awakenings. Graph memory indicates verified adherence with zero unresolved discrepancies.",
                "timestamp": t1
            }
        ]

        session_obj = {
            "session_id": sess_id,
            "patient_id": u,
            "category": "doctor_audit",
            "case_subtype": "multimodal_clinical_summary",
            "ehr_baseline": {
                "demographics": p["demographics"],
                "conditions": [p["ehr_baseline"]["condition"]],
                "prescriptions": [p["ehr_baseline"]["prescription"], p["ehr_baseline"]["secondary_rx"]],
                "threshold_protocols": {
                    "tachycardia_threshold_hr": p["ehr_baseline"]["tachy_threshold_hr"],
                    "inactivity_accel_max_g": 0.1
                },
                "psychophysiological_baseline": p["psychophysiological"]
            },
            "ground_truth_entities": [
                {"raw_text": can_cond, "canonical_name": can_cond, "code": cond_code},
                {"raw_text": can_rx, "canonical_name": can_rx, "code": rx_code}
            ],
            "ground_truth_discrepancy": {
                "detected": False
            },
            "ground_truth_audit_targets": {
                "doctor_id": doc_id,
                "query_channel": "doctor_portal",
                "includes_telemetry_audit": True,
                "includes_sleep_hrv_audit": True
            },
            "turns": turns
        }
        sessions.append(session_obj)

    return sessions

def main():
    output_path = os.path.join("benchmarks", "dataset", "mmash_grounded_eval_suite.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    sessions = generate_benchmark_suite()

    # Strict Validation Checks
    assert len(sessions) == 66, f"Expected exactly 66 sessions, got {len(sessions)}"
    
    cat_counts = {}
    user_counts = {}
    template_counts = {}
    subtype_counts = {}
    for s in sessions:
        cat = s["category"]
        u = s["patient_id"]
        tpl_id = s.get("dialogue_template_id", "none")
        subtype = s.get("case_subtype", "none")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        user_counts[u] = user_counts.get(u, 0) + 1
        template_counts[tpl_id] = template_counts.get(tpl_id, 0) + 1
        subtype_counts[subtype] = subtype_counts.get(subtype, 0) + 1

    expected_cats = {
        "implicit_reference": 20,
        "epistemic_discrepancy": 16,
        "telemetry_fusion": 13,
        "distractor_noise": 10,
        "doctor_audit": 7
    }

    print("Category Breakdown:")
    for cat, exp in expected_cats.items():
        actual = cat_counts.get(cat, 0)
        print(f"  {cat}: {actual} (expected {exp})")
        assert actual == exp, f"Category {cat} count mismatch: {actual} != {exp}"

    print(f"\nDialogue Template Distribution for Implicit Reference:")
    for tpl_id, count in sorted(template_counts.items()):
        if tpl_id != "none":
            print(f"  {tpl_id}: {count} sessions")

    print(f"\nCase Subtype Distribution across Suite:")
    for sub, count in sorted(subtype_counts.items()):
        print(f"  {sub}: {count} sessions")

    print(f"\nUser Distribution ({len(user_counts)} subjects):")
    for u in sorted(user_counts.keys(), key=lambda x: int(x.split('_')[1])):
        print(f"  {u}: {user_counts[u]} sessions")
        assert user_counts[u] == 3, f"Subject {u} does not have exactly 3 sessions: {user_counts[u]}"

    # Write out formatted JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2)

    print(f"\nSuccessfully generated and validated {len(sessions)} benchmark sessions in {output_path}")

if __name__ == "__main__":
    main()
