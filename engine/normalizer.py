import re
from typing import List, Dict, Optional, Literal, Any
import spacy

from schemas.inputs import NormalizedEntity

# Canonical medical ontology knowledge base
# Maps lowercased medical terms / phrases to canonical representation
CANONICAL_ONTOLOGY: Dict[str, Dict[str, Any]] = {
    # Medications (RxNorm)
    "blood pressure pill": {
        "canonical_name": "Metoprolol",
        "code_type": "RxNorm",
        "code_id": "6918",
        "confidence": 0.95,
        "label": "DRUG"
    },
    "morning pill": {
        "canonical_name": "Cardiovascular Agent",
        "code_type": "RxNorm",
        "code_id": "6918",
        "confidence": 0.90,
        "label": "DRUG"
    },
    "metoprolol": {
        "canonical_name": "Metoprolol",
        "code_type": "RxNorm",
        "code_id": "6918",
        "confidence": 0.99,
        "label": "DRUG"
    },
    "lopressor": {
        "canonical_name": "Metoprolol Tartrate",
        "code_type": "RxNorm",
        "code_id": "6918",
        "confidence": 0.99,
        "label": "DRUG"
    },
    "lisinopril": {
        "canonical_name": "Lisinopril",
        "code_type": "RxNorm",
        "code_id": "29046",
        "confidence": 0.99,
        "label": "DRUG"
    },
    "amlodipine": {
        "canonical_name": "Amlodipine",
        "code_type": "RxNorm",
        "code_id": "17767",
        "confidence": 0.99,
        "label": "DRUG"
    },
    "aspirin": {
        "canonical_name": "Aspirin",
        "code_type": "RxNorm",
        "code_id": "1191",
        "confidence": 0.99,
        "label": "DRUG"
    },
    "beta blocker": {
        "canonical_name": "Beta-Adrenergic Blocker",
        "code_type": "RxNorm",
        "code_id": "1483",
        "confidence": 0.95,
        "label": "DRUG"
    },
    "atorvastatin": {
        "canonical_name": "Atorvastatin",
        "code_type": "RxNorm",
        "code_id": "83367",
        "confidence": 0.99,
        "label": "DRUG"
    },

    # Conditions & Symptoms (SNOMED CT)
    "dizzy": {
        "canonical_name": "Dizziness",
        "code_type": "SNOMED",
        "code_id": "25064002",
        "confidence": 0.98,
        "label": "SYMPTOM"
    },
    "dizziness": {
        "canonical_name": "Dizziness",
        "code_type": "SNOMED",
        "code_id": "25064002",
        "confidence": 0.99,
        "label": "SYMPTOM"
    },
    "lightheaded": {
        "canonical_name": "Lightheadedness",
        "code_type": "SNOMED",
        "code_id": "386705008",
        "confidence": 0.98,
        "label": "SYMPTOM"
    },
    "lightheadedness": {
        "canonical_name": "Lightheadedness",
        "code_type": "SNOMED",
        "code_id": "386705008",
        "confidence": 0.99,
        "label": "SYMPTOM"
    },
    "racing heart": {
        "canonical_name": "Palpitations",
        "code_type": "SNOMED",
        "code_id": "80313002",
        "confidence": 0.98,
        "label": "SYMPTOM"
    },
    "heart racing": {
        "canonical_name": "Palpitations",
        "code_type": "SNOMED",
        "code_id": "80313002",
        "confidence": 0.98,
        "label": "SYMPTOM"
    },
    "palpitations": {
        "canonical_name": "Palpitations",
        "code_type": "SNOMED",
        "code_id": "80313002",
        "confidence": 0.99,
        "label": "SYMPTOM"
    },
    "tachycardia": {
        "canonical_name": "Tachycardia",
        "code_type": "SNOMED",
        "code_id": "3424008",
        "confidence": 0.99,
        "label": "SYMPTOM"
    },
    "tachycardia spikes": {
        "canonical_name": "Tachycardia",
        "code_type": "SNOMED",
        "code_id": "3424008",
        "confidence": 0.98,
        "label": "SYMPTOM"
    },
    "chest pain": {
        "canonical_name": "Chest Pain",
        "code_type": "SNOMED",
        "code_id": "29857009",
        "confidence": 0.99,
        "label": "SYMPTOM"
    },
    "chest tightness": {
        "canonical_name": "Chest Tightness",
        "code_type": "SNOMED",
        "code_id": "29857009",
        "confidence": 0.95,
        "label": "SYMPTOM"
    },
    "shortness of breath": {
        "canonical_name": "Dyspnea",
        "code_type": "SNOMED",
        "code_id": "267036007",
        "confidence": 0.98,
        "label": "SYMPTOM"
    },
    "dyspnea": {
        "canonical_name": "Dyspnea",
        "code_type": "SNOMED",
        "code_id": "267036007",
        "confidence": 0.99,
        "label": "SYMPTOM"
    },
    "fatigue": {
        "canonical_name": "Fatigue",
        "code_type": "SNOMED",
        "code_id": "84229001",
        "confidence": 0.95,
        "label": "SYMPTOM"
    },
    "sleep fragmentation": {
        "canonical_name": "Sleep Disorder",
        "code_type": "SNOMED",
        "code_id": "39898005",
        "confidence": 0.95,
        "label": "SYMPTOM"
    },
    "anxiety": {
        "canonical_name": "Anxiety State",
        "code_type": "SNOMED",
        "code_id": "197480006",
        "confidence": 0.95,
        "label": "SYMPTOM"
    },
    "fluttering": {
        "canonical_name": "Palpitations",
        "code_type": "SNOMED",
        "code_id": "80313002",
        "confidence": 0.98,
        "label": "SYMPTOM"
    },
    "flutter": {
        "canonical_name": "Palpitations",
        "code_type": "SNOMED",
        "code_id": "80313002",
        "confidence": 0.98,
        "label": "SYMPTOM"
    },
    "edema": {
        "canonical_name": "Edema",
        "code_type": "SNOMED",
        "code_id": "267038008",
        "confidence": 0.98,
        "label": "SYMPTOM"
    },
    "ankle swelling": {
        "canonical_name": "Edema",
        "code_type": "SNOMED",
        "code_id": "267038008",
        "confidence": 0.98,
        "label": "SYMPTOM"
    },
    "hypertension": {
        "canonical_name": "Hypertension",
        "code_type": "SNOMED",
        "code_id": "59621000",
        "confidence": 0.99,
        "label": "CONDITION"
    },
    "high blood pressure": {
        "canonical_name": "Hypertension",
        "code_type": "SNOMED",
        "code_id": "59621000",
        "confidence": 0.99,
        "label": "CONDITION"
    },
    "daily stress": {
        "canonical_name": "Stress",
        "code_type": "SNOMED",
        "code_id": "73595000",
        "confidence": 0.90,
        "label": "SYMPTOM"
    },

    # Observations & Lab Concepts (LOINC)
    "heart rate": {
        "canonical_name": "Heart rate",
        "code_type": "LOINC",
        "code_id": "8867-4",
        "confidence": 0.95,
        "label": "OBSERVATION"
    },
    "blood pressure": {
        "canonical_name": "Blood pressure systolic and diastolic",
        "code_type": "LOINC",
        "code_id": "85354-9",
        "confidence": 0.95,
        "label": "OBSERVATION"
    },
    "heart rate variability": {
        "canonical_name": "Heart rate variability",
        "code_type": "LOINC",
        "code_id": "80404-7",
        "confidence": 0.95,
        "label": "OBSERVATION"
    }
}

class ClinicalTextNormalizer:
    """
    Canonical medical entity extractor and normalizer.
    Utilizes ScispaCy/spaCy pipeline and ontological mapping to normalize free-text
    ailments and queries into standardized RxNorm, SNOMED CT, and LOINC entities.
    """
    def __init__(self, spacy_model: Optional[str] = None):
        self.nlp = None
        # Try loading scispacy or user-specified model if available
        if spacy_model:
            try:
                self.nlp = spacy.load(spacy_model)
            except Exception:
                self.nlp = None

        if self.nlp is None:
            # Fallback to high-speed blank English spaCy pipeline with EntityRuler
            self.nlp = spacy.blank("en")
            ruler = self.nlp.add_pipe("entity_ruler")
            patterns = []
            for phrase, meta in CANONICAL_ONTOLOGY.items():
                tokens = phrase.split()
                patterns.append({
                    "label": meta["label"],
                    "pattern": [{"LOWER": t} for t in tokens],
                    "id": phrase
                })
            ruler.add_patterns(patterns)

    def normalize(self, text: str) -> List[NormalizedEntity]:
        """
        Extracts entities from input text and resolves them to canonical medical codes.
        """
        if not text or not text.strip():
            return []

        doc = self.nlp(text)
        entities: List[NormalizedEntity] = []
        seen_keys = set()

        # 1. Match from spaCy entity recognition / EntityRuler
        for ent in doc.ents:
            clean_text = ent.text.strip()
            lower_text = clean_text.lower()
            ent_id = ent.ent_id_ or lower_text

            if ent_id in CANONICAL_ONTOLOGY:
                meta = CANONICAL_ONTOLOGY[ent_id]
                key = (meta["code_type"], meta["code_id"])
                if key not in seen_keys:
                    seen_keys.add(key)
                    entities.append(NormalizedEntity(
                        raw_text=clean_text,
                        canonical_name=meta["canonical_name"],
                        code_type=meta["code_type"],
                        code_id=meta["code_id"],
                        confidence=meta["confidence"]
                    ))

        # 2. Direct phrase fallback check for any missed phrases
        lower_input = text.lower()
        for phrase, meta in CANONICAL_ONTOLOGY.items():
            pattern = r'\b' + re.escape(phrase) + r'\b'
            match = re.search(pattern, lower_input)
            if match:
                key = (meta["code_type"], meta["code_id"])
                if key not in seen_keys:
                    seen_keys.add(key)
                    # Extract original case from text
                    start, end = match.span()
                    raw_substr = text[start:end]
                    entities.append(NormalizedEntity(
                        raw_text=raw_substr,
                        canonical_name=meta["canonical_name"],
                        code_type=meta["code_type"],
                        code_id=meta["code_id"],
                        confidence=meta["confidence"]
                    ))

        return entities
