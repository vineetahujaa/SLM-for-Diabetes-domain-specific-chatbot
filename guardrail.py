import re
from functools import lru_cache
from typing import Optional
from dataclasses import dataclass

from config import (
    AWS_REGION,
    BEDROCK_GUARDRAIL_ID,
    BEDROCK_GUARDRAIL_VERSION,
    BEDROCK_MODERATION_ENABLED,
    BEDROCK_MODERATION_TIMEOUT_SECONDS,
)

@dataclass
class ValidationResult:
    passed: bool
    message: Optional[str] = None


class MedicalResponsePolicy:

    REFUSAL_LINE = (
        "I'm your virtual healthcare professional, and I can only assist with diabetes "
        "and health-related questions."
    )

    _GREETING_PATTERN = re.compile(r"^(hi|hello|hey|dear)\b", re.IGNORECASE)

    _DIABETES_KEYWORDS = {
        "a1c", "adjustment", "autoimmune", "basal", "beta",
        "beta cell", "beta cells", "blood", "blood glucose",
        "blood sugar", "bloodsugar", "bolus", "carb",
        "carb counting", "carbohydrate", "carbohydrates",
        "cgm", "cgms", "continuous glucose monitor",
        "correction factor", "dawn phenomenon", "dextrose",
        "degludec", "detemir", "diabetes", "diabetic", "dka",
        "dpp4", "dose", "dosing", "endocrine", "endocrinologist",
        "fasting", "glp1", "glp-1", "fructosamine", "gastroparesis",
        "gestational", "glargine", "glimepiride", "glipizide",
        "glucose", "glucometer", "glulisine", "aspart", "glyburide",
        "glycaemic", "glycemic", "homa", "hba1c", "honeymoon",
        "hyperglycemia", "hypoglycemia", "insulin to carb ratio",
        "insulin", "insulin pen", "insulin pump", "insulin resistance",
        "insulin sensitivity", "insulinpump", "injection", "injections",
        "ketone", "ketone strip", "ketones", "ketoacidosis", "lente",
        "langerhans", "lipohypertrophy", "lispro", "long-acting insulin",
        "lows", "metformin", "microalbumin", "microvascular", "needles",
        "mounjaro", "nephropathy", "neuropathy", "pen",
        "oral hypoglycemic", "empagliflozin", "dapagliflozin",
        "canagliflozin", "ozempic", "pancreas", "pancreatic",
        "pioglitazone", "prediabetes", "pre-diabetes", "retinopathy",
        "rosiglitazone", "semaglutide", "sensor", "sglt2", "sglt-2",
        "sitagliptin", "sulfonylurea", "symptom", "symptoms",
        "thiazolidinedione", "titration", "time in range", "tir",
        "tirzepatide", "type", "type 1", "type1", "type-1",
        "type 2", "type2", "type-2", "wegovy",
    }

    _GENERAL_HEALTH_KEYWORDS = {
        "albumin", "amputation", "appointment", "bariatric",
        "blood pressure", "bloodwork", "bp", "cardiac",
        "cardiovascular", "cholesterol", "ckd", "clinic",
        "complication", "complications", "diet", "doctor",
        "drug", "drugs", "exercise", "fitness", "foot",
        "footcare", "glycation", "heart", "hypertension",
        "kidney", "labs", "lifestyle", "lipid", "macrovascular",
        "medication", "medications", "microvascular", "monitor",
        "monitoring", "nurse", "nutrition", "obesity", "patient",
        "physician", "renal", "tablet", "therapy", "treatment",
        "weight", "wound", "side effect", "side effects", "adverse",
        "dosage", "dose", "doses", "interact", "interaction",
        "prevalence", "incidence", "mortality", "morbidity", "epidemic",
        "global", "worldwide", "population", "statistics", "burden",
        "trend", "trends", "rate", "rates", "report", "study", "data",
    }

    _TOPIC_KEYWORDS = _DIABETES_KEYWORDS.union(_GENERAL_HEALTH_KEYWORDS)

    _FOLLOW_UP_PRONOUNS = {"it", "its", "they", "them", "those", "that", "this"}

    _FOLLOW_UP_KEYWORDS = {
        "more", "details", "detail", "information", "explain",
        "expand", "elaborate", "continue", "next", "further",
        "clarify", "specifics", "side", "effects", "effect",
        "risks", "risk", "dosage", "dose", "how", "when", "why",
        "often", "frequently", "safe", "safety", "long",
    }

    # Common English stopwords excluded from follow-up overlap matching.
    # Without this, single words like "to", "do", "is", "my" in any prior response
    # would incorrectly classify unrelated queries as medical follow-ups.
    _STOPWORDS = {
        "i", "me", "my", "we", "our", "you", "your", "he", "she", "his", "her",
        "it", "its", "they", "them", "their", "what", "which", "who", "whom",
        "this", "that", "these", "those", "a", "an", "the", "and", "but", "or",
        "nor", "for", "yet", "so", "at", "by", "in", "of", "on", "to", "up",
        "as", "is", "are", "was", "were", "be", "been", "being", "have", "has",
        "had", "do", "does", "did", "will", "would", "could", "should", "may",
        "might", "shall", "can", "not", "no", "if", "then", "than", "with",
        "from", "into", "about", "also", "any", "all", "both", "each", "few",
        "more", "most", "other", "some", "such", "own", "same", "just", "how",
        "when", "where", "why", "let", "per", "via",
    }


    def __init__(
        self,
        *,
        max_lines: int = 4,
        max_word_limit: int = 40,
        require_keywords: bool = True,
    ) -> None:
        self.max_lines = max_lines
        self.max_word_limit = max_word_limit
        self.require_keywords = require_keywords

    @staticmethod
    @lru_cache(maxsize=1024)
    def _normalized_tokens(text: str) -> frozenset:
        if not text:
            return frozenset()
        return frozenset(re.findall(r"\w+", text.lower()))

    @staticmethod
    @lru_cache(maxsize=1024)
    def _has_topic_keyword(text: str) -> bool:
        if not text:
            return False
        lowered = text.lower()
        return any(k in lowered for k in MedicalResponsePolicy._TOPIC_KEYWORDS)

    def _is_off_topic_question(self, question: str, llm=None) -> bool:
        # Fast path — known medical keyword present
        if self._has_topic_keyword(question):
            return False
        # Ask the loaded model (max_tokens=3, temp=0) — no keyword lists needed
        if llm is not None:
            try:
                from generator import classify_topic
                return not classify_topic(question, llm)
            except Exception:
                pass
        # Fallback when model not available
        tokens = self._normalized_tokens(question)
        return not tokens.intersection(self._TOPIC_KEYWORDS)

    def _looks_like_follow_up(
        self,
        question: str,
        previous_question: Optional[str],
        previous_response: Optional[str],
    ) -> bool:
        if not question:
            return False
        q_lower = question.lower().strip()
        if not q_lower:
            return False
        if self._has_topic_keyword(question):
            return True
        token_set = self._normalized_tokens(question)
        if self._FOLLOW_UP_PRONOUNS.intersection(token_set):
            return True
        if self._FOLLOW_UP_KEYWORDS.intersection(token_set):
            return True
        follow_up_phrases = (
            "tell me more", "more about", "what about",
            "what else", "how about", "anything else",
            "please continue",
        )
        if any(p in q_lower for p in follow_up_phrases):
            return True
        # Strip stopwords before overlap — prevents common words like "to", "do",
        # "my", "is" in a prior response from falsely tagging harmful queries as
        # medical follow-ups.
        meaningful_tokens = token_set - self._STOPWORDS

        if previous_question:
            prev_tokens = self._normalized_tokens(previous_question) - self._STOPWORDS
            if (
                prev_tokens.intersection(self._TOPIC_KEYWORDS)
                and meaningful_tokens.intersection(prev_tokens)
            ):
                return True
        if previous_response:
            prev_resp_tokens = self._normalized_tokens(previous_response) - self._STOPWORDS
            if (
                prev_resp_tokens.intersection(self._TOPIC_KEYWORDS)
                and meaningful_tokens.intersection(prev_resp_tokens)
            ):
                return True
        return False

    def validate(
        self,
        value: str,
        question: str,
        *,
        previous_question: Optional[str] = None,
        previous_response: Optional[str] = None,
        previous_passed: Optional[bool] = False,
        llm=None,
    ) -> ValidationResult:

        if not question:
            return ValidationResult(passed=False, message="Validator requires the original question.")

        text = value.strip()
        if not text:
            return ValidationResult(passed=False, message="Empty response.")
        if text == self.REFUSAL_LINE:
            return ValidationResult(passed=True)

        is_follow_up = bool(
            previous_passed
            and self._looks_like_follow_up(question, previous_question, previous_response)
        )

        lines              = [l for l in text.splitlines() if l.strip()]
        word_count         = len(re.findall(r"\w+", text))
        response_lower     = text.lower()
        is_off_topic       = self._is_off_topic_question(question, llm=llm)
        question_has_topic = self._has_topic_keyword(question)

        if (
            self.max_word_limit
            and word_count > self.max_word_limit
            and not is_follow_up
            and is_off_topic
        ):
            return ValidationResult(
                passed=False,
                message=f"Response exceeds {self.max_word_limit} words for off-topic query."
            )

        if len(lines) > self.max_lines and not is_follow_up and is_off_topic:
            return ValidationResult(
                passed=False,
                message=f"Response exceeds {self.max_lines} lines for off-topic query."
            )

        if lines and self._GREETING_PATTERN.search(lines[0]):
            return ValidationResult(passed=False, message="Response must not start with a greeting.")

        if is_off_topic and not is_follow_up:
            if question_has_topic:
                is_off_topic = False
            else:
                return ValidationResult(
                    passed=False,
                    message="Off-topic question — refusal line not used."
                )

        if self.require_keywords and not is_follow_up and not question_has_topic:
            if not any(k in response_lower for k in self._TOPIC_KEYWORDS):
                return ValidationResult(
                    passed=False,
                    message="Response missing diabetes/health keywords."
                )

        return ValidationResult(passed=True)


def enforce_medical_policy(
    response_text: str,
    question: str,
    *,
    validator: Optional[MedicalResponsePolicy] = None,
    previous_question: Optional[str] = None,
    previous_response: Optional[str] = None,
    previous_passed: Optional[bool] = False,
    llm=None,
) -> tuple[bool, str, Optional[str]]:
    active_validator = validator or MedicalResponsePolicy()
    result = active_validator.validate(
        response_text,
        question,
        previous_question=previous_question,
        previous_response=previous_response,
        previous_passed=previous_passed,
        llm=llm,
    )

    if result.passed:
        return True, response_text.strip(), None

    return False, MedicalResponsePolicy.REFUSAL_LINE, result.message


_CRISIS_RESPONSE = (
    "I'm really concerned about what you've shared. "
    "Please know that you are not alone, and help is available right now.\n\n"
    "🆘 **Please reach out immediately:**\n"
    "- **iCall (India):** 9152987821\n"
    "- **Vandrevala Foundation:** 1860-2662-345 *(24/7)*\n"
    "- **AASRA:** 9820466627\n\n"
    "You matter. Please talk to someone. 💙"
)

_BLOCKED_RESPONSE = (
    "This request cannot be answered. "
    "If you need help, please contact a mental health professional or local support service."
)


def _call_bedrock_guardrail(text: str) -> Optional[str]:
    if not BEDROCK_MODERATION_ENABLED or not BEDROCK_GUARDRAIL_ID:
        return None

    try:
        import boto3
        from botocore.config import Config

        client = boto3.client(
            "bedrock-runtime",
            region_name=AWS_REGION,
            config=Config(
                connect_timeout=BEDROCK_MODERATION_TIMEOUT_SECONDS,
                read_timeout=BEDROCK_MODERATION_TIMEOUT_SECONDS,
                retries={"max_attempts": 0},
            ),
        )
        response = client.apply_guardrail(
            guardrailIdentifier=BEDROCK_GUARDRAIL_ID,
            guardrailVersion=BEDROCK_GUARDRAIL_VERSION,
            source="INPUT",
            content=[{"text": {"text": text}}],
        )
        return response.get("action", "NONE")

    except Exception as exc:
        print(f"⚠️  Bedrock Guardrail unavailable ({exc.__class__.__name__}: {exc}) — falling back to offline checks.")
        return None


def check_unsafe_input(question: str) -> dict:
    q = question.strip()
    if not q:
        return {"safe": True}

    lowered = q.lower()
    action = _call_bedrock_guardrail(q)

    if action == "GUARDRAIL_INTERVENED":
        print(f"🚨 Blocked [bedrock]: {q[:60]}")
        if _SELF_HARM_CONTEXT.search(lowered):
            return {"safe": False, "reason": _CRISIS_RESPONSE}
        return {"safe": False, "reason": _BLOCKED_RESPONSE}

    # Offline fallback stays active when Bedrock moderation is disabled or unavailable.
    if _SELF_HARM_CONTEXT.search(lowered):
        print(f"🚨 Blocked [self-harm]: {q[:60]}")
        return {"safe": False, "reason": _CRISIS_RESPONSE}

    if _SEXUAL_MINOR_CONTEXT.search(lowered):
        print(f"🚨 Blocked [sexual-minors]: {q[:60]}")
        return {"safe": False, "reason": _BLOCKED_RESPONSE}

    if _VIOLENT_ABUSE_CONTEXT.search(lowered):
        print(f"🚨 Blocked [violent-abuse]: {q[:60]}")
        return {"safe": False, "reason": _BLOCKED_RESPONSE}

    return {"safe": True}



# ══════════════════════════════════════════════════
# PART 4 — HIGH-RISK DIABETES / MEDICAL TRIAGE
# ══════════════════════════════════════════════════

_URGENT_MEDICAL_RESPONSE = (
    "This could be urgent. I can share general safety information, but I can't diagnose "
    "or adjust medication doses here. Please contact your doctor/diabetes care team now, "
    "or seek urgent/emergency care if symptoms are severe.\n\n"
    "Seek emergency help now if there is chest pain, trouble breathing, confusion, fainting, "
    "seizure, repeated vomiting, very high glucose with ketones, or very low glucose that is "
    "not improving. If you use insulin, follow your prescribed sick-day/hypoglycemia plan and "
    "do not change doses without medical guidance."
)

_SEVERE_LOW_RESPONSE = (
    "A very low blood sugar can be dangerous. If the person is confused, unconscious, having a "
    "seizure, or cannot swallow safely, call emergency services immediately and use prescribed "
    "glucagon if available. If awake and able to swallow, follow your clinician's hypoglycemia "
    "plan and recheck glucose as directed."
)

_HIGH_RISK_PATTERNS = [
    r"\b(chest pain|trouble breathing|shortness of breath|can't breathe|cannot breathe)\b",
    r"\b(confused|confusion|fainted|fainting|seizure|unconscious|passed out)\b",
    r"\b(repeated vomiting|vomiting repeatedly|can't keep fluids|cannot keep fluids)\b",
    r"\b(dka|ketoacidosis)\b.{0,80}\b(my|i|me|having|symptoms|ketones|vomiting|confused)\b",
    r"\b(ketones?|positive ketones?)\b.{0,80}\b(vomit|vomiting|nausea|abdominal pain|stomach pain|confused|high sugar)\b",
    r"\b(took|taken|injected|gave myself)\b.{0,40}\b(too much|extra|wrong|double)\b.{0,40}\binsulin\b",
    r"\binsulin\b.{0,40}\b(overdose|too much|double dose|wrong dose)\b",
]

_SELF_HARM_CONTEXT = re.compile(
    r"\b(kill myself|end my life|want to die|wanna die|suicid|self[- ]?harm|hurt myself)\b",
    re.IGNORECASE,
)

_SEXUAL_MINOR_CONTEXT = re.compile(
    r"\b(child|children|minor|underage|teen|kid|kids)\b.{0,40}\b(sex|sexual|porn|nude|nudes|abuse|exploit)\b|"
    r"\b(sex|sexual|porn|nude|nudes|abuse|exploit)\b.{0,40}\b(child|children|minor|underage|teen|kid|kids)\b",
    re.IGNORECASE,
)

_VIOLENT_ABUSE_CONTEXT = re.compile(
    r"\b(rape|murder|torture|terrorist|bomb|shoot up|mass shooting)\b",
    re.IGNORECASE,
)


def _extract_glucose_value(text: str) -> Optional[float]:
    lowered = text.lower()
    patterns = [
        r"(?:blood sugar|glucose|sugar|bg)\D{0,20}(\d{2,3}(?:\.\d+)?)\s*(mg/dl|mgdl|mmol/l|mmol)?",
        r"(\d{2,3}(?:\.\d+)?)\s*(mg/dl|mgdl|mmol/l|mmol)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if not match:
            continue
        value = float(match.group(1))
        unit = (match.group(2) or "mg/dl").lower()
        if unit.startswith("mmol"):
            value *= 18.0
        return value
    return None


def check_high_risk_medical_input(question: str) -> dict:
    q = question.strip()
    if not q:
        return {"urgent": False}

    lowered = q.lower()

    if _SELF_HARM_CONTEXT.search(lowered):
        return {"urgent": False}

    glucose = _extract_glucose_value(lowered)
    if glucose is not None:
        symptomatic_high = any(
            word in lowered
            for word in ("ketone", "vomit", "vomiting", "confused", "drowsy", "abdominal", "stomach pain", "breathing")
        )
        if glucose <= 54:
            return {"urgent": True, "reason": _SEVERE_LOW_RESPONSE, "kind": "severe_low_glucose"}
        if glucose >= 300 and symptomatic_high:
            return {"urgent": True, "reason": _URGENT_MEDICAL_RESPONSE, "kind": "severe_high_glucose"}
        if glucose >= 400:
            return {"urgent": True, "reason": _URGENT_MEDICAL_RESPONSE, "kind": "very_high_glucose"}

    for pattern in _HIGH_RISK_PATTERNS:
        if re.search(pattern, lowered):
            return {"urgent": True, "reason": _URGENT_MEDICAL_RESPONSE, "kind": "pattern"}

    return {"urgent": False}


__all__ = [
    "MedicalResponsePolicy",
    "enforce_medical_policy",
    "check_unsafe_input",
    "check_high_risk_medical_input",
]