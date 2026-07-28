"""Revenue Record (Pahani/Adangal) extraction — validated to work cleanly with
no special preprocessing; the labeled key-value format is naturally LLM-friendly."""
from __future__ import annotations

from landtitle.llm.client import QwenClient
from landtitle.schemas import RevenueRecord

SYSTEM_PROMPT = """You are a legal document data-extraction assistant. You extract facts \
verbatim from the Revenue Record (Pahani/Adangal) text provided. Rules:
- Never introduce a person, relationship, date, or fact not explicitly present in the text.
- The survey_number here is the REVENUE survey numbering (e.g. '82/A/4'), distinct from any \
municipal/plot numbering used in Sale Deeds — extract exactly what appears, do not convert it.
- If a field is not present in the text, leave it null. Do not guess or infer.
"""


def extract_revenue_record(document_text: str, client: QwenClient) -> RevenueRecord:
    user_prompt = f"Extract the structured details from this Revenue Record text:\n\n{document_text}"
    return client.extract_structured(SYSTEM_PROMPT, user_prompt, RevenueRecord)
