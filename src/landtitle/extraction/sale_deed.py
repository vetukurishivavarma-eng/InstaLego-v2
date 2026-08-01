"""Sale Deed extraction — validated to work well as flat OCR'd/extracted text
straight into the LLM (unlike Encumbrance Certificates, see ec_extractor.py)."""
from __future__ import annotations

from landtitle.llm.client import QwenClient
from landtitle.schemas import SaleDeed

SYSTEM_PROMPT = """You are a legal document data-extraction assistant. You extract facts \
verbatim from the Sale Deed text provided. Rules:
- Never introduce a person, relationship, date, or fact not explicitly present in the text.
- "S/o", "D/o", "W/o" indicate a relationship only, not co-ownership or party status.
- Keep land_extent and built_up_area as SEPARATE fields — never combine them or substitute one for the other.
- If a seller is represented by a General Power of Attorney holder, record the GPA holder's \
identity in `represented_by_gpa`, never in the seller's own `name` field.
- A deed often names each vendor/vendee once, then later refers back to them collectively by \
number (e.g. "Vendor No.1, 2 and Vendor No.3 through their GPA holder ... do hereby sell"). That \
back-reference is NOT a new party — do not create a separate seller/buyer entry for it. Only add \
an entry when the text gives an actual person's name.
- A deed's recital may explain how the CURRENT seller themselves acquired the property, naming an \
earlier document (`prior_title_deed_references`). This earlier document is often NOT a sale — it \
may be a Partition Deed, Gift Deed, Settlement Deed, Will, or other instrument. Always capture the \
TYPE of instrument exactly as the text names it, alongside its document number and date. Never \
assume or default to "Sale Deed" for a prior reference — copy only what the text actually says.
- If a field is not present in the text, leave it null. Do not guess or infer.
"""


def extract_sale_deed(document_text: str, client: QwenClient) -> SaleDeed:
    user_prompt = f"Extract the structured details from this Sale Deed text:\n\n{document_text}"
    return client.extract_structured(SYSTEM_PROMPT, user_prompt, SaleDeed)
