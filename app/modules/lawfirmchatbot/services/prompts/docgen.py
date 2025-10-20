from typing import Dict, Sequence, Any, List
from os import getenv

COMMON_DIRECTIVE = (
    " Always output the finished document as filing-ready HTML (use <div>, <p>, <ol>, <table> as needed)"
    " without markdown code fences or meta commentary. Include appropriate headings, verification/attestation,"
    " and signature blocks when customary."
)

REQUIRED_FIELD_KEYWORDS: Dict[str, List[str]] = {
    "case type number": ["case no", "case number", "suit no", "petition no", "w.p.", "wp", "fir no"],
    "case title": [" v ", " vs ", "versus", "title"],
    "representing": ["petitioner", "respondent", "plaintiff", "defendant", "applicant", "represent"],
    "deponent": ["deponent", "i,", "we,"],
    "address": ["address", "resident", "residing", "at "],
    "capacity": ["authorized", "attorney", "capacity", "under instruction", "through"],
}


def detect_missing_fields(user_prompt: str) -> List[str]:
    """
    Heuristic detection of commonly required document attributes.
    Used only for non-placeholder flows.
    """
    text = (user_prompt or "").lower()
    missing = []
    for field, tokens in REQUIRED_FIELD_KEYWORDS.items():
        if not any(token in text for token in tokens):
            missing.append(field)
    return missing


def generate_docgen_prompt(user_prompt: str, doc_type: str = "legal document", placeholders: bool = False) -> str:
    """
    Modified DocGen prompt builder with Gemini placeholder support.
    """
    provider = getenv("LLM_PROVIDER", "gemini")
    doc_label = (doc_type or "legal document").replace("_", " ")

    if placeholders or provider.lower() == "gemini":
        return (
            f"Draft a structured {doc_label} based on the following request:\n\n"
            f"'{user_prompt}'\n\n"
            "Use [PLACEHOLDER] where details like case number, title, address, or capacity are missing.\n"
            "Ensure a professional legal format with clear sections (heading, parties, grounds, prayer)."
        )

    missing_fields = detect_missing_fields(user_prompt)
    if missing_fields:
        return (
            f"To draft your document, please provide the following missing details: {', '.join(missing_fields)}."
        )

    return (
        f"Draft a complete, formal {doc_label} based on the following request:\n\n"
        f"{user_prompt}\n\n"
        "Follow standard legal structure (heading, parties, body, prayer)."
    )

DOCGEN_PROMPTS: Dict[str, str] = {
    "affidavit": (
        "You are to draft a formal affidavit compliant with Pakistani courts. "
        "Use verified, declarative language, include verification and attestation blocks, "
        "and mirror the numbering/style from the provided references."
    ),
    "synopsis": (
        "You are to prepare a concise case synopsis summarizing key facts, questions presented, "
        "and relief sought. Highlight procedural posture and supporting authorities."
    ),
    "rejoinder": (
        "You are to write a rejoinder replying to the opposite party's affidavit. "
        "Address each contested allegation, rebut with clarity, and restate the relief requested."
    ),
    "legal_notice": (
        "You are to draft a formal legal notice to be served upon the respondent. "
        "State background facts, legal breaches, the demand being made, and a clear compliance deadline."
    ),
    "general": (
        "You are to prepare the exact document type requested by the user following Pakistani legal practice. "
        "Maintain professional tone, ensure headings, body, and closing sections match the intent."
    ),
}


def get_docgen_prompt(doc_type: str) -> str:
    key = (doc_type or "").lower()
    base_prompt = DOCGEN_PROMPTS.get(key, DOCGEN_PROMPTS["general"])
    return base_prompt + COMMON_DIRECTIVE


def build_docgen_prompt(
    user_query: str,
    answers: Dict[str, str],
    context: Sequence[Any],
    use_placeholders: bool = False,
) -> str:
    """
    Prepare the user prompt for doc generation with explicit references.
    """
    details = "\n".join(
        f"- {key}: {value}"
        for key, value in (answers or {}).items()
        if value
    ) or "- user_request: " + user_query.strip()

    context_snippets = []
    for chunk in list(context or [])[:3]:
        text = getattr(chunk, "page_content", None) or ""
        text = text.strip()
        if text:
            context_snippets.append(text)

    refs = "\n\n---\n\n".join(context_snippets) if context_snippets else "[No retrieved context available]"

    placeholder_rules = (
        "If facts are missing, insert [PLACEHOLDER] and keep the document structure intact.\n"
        "Do not fabricate names, dates, or case numbers; prefer placeholders instead."
    )
    strong_skeleton_rules = (
        "Output a minimal skeleton/template with headings and standard sections only.\n"
        "Use [PLACEHOLDER] for parties, dates, addresses, case numbers, and specifics.\n"
        "Do not expand into narrative text; avoid verbose paragraphs and examples."
    ) if use_placeholders else ""

    return (
        f'You are drafting based on this user intent: "{user_query}"\n\n'
        f"DETAILS:\n{details}\n\n"
        "REFERENCE DOCUMENTS (use their structure, tone, and style as guidance):\n"
        f"{refs}\n\n"
        f"{placeholder_rules}\n"
        f"{strong_skeleton_rules}"
    )
