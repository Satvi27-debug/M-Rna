"""Structured LLM output subpackage: schema, safety validation, generation.

Boundary rules
--------------
* **Safety validation is separate from presentation.** :mod:`src.llm.schemas`
  owns the Pydantic contract and the deterministic safety checks that no prompt
  can override. :mod:`src.llm.report_generator` owns the LLM call, the retry, and
  the rendering to Markdown. Neither is embedded in the other (architecture
  rule 7).
* No secret is read at import time. ``report_generator`` loads ``LLM_API_KEY``
  from the environment inside the function that needs it, so importing this
  package in a test never requires a key.

Requirements owned here
-----------------------
FR-4.1  A Pydantic schema enforcing exactly ``Eligible_For_Vaccine`` (bool),
        ``Recommended_Dose_mg`` (float), ``Reasoning_Steps`` (List[str]),
        ``FDA_Safety_Warnings_Cited`` (List[str]) and ``Disclaimer`` (str).
FR-4.4  Reject any ``Recommended_Dose_mg`` above the bound stated in the
        retrieved guidelines and set ``Eligible_For_Vaccine`` to false.
FR-4.5  Every report carries a non-empty ``Disclaimer``.
"""

from src.llm.report_generator import (
    build_prompt,
    generate_report,
    persist_report,
    render_markdown,
)
from src.llm.schemas import (
    ClinicalSafetyReport,
    DoseBound,
    SafetyViolation,
    enforce_safety_constraints,
    validate_report,
)

__all__ = [
    "ClinicalSafetyReport",
    "DoseBound",
    "SafetyViolation",
    "validate_report",
    "enforce_safety_constraints",
    "build_prompt",
    "generate_report",
    "render_markdown",
    "persist_report",
]