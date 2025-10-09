# In Guardrails/validation.py (create a new file)

from pydantic import BaseModel, Field
from typing import List

class ValidationCheck(BaseModel):
    column: str = Field(description="The column that was checked.")
    action: str = Field(description="The cleaning action that was validated.")
    passed: bool = Field(description="True if the validation check passed, False otherwise.")
    details: str = Field(description="A description of the check performed and the outcome (e.g., 'Checked dtype, found int64 as expected').")

class ValidationReport(BaseModel):
    overall_summary: str = Field(description="A high-level summary of the entire validation process.")
    checks: List[ValidationCheck] = Field(description="A list of all individual validation checks performed.")