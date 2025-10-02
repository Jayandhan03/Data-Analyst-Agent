from typing import Literal
from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    """The structured result of the data validation process."""
    status: Literal["SUCCESS", "FAILURE"] = Field(..., description="The overall validation status.")
    message: str = Field(..., description="If SUCCESS, a confirmation message. If FAILURE, the specific column and action that failed.")
