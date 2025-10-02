from typing import List
from pydantic import BaseModel, Field

# Define the structure for a single preprocessing step
class PreprocessingStep(BaseModel):
    column: str = Field(..., description="The name of the column to be processed.")
    action: str = Field(..., description="The specific, safe preprocessing command for this column.")

# Define the complete, structured output we expect from the agent
class StructuredPlanOutput(BaseModel):
    """The final structured output containing the preprocessing plan and its summary."""
    plan: List[PreprocessingStep] = Field(..., description="The detailed, column-by-column preprocessing plan as a JSON array.")
    summary: str = Field(..., description="A concise, one-sentence summary of the plan, under 100 characters.")
    details: str = Field(..., description="A full, human-readable summary of all the preprocessing actions to be taken.")