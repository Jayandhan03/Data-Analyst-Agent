from typing import List
from pydantic import BaseModel, Field

# Define the structured output we expect from the Cleaner agent
class CleaningSummary(BaseModel):
    """The final structured output containing the summary of the cleaning process."""
    summary: str = Field(..., description="A concise, one-sentence summary of the cleaning process and its outcome.")
    details: str = Field(..., description="A full, human-readable summary of all the cleaning actions taken and confirmation that the file has been saved.")