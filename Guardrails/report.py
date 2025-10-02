from typing import List, Literal
from pydantic import BaseModel, Field

# Define a nested model for the phases of the strategic plan
class StrategicPlanPhase(BaseModel):
    """A single phase of the strategic action plan."""
    phase_title: str = Field(..., description="The title of the phase, e.g., 'Phase 1: Immediate Realignment & Proof of Concept (Next 30 Days)'.")
    phase_steps: List[str] = Field(..., description="A list of specific, actionable steps for this phase.")

# Define the main structure for the entire business report
class BusinessReport(BaseModel):
    """The final structured output containing the complete business case study and strategic plan."""
    subject: str = Field(..., description="The subject line of the report, identifying the dataset's theme.")
    executive_summary: str = Field(..., description="A brief, 3-4 sentence overview for a C-suite audience, stating the core opportunity and recommendation.")
    current_state_overview: str = Field(..., description="A data-driven overview of the business's macro trends and key segments.")
    core_insight: str = Field(..., description="The heart of the case study, detailing the single biggest opportunity with specific, hard numbers.")
    supporting_analysis: str = Field(..., description="Deeper insights, such as temporal patterns or behavioral analysis, that support the core insight.")
    strategic_plan: List[StrategicPlanPhase] = Field(..., description="A detailed, multi-phase action plan.")
    metrics_for_success: List[str] = Field(..., description="A list of specific KPIs that will be tracked to measure the plan's success.")