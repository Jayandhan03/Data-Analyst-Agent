from pydantic import BaseModel, Field

class BusinessReportOutput(BaseModel):
    subject: str = Field(
        description="Title of the report summarizing dataset and purpose."
    )
    executive_summary: str = Field(
        description="3–4 paragraphs summarizing dataset theme, business context, major trends, and opportunities."
    )
    data_overview_and_quality_review: str = Field(
        description="Summary of data structure, columns, missing values, quality issues, and anomalies."
    )
    descriptive_and_diagnostic_analysis: str = Field(
        description="Detailed findings, segment-level trends, relationships, and outliers across all major variables."
    )
    key_insights_and_patterns: str = Field(
        description="5–7 key insights with supporting metrics, ratios, and performance comparisons."
    )
    strategic_opportunity: str = Field(
        description="The single biggest business opportunity or gap with data-backed justification."
    )
    recommendations_and_forecast: str = Field(
        description="3–6 actionable recommendations followed by a forward-looking forecast or trend summary."
    )
