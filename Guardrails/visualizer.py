from pydantic import BaseModel, Field
from typing import List

class VisualizationItem(BaseModel):
    title: str = Field(
        description="Title of the chart (usually derived from the filename)."
    )
    insight: str = Field(
        description="Brief 1–2 sentence description of what the chart shows."
    )
    file_path: str = Field(
        description="Absolute path to the saved plot file, e.g. '/output/plot1.png'."
    )

class VisualizationReportOutput(BaseModel):
    report_title: str = Field(
        description="Title or heading of the markdown visualization report, usually 'Data Visualizations Report'."
    )
    visualizations: List[VisualizationItem] = Field(
        description="List of 5–7 generated visualizations with insights and saved file paths."
    )
