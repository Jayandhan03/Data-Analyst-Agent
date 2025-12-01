import os
from pydantic import BaseModel
import uvicorn
from langgraph.graph import START, StateGraph
from Cleaner_Agent import DataAnalystAgent, AgentStateModel
from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import tempfile

from Report_agent import Report_agent 

import uuid
from fastapi.staticfiles import StaticFiles
from Visualizer_agent import Visualizer_agent 

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = FastAPI()
agent = DataAnalystAgent()

PLOTS_DIR = "generated_plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
app.mount("/generated_plots", StaticFiles(directory=PLOTS_DIR), name="generated_plots")

class CleanRequest(BaseModel):
    path: str
    instructions: str | None = None

class CleanResponse(BaseModel):
    status: str
    message: str
    cleaned_csv_content: str | None = None

@app.post("/clean-data", response_model=CleanResponse)
async def clean_data_endpoint(request: CleanRequest):
    try:
        print(f"Received request to clean data at path: {request.path}")

        # --- Your LangGraph Logic ---
        initial_state = AgentStateModel(
            Instructions=request.instructions,
            Path=request.path,
            messages=[], Analysis=[], next="", current_reasoning=""
        )
        graph = StateGraph(AgentStateModel)
        graph.add_node("supervisor", agent.supervisor_node)
        graph.add_node("PreprocessingPlanner_node", agent.PreprocessingPlanner_node)
        graph.add_node("Cleaner_node", agent.Cleaner_node)
        graph.add_edge(START, "supervisor")
        compiled_graph = graph.compile()
        final_state = compiled_graph.invoke(initial_state)
        # --- End of Your Logic ---

        output_filename = "cleaned_" + os.path.basename(request.path)
        output_filepath = os.path.join(os.path.dirname(request.path), output_filename)
        
        if not os.path.exists(output_filepath):
             raise FileNotFoundError(f"Cleaner did not produce the expected output file: {output_filepath}")

        with open(output_filepath, 'r', encoding='utf-8') as f:
            csv_content = f.read()
        
        print("Successfully processed data and read cleaned file.")

        return {
            "status": "success", 
            "message": "Data cleaning process completed.",
            "cleaned_csv_content": csv_content
        }

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
# --- REPORT GENERATION ENDPOINT ---
class ReportRequest(BaseModel):
    path: str
    instructions: str | None = None   # optional prompt addon


class ReportResponse(BaseModel):
    success: bool
    parsed_report: dict | None = None
    raw_output: str | None = None
    error: str | None = None




@app.post("/generate-report", response_model=ReportResponse)
async def generate_report_endpoint(request: ReportRequest):
    """
    Endpoint that triggers the Report Agent to generate a structured business report.
    Expects:
      - path: str -> path to CSV file
      - instructions: Optional custom instructions
    """
    try:
        print(f"Received request to generate business report from: {request.path}")

        # Call Reporter Agent
        result = Report_agent(request.path)

        if result.get("success"):
            return {
                "success": True,
                "parsed_report": result.get("parsed_report"),
                "raw_output": result.get("raw_output"),
                "error": None,
            }
        else:
            return {
                "success": False,
                "parsed_report": None,
                "raw_output": result.get("output"),
                "error": result.get("error"),
            }

    except Exception as e:
        print(f"Report generation error: {e}")
        return {
            "success": False,
            "parsed_report": None,
            "raw_output": None,
            "error": str(e),
        }

class VisualizeRequest(BaseModel):
    path: str

class VisualizeResponse(BaseModel):
    success: bool
    parsed_visuals: dict | None = None
    raw_output: str | None = None
    error: str | None = None

# --- 4. The Endpoint ---
@app.post("/generate-visualizations", response_model=VisualizeResponse)
async def generate_visualizations_endpoint(request: VisualizeRequest):
    """
    Endpoint that triggers the Visualizer Agent to generate charts.
    Images are saved locally and returned as accessible URLs.
    """
    try:
        print(f"Received request to visualize data from: {request.path}")

        # 1. Create a unique sub-directory for this specific run to avoid file conflicts
        # Example: generated_plots/550e8400-e29b-41d4-a716-446655440000/
        run_id = str(uuid.uuid4())
        output_dir = os.path.join(PLOTS_DIR, run_id)
        os.makedirs(output_dir, exist_ok=True)

        # 2. Run the Visualizer Agent
        # We pass the absolute path for 'output_dir' so Python knows where to write
        abs_output_dir = os.path.abspath(output_dir)
        
        result = Visualizer_agent(df_path=request.path, output_dir=abs_output_dir)

        # 3. Process the result to convert local file paths to HTTP URLs
        # The agent returns absolute paths (e.g., D:/Neon/generated_plots/uuid/plot.png)
        # We need to send back URLs (e.g., http://localhost:8000/generated_plots/uuid/plot.png)
        
        if result.get("success") and result.get("parsed_visuals"):
            base_url = "http://localhost:8000/generated_plots" # Update if deployed elsewhere
            
            visuals = result["parsed_visuals"].get("visualizations", [])
            for vis in visuals:
                # Extract filename from the full path
                filename = os.path.basename(vis["file_path"])
                # Construct the serveable URL
                vis["file_path"] = f"{base_url}/{run_id}/{filename}"

        return {
            "success": result.get("success"),
            "parsed_visuals": result.get("parsed_visuals"),
            "raw_output": result.get("raw_output"),
            "error": result.get("error"),
        }

    except Exception as e:
        print(f"Visualization error: {e}")
        return {
            "success": False,
            "parsed_visuals": None,
            "raw_output": None,
            "error": str(e),
        }
    

# --- Standard `uvicorn.run` call (No changes) ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)