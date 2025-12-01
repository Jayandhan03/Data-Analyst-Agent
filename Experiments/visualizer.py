import os
import sys
import re
from io import StringIO
from typing import Optional
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent

# --- 1. SETUP & CONFIGURATION ---

# Load environment variables

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found. Make sure it's set in the .env file.")

# Define paths
CSV_PATH = r"D:\Code Assistant\avocado.csv"
VISUALIZATIONS_DIR = "visualizations"

# Create the output directory if it doesn't exist
if not os.path.exists(VISUALIZATIONS_DIR):
    os.makedirs(VISUALIZATIONS_DIR)
    print(f"Created directory: {VISUALIZATIONS_DIR}")

# Initialize Gemini LLM
llm_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", # Using 1.5 Flash for better instruction following
    google_api_key=api_key
)

# --- 2. THE NEW, VISUALIZATION-FOCUSED PROMPT ---

VISUALIZATION_PROMPT = """You are an expert Data Analyst and Storyteller. Your sole mission is to analyze the provided CSV dataset and generate a series of 10 insightful, professional-quality visualizations that tell a compelling story about the data.

**Your Mandated Workflow:**

1.  **Initial Reconnaissance:** You MUST first call the `eda_fact_sheet` tool on the provided `df_path` to understand the data's structure, columns, and content.
2.  **Strategic Visualization Plan:** After analyzing the fact sheet, you must decide on 10 key insights you want to visualize. Your goal is to cover different aspects of the data, such as trends over time, comparisons between categories, distributions, and relationships between variables.
3.  **Execution & Generation:** For each of the 10 insights, you MUST perform the following steps:
    a. Use the `python_repl_ast` tool to write and execute Python code using `pandas`, `matplotlib`, and `seaborn`.
    b. Your code MUST generate a high-quality plot (e.g., line chart, bar chart, histogram, scatter plot, box plot).
    c. **CRITICAL SAVE COMMAND:** You MUST save each plot as a unique `.png` file directly into the `visualizations/` directory. Use a clear, descriptive filename (e.g., `visualizations/avg_price_by_region.png`).
    d. **CRITICAL REPORTING COMMAND:** After generating all 10 plots, your final output must be a well-structured text report.
4.  **Final Report Synthesis:** Your final output will be a markdown-formatted report that lists each visualization you created. For each visualization, you MUST provide:
    a. A clear title for the visualization.
    b. The exact file path in the format `(File: visualizations/your_filename.png)`.
    c. A brief, 1-2 sentence explanation of what the visualization shows and the business insight it reveals.

**CRITICAL DIRECTIVES:**
- You MUST generate exactly 10 visualizations.
- Every visualization MUST be saved to the `visualizations/` directory.
- Your final output MUST be the text report containing the list of all 10 generated files and their explanations. Do not output anything else.
- Ensure your Python code for plotting is robust: create a figure, generate the plot, add a title and labels, save the figure, and then close the plot using `plt.close()` to avoid memory issues.
"""

# --- 3. TOOL DEFINITIONS (Your existing tools are perfect) ---

@tool
def eda_fact_sheet(path:str,sample_size: int = 3, freq_top: int = 3):
    """
    Generates a batchwise fact sheet for a dataset.

    Args:
        path (str): Path to CSV or Excel dataset.
        sample_size (int): Number of example values per column.
        freq_top (int): Number of top frequent values per column.

    Returns:
        dict: Fact sheet containing dataset-level info and column-level stats in batches.
    """
    path = path.strip().strip('"').strip("'")
    path = os.path.normpath(path)

    #--- Load dataset ---
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".xlsx") or path.endswith(".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError("Unsupported file format. Use CSV or Excel.")

    df = pd.read_csv(path)

    n_rows, n_columns = df.shape

    fact_sheet = {
        "n_rows": n_rows,
        "n_columns": n_columns,
        "batches": []
    }

    # --- Determine batch splits ---
    batch_size = 15
    batch_splits = [df.columns[i:i+batch_size].tolist() for i in range(0, n_columns, batch_size)]

    # --- Process each batch ---
    for batch_cols in batch_splits:
        batch_profile = {"columns": {}}
        for col in batch_cols:
            series = df[col]
            col_profile = {}

            total = len(series)

            # Core stats
            col_profile["dtype"] = str(series.dtype)
            col_profile["null_percent"] = round(float(series.isna().sum() / total * 100), 2)
            col_profile["unique_percent"] = round(float(series.nunique(dropna=True) / total * 100), 2)

            # Example values
            try:
                col_profile["examples"] = series.dropna().sample(
                    min(sample_size, series.dropna().shape[0]),
                    random_state=42
                ).tolist()
            except ValueError:
                col_profile["examples"] = []

            # Top frequent values
            if not series.isna().all():
                top_freq = series.value_counts(dropna=True).head(freq_top)
                col_profile["top_values"] = {str(k): int(v) for k, v in top_freq.to_dict().items()}

            # Numeric columns
            if pd.api.types.is_numeric_dtype(series):
                col_profile.update({
                    "min": float(series.min(skipna=True)),
                    "max": float(series.max(skipna=True)),
                    "mean": float(series.mean(skipna=True)),
                    "std": float(series.std(skipna=True))
                })
                if series.std(skipna=True) > 0:
                    z_scores = ((series - series.mean(skipna=True)) / series.std(skipna=True)).abs()
                    col_profile["has_outliers"] = bool((z_scores > 3).any())
                else:
                    col_profile["has_outliers"] = False

            # Datetime columns
            elif pd.api.types.is_datetime64_any_dtype(series):
                if not series.dropna().empty:
                    col_profile["min_date"] = str(series.min())
                    col_profile["max_date"] = str(series.max())

            # Text/categorical columns
            elif pd.api.types.is_object_dtype(series):
                lengths = series.dropna().astype(str).map(len)
                if not lengths.empty:
                    col_profile["avg_length"] = float(lengths.mean())
                    col_profile["max_length"] = int(lengths.max())
                unusual = series.dropna().astype(str).str.contains(r"[^a-zA-Z0-9\s]").sum()
                col_profile["unusual_char_percent"] = round(float(unusual / total * 100), 2)

            # Flags
            if series.nunique(dropna=True) == total:
                col_profile["is_identifier"] = True
            elif series.nunique(dropna=True) <= 1:
                col_profile["is_constant"] = True

            batch_profile["columns"][col] = col_profile

        fact_sheet["batches"].append(batch_profile)

    return fact_sheet


class PythonInputs(BaseModel):
    query: str = Field(description="A valid python command to run.")

@tool(args_schema=PythonInputs)
def python_repl_ast(query: str, path: Optional[str] = None) -> str:
    """
    Runs a Python command and returns the result.
    If `path` is provided, the dataframe at that path will be loaded as `df`.
    This tool has access to pandas, matplotlib, and seaborn.
    All plots MUST be saved to the 'visualizations/' directory.
    """
    # Use a global df to persist changes between calls if needed, but it's cleaner to reload
    df = None
    if path:
        path = path.strip().strip('"').strip("'")
        path = os.path.normpath(path)
        if os.path.exists(path):
            df = pd.read_csv(path)
        else:
            return f"Error: Path does not exist - {path}"

    # Restricted namespaces
    # Important: Pass the visualization libraries to the exec environment
    local_namespace = {
        "pd": pd,
        "plt": plt,
        "sns": sns,
        "df": df
    }
    global_namespace = {}

    # Capture stdout for print()
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        exec(query, global_namespace, local_namespace)
        output = captured_output.getvalue()
        if not output:
            output = "Code executed successfully. Plot saved."
        return output
    except Exception as e:
        return f"Execution failed with error: {e}"
    finally:
        sys.stdout = old_stdout

# --- 4. AGENT AND EXECUTOR SETUP ---

# Note: The system prompt is simplified. The main instructions are now in the user-facing prompt.
system_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
           "system",
            "You are a helpful data analysis agent. Your primary goal is to help the user by generating visualizations based on their instructions. You have access to a CSV file at the path: {df_path}."
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)

# Create the agent
visualizer_agent = create_tool_calling_agent(
    llm=llm_model,
    tools=[eda_fact_sheet, python_repl_ast],
    prompt=system_prompt_template
)

# Create the Agent Executor
agent_executor = AgentExecutor(
    agent=visualizer_agent,
    tools=[eda_fact_sheet, python_repl_ast],
    verbose=True,
    handle_parsing_errors=True,
)

# --- 5. EXECUTION AND POST-PROCESSING ---

def main():
    print("--- Starting Visualization Agent ---")

    # The task_prompt now directly uses our new visualization-focused prompt
    task_prompt = VISUALIZATION_PROMPT

    # Invoke the agent
    result = agent_executor.invoke({
        "input": task_prompt,
        "df_path": CSV_PATH,
        "chat_history": []
    })

    # --- Post-processing and presenting the final output ---
    final_report_text = result.get("output", "Agent did not produce a final report.")

    # Use regular expressions to find all visualization paths in the report
    visualization_paths = re.findall(r"\(File: (visualizations/[-_a-zA-Z0-9]+\.png)\)", final_report_text)

    print("\n" + "="*50)
    print("✅ AGENT RUN COMPLETE")
    print("="*50 + "\n")

    # Print the text of the report
    print("--- AGENT'S FINAL REPORT ---")
    print(final_report_text)

    # Print the list of generated assets and verify they exist
    print("\n" + "="*50)
    print("🖼️  VERIFIED VISUAL ASSETS")
    print("="*50)

    if visualization_paths:
        print(f"Agent reported creating {len(visualization_paths)} visualizations:\n")
        for path in visualization_paths:
            if os.path.exists(path):
                print(f"  - ✅ {path} (File found)")
            else:
                print(f"  - ❌ {path} (ERROR: File not found!)")
    else:
        print("Agent did not report creating any visualization files.")

    # Count actual files in the directory for a final check
    actual_files = os.listdir(VISUALIZATIONS_DIR)
    print(f"\nFound {len(actual_files)} files in the '{VISUALIZATIONS_DIR}' directory.")
    print("="*50)

if __name__ == "__main__":
    main()

