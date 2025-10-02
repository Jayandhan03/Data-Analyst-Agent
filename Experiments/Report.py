import os
import sys
import re
import json
import time
from io import StringIO
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent

# --- PROMPTS ---

ARCHITECT_PROMPT = """You are a Lead Data Strategist. Your task is to analyze the initial factsheet of a dataset and design the blueprint for a comprehensive business report. You must think about a compelling narrative arc: start broad, narrow down to a core insight, and end with actionable recommendations.

Your output MUST be a JSON array of objects, where each object represents a section of the final report. You must decide on the best sections yourself based on the data.

Each JSON object in the array must contain:
1.  "section_title": A clear, business-focused title for the section.
2.  "key_question": The central business question this section will answer.
3.  "analysis_type": A brief description of the pandas analysis needed (e.g., "Group by region and calculate mean price").
4.  "visualization_needed": A description of the visualization that will best prove the insight (e.g., "Bar chart of top 5 regions by price").
5.  "python_code": The exact, complete, and runnable Python code using pandas, matplotlib, and seaborn to perform the analysis AND save the visualization to the `visualizations/` directory with a descriptive filename.

Example for one section:
{
    "section_title": "The Premium Price Divide: Conventional vs. Organic",
    "key_question": "What is the price difference between organic and conventional avocados, and how does this vary?",
    "analysis_type": "Compare the average price distribution for each 'type'.",
    "visualization_needed": "A box plot comparing the 'AveragePrice' for 'conventional' and 'organic' types.",
    "python_code": "df = pd.read_csv(r'D:\\Code Assistant\\avocado.csv'); plt.figure(figsize=(8, 6)); sns.boxplot(x='type', y='AveragePrice', data=df); plt.title('Price Distribution: Conventional vs. Organic'); plt.savefig('visualizations/price_distribution_by_type.png'); plt.close(); print(df.groupby('type')['AveragePrice'].describe().to_string())"
}

Based on the provided factsheet, create a complete report plan with 3 to 5 comprehensive sections.
"""

# --- MODIFIED: The new, efficient Publisher Prompt ---
PUBLISHER_PROMPT = """You are a specialized "Document Publisher" agent. Your only function is to write Python code to generate a professional PDF document from a pre-existing Python variable.

**Your Task:**
You MUST write a single, complete Python script that performs the following:

1.  **Assume Data Exists:** Your script will be executed in an environment where a Python list of dictionaries named `completed_report_content` already exists.
2.  **Data Structure:** This variable has the following structure:
    ```
    [
        {
            "section_title": "Title of Section 1",
            "narrative": "Text content for section 1...",
            "visualization_path": "visualizations/chart1.png"
        },
        ...
    ]
    ```
3.  **Generate PDF:** Using the `fpdf` library, you must:
    a. Initialize a PDF document.
    b. Add a main title page for the report.
    c. Loop through the `completed_report_content` list. For each section dictionary:
        i.  Add the `section_title` as a heading.
        ii. Write the `narrative` text.
        iii. Embed the image from the `visualization_path` if it exists and is valid.
    d. Save the final document as `reports/Final_Business_Report.pdf`.

**CRITICAL DIRECTIVE:** Your final output must ONLY be the Python code required to generate the PDF. Do not add any other explanation, text, or markdown formatting. The code should not define the `completed_report_content` variable; it should use it directly.
"""

# --- 1. SETUP & CONFIGURATION ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found.")

CSV_PATH = r"D:\Code Assistant\avocado.csv"
VISUALIZATIONS_DIR = "visualizations"
REPORTS_DIR = "reports"

for dir_path in [VISUALIZATIONS_DIR, REPORTS_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

llm_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

# --- 2. TOOL DEFINITIONS ---
@tool
def eda_fact_sheet(path:str, sample_size: int = 3, freq_top: int = 3):
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

    n_rows, n_columns = df.shape
    fact_sheet = {"n_rows": n_rows, "n_columns": n_columns, "batches": []}
    batch_size = 15
    batch_splits = [df.columns[i:i+batch_size].tolist() for i in range(0, n_columns, batch_size)]

    for batch_cols in batch_splits:
        batch_profile = {"columns": {}}
        for col in batch_cols:
            series = df[col]
            col_profile = {}
            total = len(series)
            col_profile["dtype"] = str(series.dtype)
            col_profile["null_percent"] = round(float(series.isna().sum() / total * 100), 2)
            col_profile["unique_percent"] = round(float(series.nunique(dropna=True) / total * 100), 2)
            try:
                col_profile["examples"] = series.dropna().sample(min(sample_size, series.dropna().shape[0]), random_state=42).tolist()
            except ValueError:
                col_profile["examples"] = []
            if not series.isna().all():
                top_freq = series.value_counts(dropna=True).head(freq_top)
                col_profile["top_values"] = {str(k): int(v) for k, v in top_freq.to_dict().items()}
            if pd.api.types.is_numeric_dtype(series):
                col_profile.update({"min": float(series.min(skipna=True)), "max": float(series.max(skipna=True)), "mean": float(series.mean(skipna=True)), "std": float(series.std(skipna=True))})
                if series.std(skipna=True) > 0:
                    z_scores = ((series - series.mean(skipna=True)) / series.std(skipna=True)).abs()
                    col_profile["has_outliers"] = bool((z_scores > 3).any())
                else:
                    col_profile["has_outliers"] = False
            elif pd.api.types.is_datetime64_any_dtype(series):
                if not series.dropna().empty:
                    col_profile["min_date"] = str(series.min())
                    col_profile["max_date"] = str(series.max())
            elif pd.api.types.is_object_dtype(series):
                lengths = series.dropna().astype(str).map(len)
                if not lengths.empty:
                    col_profile["avg_length"] = float(lengths.mean())
                    col_profile["max_length"] = int(lengths.max())
                unusual = series.dropna().astype(str).str.contains(r"[^a-zA-Z0-9\s]").sum()
                col_profile["unusual_char_percent"] = round(float(unusual / total * 100), 2)
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
def python_repl_ast(query: str) -> str:
    """
    Runs a Python command and returns the result. This tool has access to pandas, matplotlib, and seaborn.
    All plots MUST be saved to the 'visualizations/' directory.
    """
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        # --- MODIFIED: More robust execution context ---
        # Provide common libraries automatically to the execution environment.
        exec_globals = {
            "pd": pd,
            "plt": plt,
            "sns": sns,
            "os": os
        }
        exec(query, exec_globals)
        output = captured_output.getvalue()
        if not output:
            output = "Code executed successfully, no output printed."
        return output
    except Exception as e:
        return f"Execution failed with error: {e}"
    finally:
        sys.stdout = old_stdout

# --- 3. AGENT & EXECUTOR SETUP ---
tools = [eda_fact_sheet, python_repl_ast]
agent_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. You have access to tools to analyze data and generate reports."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])
agent = create_tool_calling_agent(llm=llm_model, tools=tools, prompt=agent_prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- 4. THE MAIN ORCHESTRATION WORKFLOW ---
def main():
    print("="*50)
    print("📊 STARTING AUTONOMOUS DATA ANALYST WORKFLOW 📊")
    print("="*50)

    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 5

    # === STAGE 1: THE ARCHITECT (PLANNING) ===
    print("\n--- [STAGE 1/3] Architect is designing the report blueprint... ---")
    initial_factsheet = eda_fact_sheet.invoke({"path": CSV_PATH})
    architect_input = f"Here is the factsheet for the dataset:\n{json.dumps(initial_factsheet, indent=2)}\n\n{ARCHITECT_PROMPT}"
    
    report_plan_str = None
    for attempt in range(MAX_RETRIES):
        try:
            print(f"Architect attempt {attempt + 1} of {MAX_RETRIES}...")
            architect_response = agent_executor.invoke({"input": architect_input, "chat_history": []})
            report_plan_str = architect_response.get("output", "")
            if report_plan_str and "python_code" in report_plan_str:
                break
        except Exception as e:
            print(f"   ⚠️ Architect attempt {attempt + 1} failed with error: {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"   Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                print("   ❌ Max retries reached. Architect stage failed.")
    
    if not report_plan_str:
        print("Aborting workflow as the Architect stage failed.")
        return

    try:
        report_plan = json.loads(report_plan_str.replace("```json", "").replace("```", "").strip())
        print("✅ Architect created the blueprint successfully.")
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Architect failed to produce a valid JSON plan. Aborting. Error: {e}\nRaw output was:\n{report_plan_str}")
        return

    print(f"\nPausing for {RETRY_DELAY_SECONDS} seconds before starting analysis...")
    time.sleep(RETRY_DELAY_SECONDS)

    # === STAGE 2: THE ANALYST (EXECUTION) ===
    print("\n--- [STAGE 2/3] Analyst is executing the plan (gathering insights & creating visuals)... ---")
    completed_report_content = []
    for i, section in enumerate(report_plan):
        print(f"\n-> Analyzing Section {i+1}: {section['section_title']}")
        try:
            analysis_result = python_repl_ast.invoke({"query": section["python_code"]})
            vis_path_match = re.search(r"savefig\(['\"](visualizations/[^'\"]+\.png)['\"]\)", section["python_code"])
            vis_path = vis_path_match.group(1) if vis_path_match else "No visualization generated."
            completed_section = {
                "section_title": section["section_title"],
                "narrative": analysis_result.strip(),
                "visualization_path": vis_path
            }
            completed_report_content.append(completed_section)
            print(f"   ✅ Section completed. Visual saved to {vis_path}")
        except Exception as e:
            print(f"   ❌ ERROR analyzing section {i+1}. Skipping. Error: {e}")
        time.sleep(2)

    print(f"\nPausing for {RETRY_DELAY_SECONDS} seconds before starting publishing...")
    time.sleep(RETRY_DELAY_SECONDS)

    # === STAGE 3: THE PUBLISHER (PDF GENERATION) ===
    print("\n--- [STAGE 3/3] Publisher is assembling the final PDF report... ---")
    
    # --- MODIFIED: The new, efficient input for the Publisher ---
    publisher_input = PUBLISHER_PROMPT
    
    pdf_code = None
    for attempt in range(MAX_RETRIES):
        try:
            print(f"Publisher attempt {attempt + 1} of {MAX_RETRIES}...")
            publisher_response = agent_executor.invoke({"input": publisher_input, "chat_history": []})
            pdf_code = publisher_response.get("output", "").replace("```python", "").replace("```", "").strip()
            if pdf_code and "fpdf" in pdf_code.lower():
                break
        except Exception as e:
            print(f"   ⚠️ Publisher attempt {attempt + 1} failed with error: {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"   Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                print("   ❌ Max retries reached. Publisher stage failed.")

    if pdf_code:
        print("✅ Publisher generated the PDF creation code. Now executing it...")
        try:
            # Execute the code generated by the Publisher to create the PDF
            exec_globals = {
                "FPDF": FPDF,
                "completed_report_content": completed_report_content,
                "os": os # Provide 'os' module for path validation
            }
            exec(pdf_code, exec_globals)
            
            final_pdf_path = "reports/Final_Business_Report.pdf"
            if os.path.exists(final_pdf_path):
                 print("\n" + "="*50)
                 print(f"🎉 SUCCESS! Your complete business report is ready. 🎉")
                 print(f"You can find it at: {os.path.abspath(final_pdf_path)}")
                 print("="*50)
            else:
                raise FileNotFoundError("PDF code executed but file was not found.")
        except Exception as e:
            print(f"❌ ERROR: Failed to execute the PDF generation code. Error: {e}")
    else:
        print("❌ ERROR: Publisher failed to generate PDF creation code.")

if __name__ == "__main__":
    main()