from langchain_core.tools import tool
from dotenv import load_dotenv
import sys,os
from io import StringIO
from pydantic import BaseModel, Field
import pandas as pd
from typing import Optional
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import os

class PythonInputs(BaseModel):
    query: str = Field(description="A valid python command to run.")

load_dotenv()

@tool(args_schema=PythonInputs)
def python_repl_ast(query: str, path: Optional[str] = None) -> str:
    """
    Runs a Python command and returns the result.
    If `path` is provided, the dataframe at that path will be loaded as `df`.
    Supports CSV, Excel, and Parquet. 
    Any in-place modifications to `df` are saved back to the same file.
    """

    df = None
    if path:
        ext = os.path.splitext(path)[-1].lower()
        if ext == ".csv":
            df = pd.read_csv(path)
        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(path)
        elif ext == ".parquet":
            df = pd.read_parquet(path)
        else:
            return f"Unsupported file type: {ext}"

    # Restricted namespaces
    local_namespace = {"df": df}
    global_namespace = {}

    # Capture stdout for print()
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        # Run code
        exec(query, global_namespace, local_namespace)

        # Capture output
        output = captured_output.getvalue()
        if not output:
            try:
                # Eval last expression if no print output
                lines = [line for line in query.strip().split("\n") if line.strip()]
                if lines:
                    last_line_result = eval(lines[-1], global_namespace, local_namespace)
                    output = str(last_line_result)
                else:
                    output = "Executed successfully, but no output was produced."
            except Exception:
                output = "Executed successfully, but no output was produced."

        # Save back modified DataFrame if applicable
        if path and "df" in local_namespace and isinstance(local_namespace["df"], pd.DataFrame):
            df = local_namespace["df"]
            if ext == ".csv":
                df.to_csv(path, index=False)
            elif ext in [".xls", ".xlsx"]:
                df.to_excel(path, index=False)
            elif ext == ".parquet":
                df.to_parquet(path, index=False)

        return output

    except Exception as e:
        return f"Execution failed with error: {e}"

    finally:
        sys.stdout = old_stdout


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

@tool
def text_to_pdf(text: str, output_filename: str = "output.pdf") -> str:
    """
    Convert a given text corpus to a PDF file in the project root.

    Args:
        text (str): The input text corpus.
        output_filename (str): PDF file name (default: output.pdf)

    Returns:
        str: Absolute path to the saved PDF in the project root.
    """
    # Always write to project root
    output_path = os.path.join(os.getcwd(), output_filename)

    # Set up canvas
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    margin = 50
    max_width = width - 2 * margin
    line_height = 14

    # Split text into lines that fit the page width
    lines = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        current_line = ""
        for word in words:
            if c.stringWidth(current_line + " " + word, "Helvetica", 12) < max_width:
                current_line += " " + word if current_line else word
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        lines.append("")  # Blank line after paragraph

    # Draw text line by line
    y = height - margin
    for line in lines:
        if y < margin:
            c.showPage()  # New page
            y = height - margin
        c.drawString(margin, y, line)
        y -= line_height

    c.save()
    return output_path

"""
def PreprocessingPlanner_node(self, state: AgentStateModel) -> Command[Literal['supervisor']]:
        print("*****************called PreprocessingPlanner node************")

        Instructions = state.Instructions

        # 1. Instantiate the parser for our new structured output
        parser = PydanticOutputParser(pydantic_object=StructuredPlanOutput)

        task_prompt = (
        f"Find the instructions given by the user here : {Instructions} and follow this {PreprocessingPlanner_prompt} to the letter."
    )
        print(f"--- Sending this direct task to the agent ---\n{task_prompt}\n---------------------------------------------")

        system_prompt = ChatPromptTemplate.from_messages([
        
        ("system",
         "You are a DataFrame analyzer. Your primary tool is `eda_fact_sheet`. "
         "First, call the tool to get data insights. Then, based on the tool's output, "
         "provide a final answer formatted as a JSON object containing the preprocessing plan and summaries."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")        
    ])
        
        Analyzer_agent = create_tool_calling_agent(
            llm=self.llm_model,
            tools=[eda_fact_sheet],
            prompt=system_prompt
        )

        agent_executor = AgentExecutor(
            agent=Analyzer_agent,
            tools=[eda_fact_sheet],
            verbose=True,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )

        result = agent_executor.invoke({
                "input": task_prompt,
                "chat_history": []     
            })
        
        # 4. Parse the agent's final string output using the Pydantic parser
        # This acts as your guardrail. If the output is not valid JSON, it will fail here.
        try:
            final_output_string = result.get("output", "")
            parsed_output: StructuredPlanOutput = parser.parse(final_output_string)
            
            # Extract the components from the parsed object
            # --- THIS LINE IS CORRECTED ---
            plan_dict = {"plan": [step.model_dump() for step in parsed_output.plan]}
            # ---------------------------
            summary_str = f"{parsed_output.summary}\n{parsed_output.details}"

        except Exception as e:
            # Handle cases where the LLM failed to produce valid JSON
            print(f"--- FAILED TO PARSE AGENT OUTPUT ---\nError: {e}")
            print(f"Raw Output:\n{result.get('output')}")
            # Fallback or error state
            return Command(
                update={
                    "messages": state.messages[-1:] + [AIMessage(content="Error: The analysis agent failed to produce a valid preprocessing plan.", name="Analyzer_node_Error")],
                    "Analysis": [{"error": "Parsing failed", "output": result.get('output')}] 
                },
                goto="supervisor",
            )
        
        agent_dict_output = {"final_answer": result.get("output")}

        parsed = summarize_tool_output(agent_dict_output,self.llm_model)

        parsed_str = f"{parsed.summary}\n{parsed.details}"

        return Command(
            update={
                "messages": state.messages[-1:] + [AIMessage(content=parsed_str, name="Analyzer_node")],
                "Analysis": [agent_dict_output] 
            },
            goto="supervisor",
        )
"""