from io import StringIO
from langchain_core.tools import tool
import sys
import os
from pydantic import BaseModel, Field
import pandas as pd
from typing import Optional

class CleaningToolInput(BaseModel):
    query: str = Field(..., description="A valid pandas code snippet that modifies the 'df' DataFrame.")
    path: str = Field(..., description="Full path to the source dataset file (CSV, Excel, etc.). A new cleaned file will be created based on this path.")

@tool(args_schema=CleaningToolInput)
def python_cleaning_tool(query: str, path: str) -> str:
    """
    Executes a pandas command on a DataFrame loaded from a file path.
    IMPORTANT: This tool SAVES the modified DataFrame to a NEW file prefixed with 'cleaned_'.
    Use this for cleaning and transformation tasks. The DataFrame is accessible as 'df'.
    """
    # Normalize the path to avoid OS-specific issues
    path = os.path.normpath(path)
    if not os.path.exists(path):
        return f"❌ Source file not found: {path}"

    ext = os.path.splitext(path)[-1].lower()
    try:
        if ext == ".csv":
            df = pd.read_csv(path, encoding='utf-8')
        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(path)
        else:
            return f"❌ Unsupported file type: {ext}"
    except Exception as e:
        return f"❌ Error reading file '{path}': {e}"

    local_namespace = {"df": df, "pd": pd}
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        # Execute the cleaning code provided by the agent
        exec(query, {}, local_namespace)
        output = captured_output.getvalue()

        # Get the modified DataFrame from the local namespace
        modified_df = local_namespace.get("df")
        if modified_df is None:
            return "❌ Error: The 'df' DataFrame was not found after code execution."

        # === THIS IS THE CRITICAL FIX ===
        # 1. Construct the NEW output path with the 'cleaned_' prefix
        directory = os.path.dirname(path)
        filename = os.path.basename(path)
        output_filename = f"cleaned_{filename}"
        output_filepath = os.path.join(directory, output_filename)

        # 2. Save the modified DataFrame to the NEW file
        if ext == ".csv":
            modified_df.to_csv(output_filepath, index=False, encoding='utf-8')
        elif ext in [".xls", ".xlsx"]:
            modified_df.to_excel(output_filepath, index=False)
        # ===============================

        return f"✅ Executed and saved cleaned data to {output_filepath}.\n{output or 'No print output.'}"

    except Exception as e:
        return f"❌ Execution failed: {e}"
    finally:
        sys.stdout = old_stdout

# --- Tool 2: For the Validation Node (Read-Only) ---

class ValidationToolInput(BaseModel):
    query: str = Field(..., description="A valid pandas code snippet that inspects 'df' and prints results.")
    path: str = Field(..., description="Full path to the dataset file (CSV, Excel, etc.) to be inspected.")

@tool(args_schema=ValidationToolInput)
def python_validation_tool(query: str, path: str) -> str:
    """
    Executes a read-only pandas command to inspect a DataFrame loaded from a file path.
    IMPORTANT: This tool DOES NOT SAVE any changes. It loads all data as strings to
    accurately check the saved state of the file. Use this for validation checks.
    The DataFrame is accessible as 'df'.
    """
    path = os.path.normpath(path)
    if not os.path.exists(path):
        return f"❌ File not found: {path}"

    ext = os.path.splitext(path)[-1].lower()
    try:
        # *** THE KEY DIFFERENCE ***
        # Load all columns as strings to prevent automatic type inference by pandas.
        # This allows us to see the "raw" state of the saved CSV.
        if ext == ".csv":
            df = pd.read_csv(path, encoding='utf-8', dtype=str)
        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(path, dtype=str)
        else:
            return f"❌ Unsupported file type: {ext}"
    except Exception as e:
        return f"❌ Error reading file '{path}': {e}"

    local_namespace = {"df": df, "pd": pd}
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        exec(query, {}, local_namespace)
        output = captured_output.getvalue()

        # NOTE: There is NO code here to save the dataframe. This is read-only.

        return f"✅ Inspection executed successfully.\n{output or 'No print output.'}"
    except Exception as e:
        return f"❌ Execution failed: {e}"
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

class PythonInputs(BaseModel):
    query: str = Field(description="A valid python command to run.")


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
