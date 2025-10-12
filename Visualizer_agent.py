from dotenv import load_dotenv
from Toolkit.Tools import eda_fact_sheet, python_repl_ast
from llm import llm_model
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
import time
import re  # +++ IMPORTED for parsing file paths
import os  # +++ IMPORTED to verify file paths

load_dotenv()

from Prompts import Visualizer_prompt

# +++ MODIFIED function signature to accept output_dir +++
def Visualizer_agent(df_path: str, output_dir: str):

    system_prompt = ChatPromptTemplate.from_messages(
        [
            (
            "system",
                "You are a Business report agent. You have access to a tool `eda_fact_sheet(df_path)` which generates a full fact sheet for a CSV/DataFrame. "
                "The csv is in this path : {df_path}. Use the tool first, then reason about the factsheet and decide the impactful visual plots can be made from the data."
                "Use the python_repl_ast tool to perform the plots and output 5-7 meaningful plots from the data."
            ),
            # Chat history is not used in the invoke call, can be removed for simplicity
            # ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")        
        ]
    )

    # +++ MODIFIED task_prompt to include the output directory instruction +++
    task_prompt = (
            f"Follow this {Visualizer_prompt} to the letter. Access the data from the path '{df_path}'. "
            f"You MUST save all generated plot files into this specific directory: '{output_dir}'. "
            "This path is also available as the `output_dir` variable inside the python tool."
        )

    visualizer_agent = create_tool_calling_agent(
                llm=llm_model,
                tools=[eda_fact_sheet, python_repl_ast],
                prompt=system_prompt
            )

    # +++ MODIFIED AgentExecutor to pass the output_dir to the python tool's environment +++
    # This makes the `output_dir` variable available inside the executed python code
    agent_executor = AgentExecutor(
                agent=visualizer_agent,
                tools=[eda_fact_sheet, python_repl_ast],
                verbose=True,
                handle_parsing_errors=True,
                # return_intermediate_steps=True # Not strictly needed for the final output
                tool_run_kwargs={"tool_input": {"output_dir": output_dir}}
            )

    max_attempts = 3
    attempt = 0

    while attempt < max_attempts:
        attempt += 1
        print(f"--- Starting Visualization Generation, Attempt {attempt} of {max_attempts} ---")

        try:
            result = agent_executor.invoke({
                "input": task_prompt,
                "df_path": df_path,
                # "chat_history": [] # Not used
            })

            if not result or "output" not in result or not result["output"]:
                raise ValueError("Agent executed successfully but produced an empty or invalid output.")
            
            # +++ NEW: Post-processing and path extraction logic +++
            final_report_text = result["output"]
            print("--- Agent finished. Parsing final report for file paths. ---")

            # Use regex to robustly find all file paths in the format (File: ...)
            # This handles whitespace and variations. It captures the path inside the parentheses.
            image_paths = re.findall(r"\(File:\s*([^)]+)\)", final_report_text)
            
            if not image_paths:
                # If no paths are found, it might be an error or a different output format.
                # We raise this as a failure for the retry logic.
                raise ValueError("No file paths in the format (File: ...) were found in the final report.")

            # Clean up paths (remove leading/trailing spaces) and verify they actually exist
            verified_paths = []
            for path in image_paths:
                clean_path = path.strip()
                if os.path.exists(clean_path):
                    verified_paths.append(clean_path)
                else:
                    print(f"Warning: Agent reported path '{clean_path}', but it was not found on disk.")
            
            if not verified_paths:
                raise ValueError("Agent reported file paths, but none of them could be verified on the filesystem.")

            print(f"--- Successfully found and verified {len(verified_paths)} plot files. ---")
            
            # Return a structured dictionary with the report and the clean list of paths
            return {"report": final_report_text, "paths": verified_paths}

        except Exception as e:
            error_msg = (
                f"Attempt {attempt} failed: {str(e)}. "
                "Please analyze the error and retry. Ensure your final output is a markdown report "
                "containing file paths in the format `(File: /absolute/path/to/plot.png)`."
            )
            print(f"--- Runtime error or parsing failure ---\n{error_msg}")
            
            task_prompt = (
                f"Your previous attempt failed with this error: {error_msg}. "
                f"Please correct your approach. Original task:\n---\n{Visualizer_prompt}"
            )
            time.sleep(3)

    # Fallback if all attempts fail
    error_message = (
        f"Error: Visualizer agent failed to generate and verify plots after {max_attempts} attempts."
    )
    print(error_message)
    # Return a consistent dictionary format on failure
    return {"report": error_message, "paths": []}