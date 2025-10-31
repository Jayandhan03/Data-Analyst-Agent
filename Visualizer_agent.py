from dotenv import load_dotenv
from Toolkit.Tools import eda_fact_sheet, python_repl_ast
from llm import llm_model
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
import time
import re,os
from langchain.output_parsers import PydanticOutputParser,OutputFixingParser
from Guardrails.visualizer import VisualizationReportOutput

load_dotenv()

from Prompts import Visualizer_prompt

def Visualizer_agent(df_path: str, output_dir: str):
    """
    Synchronous agent that generates and saves data visualizations.
    """
    parser = PydanticOutputParser(pydantic_object=VisualizationReportOutput)
    fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm_model)

    system_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a data visualization expert. Your primary goal is to use tools to generate plot files and then structure your findings into a single, clean JSON object as your final answer. Do not add any extra commentary outside of the JSON."
            ),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ]
    )

    task_prompt = (
        f"Follow this instruction set to the letter: {Visualizer_prompt}. Access the data from the path '{df_path}'. "
        f"You MUST save all generated plot files into this specific directory: '{output_dir}'. "
        "This path is also available as the `output_dir` variable inside the python tool."
    )

    visualizer_agent = create_tool_calling_agent(
        llm=llm_model,
        tools=[eda_fact_sheet, python_repl_ast],
        prompt=system_prompt
    )

    agent_executor = AgentExecutor(
        agent=visualizer_agent,
        tools=[eda_fact_sheet, python_repl_ast],
        verbose=True,
        handle_parsing_errors=True,
        # tool_run_kwargs is not needed here as variables are passed in the prompt
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
            })

            final_output_string = result.get("output", "")
            if not final_output_string:
                raise ValueError("Agent executed successfully but produced an empty output.")

            parsed_output: VisualizationReportOutput = fixing_parser.parse(final_output_string)

            # Verify that the files listed in the report actually exist on disk
            verified_visuals = []
            for vis in parsed_output.visualizations:
                # FIXED: Changed vis.image_path to vis.file_path to match the Pydantic model
                if os.path.exists(vis.file_path):
                    verified_visuals.append(vis)
                else:
                    print(f"Warning: Image path from report '{vis.file_path}' not found on disk.")

            if not verified_visuals:
                raise ValueError("Agent reported plot file paths, but none were found on disk.")

            # Update the parsed output to only include verified files
            parsed_output.visualizations = verified_visuals

            print(f"--- Visualization Generation successful on Attempt {attempt} ---")
            return {
                "success": True,
                "parsed_visuals": parsed_output.model_dump(),
                "raw_output": final_output_string,
            }

        except Exception as e:
            error_msg = (
                f"Attempt {attempt} failed with error: {str(e)}. "
                f"Please re-analyze and strictly follow the JSON output schema:\n"
                f"{VisualizationReportOutput.model_json_schema()}"
            )
            print(f"--- Runtime/Parsing error encountered ---\n{error_msg}")

            task_prompt = (
                f"The previous attempt failed with this error: {error_msg}. "
                f"Correct your output and retry. Original task:\n---\n{Visualizer_prompt}"
            )

            time.sleep(5)

    error_message = (
        f"Error: Visualizer agent failed to generate valid visualizations for dataset at '{df_path}' "
        f"after {max_attempts} attempts."
    )
    print(error_message)

    return {
        "success": False,
        "error": error_message,
        "output": "Failed to generate visualizations after retries.",
    }