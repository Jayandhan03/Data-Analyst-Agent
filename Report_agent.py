from dotenv import load_dotenv
from Toolkit.Tools import eda_fact_sheet, python_repl_ast
from llm import llm_model
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from Prompts import Reporter_prompt
import time  
from langchain.output_parsers import PydanticOutputParser,OutputFixingParser
from Guardrails.report import BusinessReportOutput

load_dotenv()

def Report_agent(df_path: str):
    """
    Synchronous agent that generates a business report from a dataset.
    """
    parser = PydanticOutputParser(pydantic_object=BusinessReportOutput)
    fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm_model)
    
    system_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a Business report agent. You have access to a tool `eda_fact_sheet(df_path)` which generates a full fact sheet for a CSV/DataFrame. "
                "The csv is in this path : {df_path}. Use the tool first, then provide a comprehensive business report based on the dataset analysis."
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ]
    )

    task_prompt = (
        f"Follow this {Reporter_prompt} to the letter and access the data from here {df_path}"
    )

    reporter_agent = create_tool_calling_agent(
        llm=llm_model,
        tools=[eda_fact_sheet, python_repl_ast],
        prompt=system_prompt
    )

    tools = [eda_fact_sheet, python_repl_ast]

    agent_executor = AgentExecutor(
        agent=reporter_agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )

    max_attempts = 3
    attempt = 0

    while attempt < max_attempts:
        attempt += 1
        print(f"--- Starting Report Generation, Attempt {attempt} of {max_attempts} ---")

        try:
            # Use the synchronous 'invoke' method
            result = agent_executor.invoke({
                "input": task_prompt,
                "df_path": df_path,
                "chat_history": []
            })

            # 2. Extract and Parse Output
            final_output_string = result.get("output", "")
            if not final_output_string:
                raise ValueError("Agent executed successfully but produced an empty output.")

            parsed_output: BusinessReportOutput = fixing_parser.parse(final_output_string)

            # 3. If parsing succeeds, return structured output
            print(f"--- Report Generation successful on Attempt {attempt} ---")
            return {
                "success": True,
                "parsed_report": parsed_output.model_dump(),
                "raw_output": final_output_string
            }

        except Exception as e:
            # 4. Handle and log parsing/runtime errors
            error_msg = (
                f"Attempt {attempt} failed with error: {str(e)}. "
                f"Please re-analyze the data and strictly follow the output schema:\n"
                f"{BusinessReportOutput.model_json_schema()}"
            )
            print(f"--- Runtime/Parsing error encountered ---\n{error_msg}")

            # 5. Inject error context into next prompt
            task_prompt = (
                f"The previous attempt failed with this error: {error_msg}. "
                f"Correct your output format and retry the task. Here is the original instruction:\n---\n{task_prompt}"
            )

            time.sleep(5)

    # --- Fallback if all attempts fail ---
    error_message = (
        f"Error: Report agent failed to generate a valid report for dataset at '{df_path}' "
        f"after {max_attempts} attempts. Please review the dataset or agent configuration."
    )
    print(error_message)

    return {
        "success": False,
        "error": error_message,
        "output": "Failed to generate report after retries."
    }
