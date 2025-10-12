from dotenv import load_dotenv
from Toolkit.Tools import eda_fact_sheet,python_repl_ast
from llm import llm_model
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from Prompts import Reporter_prompt
import time


load_dotenv()

def Report_agent(df_path: str):

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
                tools=[eda_fact_sheet,python_repl_ast],
                prompt=system_prompt
            )

    tools = [eda_fact_sheet,python_repl_ast]

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
            result = agent_executor.invoke({
                            "input": task_prompt,
                            "df_path": df_path,
                            "chat_history": []     
                        })
            
            if result and "output" in result and result["output"]:
                print(f"--- Report Generation successful on Attempt {attempt} ---")
                return result # Return the entire successful result dictionary
            else:
                # This handles cases where the agent runs but produces no output content
                raise ValueError("Agent executed successfully but produced an empty output.")
            
        except Exception as e:
            # 3. On failure, create an error message for the next attempt
            error_msg = (
                f"Attempt {attempt} failed with the following error: {str(e)}. "
                f"Please analyze the error and your previous steps, then try again. "
                "Ensure you call the `eda_fact_sheet` tool first and then generate the report based on its findings."
            )
            print(f"--- Runtime error encountered ---\n{error_msg}")

            # Prepend the error to the prompt for the next retry
            task_prompt = (
                f"Your previous attempt failed with this error: {error_msg}. "
                f"Please correct your approach and try again. Here is the original task:\n---\n{task_prompt}"
            )
            
            time.sleep(5) # Optional delay before retrying

    # 4. If all attempts fail, return an error message
    error_message = f"Error: Report agent failed to generate a report for the dataset at '{df_path}' after {max_attempts} attempts. Please check the data and agent configuration."
    print(error_message)
    return {"error": error_message, "output": "Failed to generate report."}   

