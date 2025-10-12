from dotenv import load_dotenv
from Toolkit.Tools import eda_fact_sheet,python_repl_ast
from llm import llm_model
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from Prompts import Visualizer_prompt
import time

load_dotenv()

def Visualizer_agent(df_path: str):

    system_prompt = ChatPromptTemplate.from_messages(
        [
            (
            "system",
                "You are a Business report agent. You have access to a tool `eda_fact_sheet(df_path)` which generates a full fact sheet for a CSV/DataFrame. "
                "The csv is in this path : {df_path}. Use the tool first, then reason about the factsheet and decide the impactful visual plots can be made from the data."
                "Use the python_repl_ast tool to perform the plots and output 5-7 meaningful plots from the data."
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")        
    
        ]
    )

    task_prompt = (
            f"follow this {Visualizer_prompt} to the letter and access the data from here {df_path}"
        )

    visualizer_agent = create_tool_calling_agent(
                llm=llm_model,
                tools=[eda_fact_sheet,python_repl_ast],
                prompt=system_prompt
            )

    agent_executor = AgentExecutor(
                agent=visualizer_agent,
                tools=[eda_fact_sheet,python_repl_ast],
                verbose=True,
                handle_parsing_errors=True,
                return_intermediate_steps=True
            )

    max_attempts = 3
    attempt = 0

    # --- Execution loop with retry logic ---
    while attempt < max_attempts:
        attempt += 1
        print(f"--- Starting Visualization Generation, Attempt {attempt} of {max_attempts} ---")

        try:
            # 1. Invoke the agent
            result = agent_executor.invoke({
                "input": task_prompt,
                "df_path": df_path,
                "chat_history": []
            })

            # 2. Check for a valid, non-empty output
            if result and "output" in result and result["output"]:
                print(f"--- Visualization Generation successful on Attempt {attempt} ---")
                return result  # Return the entire successful result dictionary
            else:
                # Handle cases where the agent runs but produces no meaningful output
                raise ValueError("Agent executed successfully but produced an empty or invalid output.")

        except Exception as e:
            # 3. On failure, create a detailed error message for the next attempt
            error_msg = (
                f"Attempt {attempt} failed with the following error: {str(e)}. "
                f"Please carefully analyze the error and your previous steps, then try again. "
                "Remember your process: first call `eda_fact_sheet`, then use `python_repl_ast` to generate and save the plots."
            )
            print(f"--- Runtime error encountered ---\n{error_msg}")

            # 4. Prepend the error to the prompt for the next retry, giving the agent context
            task_prompt = (
                f"Your previous attempt failed with this error: {error_msg}. "
                f"Please correct your approach and retry. Here is the original task:\n---\n{task_prompt}"
            )
            
            time.sleep(5)  # Optional delay to prevent rapid-fire retries

    # 5. If all attempts fail, return a final error message (fallback)
    error_message = (
        f"Error: Visualizer agent failed to generate plots for the dataset at '{df_path}' "
        f"after {max_attempts} attempts. Please check the data and agent configuration."
    )
    print(error_message)
    return {"error": error_message, "output": "Failed to generate visualizations."}