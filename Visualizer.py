from dotenv import load_dotenv
from Toolkit.Tools import eda_fact_sheet,python_repl_ast
from llm import llm_model
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from Prompts import Visualizer_prompt

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

    Instructions = "Data is about sales, perform the EDA , here's the path = {path}"

    task_prompt = (
            f"follow this {Visualizer_prompt} to the letter."
        )

    Analyzer_agent = create_tool_calling_agent(
                llm=llm_model,
                tools=[eda_fact_sheet,python_repl_ast],
                prompt=system_prompt
            )

    tools = [eda_fact_sheet,python_repl_ast]



    agent_executor = AgentExecutor(
                agent=Analyzer_agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors=True,
                return_intermediate_steps=True
            )

    result = agent_executor.invoke({
                    "input": task_prompt,
                    "df_path": df_path,
                    "chat_history": []     
                })
    return result

