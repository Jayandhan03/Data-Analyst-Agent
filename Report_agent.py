from langchain_core.tools import tool
from dotenv import load_dotenv
import sys,os
from io import StringIO
from pydantic import BaseModel, Field
import pandas as pd
from typing import Optional
from Toolkit.Tools import eda_fact_sheet,python_repl_ast
from llm import llm_model
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from Prompts import Reporter_prompt



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

    Instructions = "Data is about sales, perform the EDA , here's the path = {path}"

    task_prompt = (
            f"Find the instructions given by the user here : {Instructions} and follow this {Reporter_prompt} to the letter."
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