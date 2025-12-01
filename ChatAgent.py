import pandas as pd
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from llm import llm_model  # Importing your existing LLM configuration

def Chat_Data_Agent(df_path: str, query: str):
    """
    Agent that converts a CSV into a temporary SQL database and answers 
    user queries regarding the data using an SQL Agent.
    """
    print(f"--- Initializing Chat Agent for file: {df_path} ---")

    try:
        # 1. Load the CSV Data
        # We try 'utf-8' first, then 'latin1' to handle common encoding errors
        try:
            df = pd.read_csv(df_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(df_path, encoding='latin1')

        # 2. Sanitize Column Names
        # SQL agents struggle with spaces in column names, so we replace them with underscores
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace(r'[^\w]', '', regex=True)

        # 3. Create a Temporary SQL Database
        # We create a local sqlite file to act as the database
        db_engine = create_engine("sqlite:///temp_chat_db.db")
        
        # Write the dataframe to the SQL table named 'data_table'
        df.to_sql("data_table", db_engine, index=False, if_exists='replace')

        # 4. Initialize LangChain SQL Tools
        db = SQLDatabase(engine=db_engine)

        # 5. Create the SQL Agent
        # agent_type="openai-tools" is efficient for models that support function calling
        agent_executor = create_sql_agent(
            llm=llm_model,
            db=db,
            agent_type="openai-tools",
            verbose=True,
            handle_parsing_errors=True
        )

        # 6. Run the Query
        print(f"--- Executing Query: {query} ---")
        
        # We wrap the user query to ensure the agent looks at the specific table
        formatted_prompt = (
            f"Input: {query}. \n"
            f"Query the table 'data_table' to find the answer. "
            f"If the result is a long list, limit it to the top 5 items."
        )

        result = agent_executor.invoke({"input": formatted_prompt})

        # 7. Return Structured Output
        output_text = result.get("output", "I could not generate an answer.")
        
        return {
            "success": True,
            "response": output_text,
            "query_processed": query
        }

    except Exception as e:
        error_msg = f"Error during Chat with Data execution: {str(e)}"
        print(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "response": "Sorry, I encountered an internal error while processing your data."
        }