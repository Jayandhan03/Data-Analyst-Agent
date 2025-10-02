from typing import Literal,Any
from langgraph.types import Command
from langgraph.graph import END
from langgraph.graph.message import add_messages
from typing_extensions import  Annotated
from llm import llm_model
from pydantic import BaseModel,Field
from typing import List, Any, Literal, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from Prompts import supervisor_prompt,PreprocessingPlanner_prompt,cleaner_prompt,validation_prompt,Reporter_prompt
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from Toolkit.Tools import python_repl_ast,eda_fact_sheet,text_to_pdf
from Guardrails.Guardrail import summarize_tool_output
from Guardrails.Preprocessing import StructuredPlanOutput
from langchain.output_parsers import PydanticOutputParser


class Router(BaseModel):
    next: Literal["PreprocessingPlanner_node","Cleaner_node","Validation_node","Reporter_node",END]= Field(description="The next node to route to. Must be one of the available nodes.")
    reasoning: str = Field(description="A short reasoning for the decision made.")

class AgentStateModel(BaseModel):
    messages: Optional[Annotated[List[Any], add_messages]] = None
    Instructions: Optional[str] = None
    Analysis: Optional[List[dict]] = None
    Preprocessing: Optional[List[dict]] = None
    validation: Optional[List[dict]] = None
    Report: Optional[List[dict]] = None
    Path: Optional[str] = None
    next: Optional[str] = None
    current_reasoning: Optional[str] = None
    
class DataAnalystAgent:
    def __init__(self):
        self.llm_model = llm_model

    def supervisor_node(self,state:AgentStateModel) -> Command[Literal["PreprocessingPlanner_node","Cleaner_node","Validation_node","Reporter_node", END]]:

        print("**************************below is my state right after entering****************************")
        print(state)

        human_instruction = (
        f"Here is the current state for your evaluation:\n"
        f"- Analysis: {state.Analysis}\n"
        f"- Preprocessing: {state.Preprocessing}\n"
        f"- Validation: {state.validation}\n"
        f"- Report: {state.Report}"
    )
        
        messages_for_llm = [
            SystemMessage(content=supervisor_prompt),
            HumanMessage(content=human_instruction),
        ]
        
        if state.messages:
            last_message = state.messages[-1]
            messages_for_llm.append(last_message)
            print(f"--- Attaching last message ---\nType: {type(last_message).__name__}\nContent: {last_message.content}\n----------------------------------")

        print("***********************Invoking LLM for routing decision************************")
        
        # response = self.llm_model.with_structured_output(Router).invoke(messages_for_llm)

        chain = self.llm_model.with_structured_output(Router).with_retry(
            stop_after_attempt=3
        )
        
        # Now invoke the complete, resilient chain
        response = chain.invoke(messages_for_llm)
        
        goto = response.next
        
        print("********************************this is my goto*************************")
        print(goto)
        
        print("********************************")
        print(response.reasoning)
            
        if goto == "END":
            goto = END 
            
        print("**************************below is my state****************************")
        print(state)
        
        return Command(goto=goto, update={'next': goto, 
                                        'current_reasoning': response.reasoning}
                    )

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
        
        # agent_dict_output = {"final_answer": result.get("output")}

        # parsed = summarize_tool_output(agent_dict_output,self.llm_model)

        # parsed_str = f"{parsed.summary}\n{parsed.details}"

        # 5. Update the state with the structured and summarized data
        return Command(
            update={
                # Update messages with the human-readable summary
                "messages": state.messages[-1:] + [AIMessage(content=summary_str, name="PreprocessingPlanner_node")],
                # Update Analysis with the structured, machine-readable plan
                "Analysis": [{"final_answer": plan_dict}]
            },
            goto="supervisor",
        )

    def Cleaner_node(self, state: AgentStateModel) -> Command[Literal['supervisor']]:
        print("*****************called cleaner node************")

        Analysis = state.Analysis

        Path = state.Path

        task_prompt = (
        f"""Find the preprocessing steps for every column here : {Analysis} and follow this {cleaner_prompt} to the letter.
        The dataframe is in this path: {Path} modify the data in this path directly."""
    )
        print(f"--- Sending this direct task to the agent ---\n{task_prompt}\n---------------------------------------------")

        system_prompt = ChatPromptTemplate.from_messages(
    [
        (
           "system",
            "You are a DataFrame cleaner agent. You have access to a tool `python_repl_ast(query)` to modify the dataframe CSV/DataFrame."
            "Follow the instructions : {cleaner_prompt} to the letter and make the necessary changes to the dataframe."
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")        
    ]
)
        Cleaner_agent = create_tool_calling_agent(
            llm=self.llm_model,
            tools=[python_repl_ast],
            prompt=system_prompt
        )

        tools = [python_repl_ast]

        agent_executor = AgentExecutor(
            agent=Cleaner_agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )

        result = agent_executor.invoke({
                "input": task_prompt,
                "cleaner_prompt": cleaner_prompt,
                "chat_history": []     
            })
        
        agent_dict_output = {"final_answer": result.get("output")}

        parsed = summarize_tool_output(agent_dict_output,self.llm_model)

        parsed_str = f"{parsed.summary}\n{parsed.details}"

        return Command(
            update={
                "messages": state.messages[-1:] + [AIMessage(content=parsed_str, name="cleaner_node")],
                "Preprocessing": [agent_dict_output] 
            },
            goto="supervisor",
        )
    
    def Validation_node(self, state: AgentStateModel):
        print("*****************INVOKING VALIDATION NODE************")

        analysis_plan = state.Analysis
        file_path = state.Path

        # This becomes the specific mission for this run.
        task_input = (
            f"Validation Mission:\n"
            f"1.  **Review the Plan:** Here is the preprocessing plan you must validate:\n{analysis_plan}\n\n"
            f"2.  **Inspect the Asset:** The CSV file you must inspect is located at the following path:\n{file_path}\n\n"
            f"3.  **Execute Validation:** Use the `python_repl_ast` tool to run checks for each action in the plan. "
            f"Follow your core instructions from the system prompt precisely. Report your findings as instructed."
        )

        print(f"--- Sending this direct task to the agent ---\n{task_input}\n---------------------------------------------")

        # The validation_prompt is the agent's core identity/persona. It belongs in the system message.
        system_prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                "system",
                    "{validation_prompt}" # Pass the detailed playbook here
                ),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ]
        )

        validator_agent = create_tool_calling_agent(
            llm=self.llm_model,
            tools=[python_repl_ast],
            prompt=system_prompt_template
        )

        agent_executor = AgentExecutor(
            agent=validator_agent,
            tools=[python_repl_ast],
            verbose=True,
            handle_parsing_errors=True,
        )

        # Invoke the agent with the clear separation of roles
        result = agent_executor.invoke({
                "input": task_input,
                "validation_prompt": validation_prompt, # The detailed playbook is passed to the system prompt
                "chat_history": []
            })
        
        # ... The rest of your function to process and return the result remains the same ...
        agent_dict_output = {"final_answer": result.get("output")}
        parsed = summarize_tool_output(agent_dict_output, self.llm_model)
        parsed_str = f"{parsed.summary}\n{parsed.details}"

        return Command(
            update={
                "messages": state.messages[-1:] + [AIMessage(content=parsed_str, name="Validation_node")],
                "validation": [agent_dict_output]
            },
            goto="supervisor",
        )
    
    def Reporter_node(self, state: AgentStateModel) -> Command[Literal['supervisor']]:
        print("*****************called Reporter node************")

        Instructions = state.Instructions

        df_path = state.Path

        task_prompt = (
        f"Find the instructions given by the user here : {Instructions} and follow this {Reporter_prompt} to the letter."
    )
        print(f"--- Sending this direct task to the agent ---\n{task_prompt}\n---------------------------------------------")

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
        Analyzer_agent = create_tool_calling_agent(
            llm=self.llm_model,
            tools=[eda_fact_sheet,python_repl_ast,text_to_pdf],
            prompt=system_prompt
        )

        tools = [eda_fact_sheet,python_repl_ast,text_to_pdf]

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
        
        agent_dict_output = {"final_answer": result.get("output")}

        parsed = summarize_tool_output(agent_dict_output,self.llm_model)

        parsed_str = f"{parsed.summary}\n{parsed.details}"

        return Command(
            update={
                "messages": state.messages[-1:] + [AIMessage(content=parsed_str, name="Reporter_node")],
                "Report": [agent_dict_output] 
            },
            goto="supervisor",
        )
    
#Data is about sales,provide the data overview along with the preprocessing steps needed to perform EDA , here's the path path = r"D:\Code Assistant\Bakery sale.csv"
#dbuser1 user and pass for cluster
