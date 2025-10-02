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
from Guardrails.cleaner import CleaningSummary  
from Guardrails.validation import ValidationResult
from Guardrails.report import BusinessReport
from langchain.output_parsers import PydanticOutputParser,OutputFixingParser


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

        parser = PydanticOutputParser(pydantic_object=Router)

        fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_model)

        # Build the retry chain
        chain = self.llm_model | fixing_parser

        # Add retries
        max_attempts = 3
        attempt = 0
        error_msg = None
        response = None

        while attempt < max_attempts:
            attempt += 1
            print(f"--- Attempt {attempt} ---")

            # Compose messages for this attempt
            messages_for_this_attempt = list(messages_for_llm)
            if error_msg:
                # Inject previous error info to let LLM know what failed
                messages_for_this_attempt.append(HumanMessage(content=f"Previous attempt failed due to: {error_msg}. Please follow the schema strictly: {Router.schema_json()}"))

            try:
                response = chain.invoke(messages_for_this_attempt)
                # If parse succeeds, break the loop
                break
            
            except Exception as e:
                error_msg = str(e)
                print(f"--- Error on attempt {attempt}: {error_msg} ---")
                # If last attempt, will exit loop and propagate error

            if response is None:
                # All retries failed, fallback error
                fallback_msg = f"All {max_attempts} attempts failed. Last error: {error_msg}"
                print(f"--- Supervisor node failed ---\n{fallback_msg}")
                return Command(
                    goto="END",
                    update={
                        "next": "END",
                        "current_reasoning": fallback_msg
                    }
                )

        # Invoke chain with error-aware retries
        # response = chain.invoke(messages_for_llm)

        # chain = self.llm_model.with_structured_output(Router).with_retry(
        #     stop_after_attempt=3
        # )
        # Now invoke the complete, resilient chain
        # try:
        # # First attempt
        #   response = chain.invoke(messages_for_llm)

        # except Exception as e:
        #     # Catch runtime errors (rate limit, timeout, etc.)
        #     error_msg = (
        #         f"Previous attempt failed due to runtime error: {str(e)}. "
        #         f"Please try again strictly following schema: {Router.schema_json()}"
        #     )
        #     print(f"--- Runtime error encountered, re-prompting LLM ---\n{error_msg}")
            
        #     # Retry by injecting the error into messages
        #     response = chain.invoke(messages_for_llm + [HumanMessage(content=error_msg)])

        
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

        fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_model)

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

        # 5. Wrap execution in a retry loop
        max_attempts = 3
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            try:
                result = agent_executor.invoke({
                    "input": task_prompt,
                    "chat_history": []
                })

                # Try parsing the final output
                final_output_string = result.get("output", "")
                parsed_output: StructuredPlanOutput = fixing_parser.parse(final_output_string)

                # Successfully parsed → extract plan and summary
                plan_dict = {"plan": [step.model_dump() for step in parsed_output.plan]}
                summary_str = f"{parsed_output.summary}\n{parsed_output.details}"

                # Update state and return
                return Command(
                    update={
                        "messages": state.messages[-1:] + [
                            AIMessage(content=summary_str, name="PreprocessingPlanner_node")
                        ],
                        "Analysis": [{"final_answer": plan_dict}]
                    },
                    goto="supervisor",
                )

            except Exception as e:
                error_msg = (
                    f"Attempt {attempt} failed due to error: {str(e)}. "
                    f"Please strictly follow the schema: {StructuredPlanOutput.model_json_schema()}"
                )
                print(f"--- Runtime/Parsing error encountered ---\n{error_msg}")
                # Inject error into prompt for next retry
                task_prompt = task_prompt + "\n\n" + error_msg

        # If all attempts fail, fallback to supervisor with error message
        return Command(
            update={
                "messages": state.messages[-1:] + [
                    AIMessage(content="Error: The analysis agent failed to produce a valid preprocessing plan after multiple attempts.", 
                            name="Analyzer_node_Error")
                ],
                "Analysis": [{"error": "Parsing failed after retries"}]
            },
            goto="supervisor",
        )

        # # Build a robust chain with retry
        # chain = self.llm_model | fixing_parser
        # chain = chain.with_retry(stop_after_attempt=3)

        # # 3. Wrap the LLM execution with retry + structured output enforcement
        # # chain = self.llm_model.with_structured_output(StructuredPlanOutput).with_retry(
        # #     stop_after_attempt=3
        # # )

        # # Try-catch with LLM-aware error injection

        # messages_for_llm = ({
        #             "input": task_prompt,
        #             "chat_history": []     
        #         })
        # try:
        #     response = chain.invoke(messages_for_llm)
        # except Exception as e:
        #     error_msg = (
        #         f"Previous attempt failed due to runtime error: {str(e)}. "
        #         f"Please try again strictly following schema: {StructuredPlanOutput.schema_json()}"
        #     )
        #     print(f"--- Runtime error encountered, re-prompting LLM ---\n{error_msg}")
        #     response = chain.invoke(messages_for_llm + [HumanMessage(content=error_msg)])

            # result = agent_executor.invoke({
            #         "input": task_prompt,
            #         "chat_history": []     
            #     })
        
        # 4. Parse the agent's final string output using the Pydantic parser
        # This acts as your guardrail. If the output is not valid JSON, it will fail here.
        # try:
        #     final_output_string = result.get("output", "")
        #     parsed_output: StructuredPlanOutput = parser.parse(final_output_string)
            
        #     # Extract the components from the parsed object
        #     # --- THIS LINE IS CORRECTED ---
        #     plan_dict = {"plan": [step.model_dump() for step in parsed_output.plan]}
        #     # ---------------------------
        #     summary_str = f"{parsed_output.summary}\n{parsed_output.details}"

        # except Exception as e:
        #     # Handle cases where the LLM failed to produce valid JSON
        #     print(f"--- FAILED TO PARSE AGENT OUTPUT ---\nError: {e}")
        #     print(f"Raw Output:\n{result.get('output')}")
        #     # Fallback or error state
        #     return Command(
        #         update={
        #             "messages": state.messages[-1:] + [AIMessage(content="Error: The analysis agent failed to produce a valid preprocessing plan.", name="Analyzer_node_Error")],
        #             "Analysis": [{"error": "Parsing failed", "output": result.get('output')}] 
        #         },
        #         goto="supervisor",
        #     )
        
        # agent_dict_output = {"final_answer": result.get("output")}

        # parsed = summarize_tool_output(agent_dict_output,self.llm_model)

        # parsed_str = f"{parsed.summary}\n{parsed.details}"

        # 5. Update the state with the structured and summarized data
        # return Command(
        #     update={
        #         # Update messages with the human-readable summary
        #         "messages": state.messages[-1:] + [AIMessage(content=summary_str, name="PreprocessingPlanner_node")],
        #         # Update Analysis with the structured, machine-readable plan
        #         "Analysis": [{"final_answer": plan_dict}]
        #     },
        #     goto="supervisor",
        # )

    def Cleaner_node(self, state: AgentStateModel) -> Command[Literal['supervisor']]:

        print("*****************called cleaner node************")

        Analysis = state.Analysis

        Path = state.Path

        # 1. Instantiate the parser for our new structured output
        parser = PydanticOutputParser(pydantic_object=CleaningSummary)

        fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_model)

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

        # 5. Wrap execution in a retry loop for robust parsing
        max_attempts = 3
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            try:
                result = agent_executor.invoke({
                    "input": task_prompt,
                    "cleaner_prompt": cleaner_prompt,
                    "chat_history": []
                })

                # Try parsing the final output using the fixing parser
                final_output_string = result.get("output", "")
                parsed_output: CleaningSummary = fixing_parser.parse(final_output_string)

                # Successfully parsed → format summary string and update state
                parsed_str = f"{parsed_output.summary}\n{parsed_output.details}"

                return Command(
                    update={
                        "messages": state.messages[-1:] + [AIMessage(content=parsed_str, name="cleaner_node")],
                        "Preprocessing": [{"final_answer": final_output_string}]
                    },
                    goto="supervisor",
                )

            except Exception as e:
                error_msg = (
                    f"Attempt {attempt} failed due to a parsing error: {str(e)}. "
                    f"You MUST provide a final answer that is a valid JSON object. Please strictly follow this schema: "
                    f"{CleaningSummary.model_json_schema()}"
                )
                print(f"--- Runtime/Parsing error encountered ---\n{error_msg}")
                # Inject the error and schema into the prompt for the next retry
                task_prompt = task_prompt + "\n\n" + error_msg

        # If all attempts fail, fallback to the supervisor with an error message
        return Command(
            update={
                "messages": state.messages[-1:] + [
                    AIMessage(content="Error: The cleaner agent failed to produce a valid summary after multiple attempts.",
                              name="Cleaner_node_Error")
                ],
                "Preprocessing": [{"error": "Parsing failed after all retries"}]
            },
            goto="supervisor",
        )

        # result = agent_executor.invoke({
        #         "input": task_prompt,
        #         "cleaner_prompt": cleaner_prompt,
        #         "chat_history": []     
        #     })
        
        # agent_dict_output = {"final_answer": result.get("output")}

        # parsed = summarize_tool_output(agent_dict_output,self.llm_model)

        # parsed_str = f"{parsed.summary}\n{parsed.details}"

        # return Command(
        #     update={
        #         "messages": state.messages[-1:] + [AIMessage(content=parsed_str, name="cleaner_node")],
        #         "Preprocessing": [agent_dict_output] 
        #     },
        #     goto="supervisor",
        # )
    
    def Validation_node(self, state: AgentStateModel):
        print("*****************INVOKING VALIDATION NODE************")

        analysis_plan = state.Analysis

        file_path = state.Path

        # 1. Instantiate the parser for our new structured output
        parser = PydanticOutputParser(pydantic_object=ValidationResult)

        fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_model)

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

        max_attempts = 3
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            try:
                result = agent_executor.invoke({
                    "input": task_input,
                    "validation_prompt": validation_prompt, # Pass the playbook to the system prompt
                    "chat_history": []
                })

                # Try parsing the final output using the fixing parser
                final_output_string = result.get("output", "")

                parsed_output: ValidationResult = fixing_parser.parse(final_output_string)

                # Successfully parsed → update state with the clear message from the agent
                return Command(
                    update={
                        "messages": state.messages[-1:] + [AIMessage(content=parsed_output.message, name="Validation_node")],
                        "validation": [{"final_answer": parsed_output.model_dump()}] # Store the structured result
                    },
                    goto="supervisor",
                )

            except Exception as e:
                error_msg = (
                    f"Attempt {attempt} failed due to a parsing error: {str(e)}. "
                    f"You MUST provide a final answer that is a valid JSON object. Please strictly follow this schema: "
                    f"{ValidationResult.model_json_schema()}"
                )
                print(f"--- Runtime/Parsing error encountered ---\n{error_msg}")
                # Inject the error and schema into the prompt for the next retry
                task_input = task_input + "\n\n" + error_msg

        # If all attempts fail, fallback to the supervisor with an error message
        return Command(
            update={
                "messages": state.messages[-1:] + [
                    AIMessage(content="Error: The validation agent failed to produce a valid result after multiple attempts.",
                            name="Validation_node_Error")
                ],
                "validation": [{"error": "Parsing failed after all retries"}]
            },
            goto="supervisor",
        )

        # Invoke the agent with the clear separation of roles
        # result = agent_executor.invoke({
        #         "input": task_input,
        #         "validation_prompt": validation_prompt, # The detailed playbook is passed to the system prompt
        #         "chat_history": []
        #     })
        
        # # ... The rest of your function to process and return the result remains the same ...
        # agent_dict_output = {"final_answer": result.get("output")}
        # parsed = summarize_tool_output(agent_dict_output, self.llm_model)
        # parsed_str = f"{parsed.summary}\n{parsed.details}"

        # return Command(
        #     update={
        #         "messages": state.messages[-1:] + [AIMessage(content=parsed_str, name="Validation_node")],
        #         "validation": [agent_dict_output]
        #     },
        #     goto="supervisor",
        # )
    
    def Reporter_node(self, state: AgentStateModel) -> Command[Literal['supervisor']]:
        print("*****************called Reporter node************")

        Instructions = state.Instructions

        df_path = state.Path

        # 1. Instantiate the parser for our new structured output
        parser = PydanticOutputParser(pydantic_object=BusinessReport)

        fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_model)

        task_prompt = (
        f"Find the instructions given by the user here : {Instructions} and follow this {Reporter_prompt} to the letter."
    )
        print(f"--- Sending this direct task to the agent ---\n{task_prompt}\n---------------------------------------------")

        system_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a Business Intelligence consultant. You have access to `eda_fact_sheet(df_path)` for initial recon and `python_repl_ast(query)` for deep-dive analysis. "
             "The CSV is at this path: {df_path}. "
             "Your mission is to autonomously analyze the data and produce a strategic business report. "
             f"Follow these instructions precisely: {Reporter_prompt}"),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        Reporter_agent = create_tool_calling_agent(
            llm=self.llm_model,
            tools=[eda_fact_sheet, python_repl_ast],
            prompt=system_prompt
        )

        agent_executor = AgentExecutor(
            agent=Reporter_agent,
            tools=[eda_fact_sheet, python_repl_ast],
            verbose=True,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )

        # 2. Wrap execution in a retry loop for robust parsing
        max_attempts = 3
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            try:
                result = agent_executor.invoke({
                    "input": task_prompt,
                    "df_path": df_path,
                    "chat_history": []
                })

                # Try parsing the final output using the fixing parser
                final_output_string = result.get("output", "")
                parsed_output: BusinessReport = fixing_parser.parse(final_output_string)

                # Successfully parsed → create a summary and update the state
                summary_str = f"{parsed_output.subject}\n\n{parsed_output.executive_summary}"

                return Command(
                    update={
                        "messages": state.messages[-1:] + [AIMessage(content=summary_str, name="Reporter_node")],
                        "Report": [{"final_answer": parsed_output.model_dump()}] # Store the full structured report
                    },
                    goto="supervisor",
                )

            except Exception as e:
                error_msg = (
                    f"Attempt {attempt} failed due to a parsing error: {str(e)}. "
                    f"You MUST provide a final answer that is a valid JSON object. Please strictly follow this schema: "
                    f"{BusinessReport.model_json_schema()}"
                )
                print(f"--- Runtime/Parsing error encountered ---\n{error_msg}")
                # Inject the error and schema into the prompt for the next retry
                task_prompt = task_prompt + "\n\n" + error_msg

        # If all attempts fail, fallback to the supervisor with an error message
        return Command(
            update={
                "messages": state.messages[-1:] + [
                    AIMessage(content="Error: The reporter agent failed to produce a valid business report after multiple attempts.",
                              name="Reporter_node_Error")
                ],
                "Report": [{"error": "Parsing failed after all retries"}]
            },
            goto="supervisor",
        )


        # Analyzer_agent = create_tool_calling_agent(
        #     llm=self.llm_model,
        #     tools=[eda_fact_sheet,python_repl_ast],
        #     prompt=system_prompt
        # )

        # tools = [eda_fact_sheet,python_repl_ast]

        # agent_executor = AgentExecutor(
        #     agent=Analyzer_agent,
        #     tools=tools,
        #     verbose=True,
        #     handle_parsing_errors=True,
        #     return_intermediate_steps=True
        # )

        # result = agent_executor.invoke({
        #         "input": task_prompt,
        #         "df_path": df_path,
        #         "chat_history": []     
        #     })
        
        # agent_dict_output = {"final_answer": result.get("output")}

        # parsed = summarize_tool_output(agent_dict_output,self.llm_model)

        # parsed_str = f"{parsed.summary}\n{parsed.details}"

        # return Command(
        #     update={
        #         "messages": state.messages[-1:] + [AIMessage(content=parsed_str, name="Reporter_node")],
        #         "Report": [agent_dict_output] 
        #     },
        #     goto="supervisor",
        # )
    
#Data is about sales,provide the data overview along with the preprocessing steps needed to perform EDA , here's the path path = r"D:\Code Assistant\retail_sales_dataset.csv"
#dbuser1 user and pass for cluster
