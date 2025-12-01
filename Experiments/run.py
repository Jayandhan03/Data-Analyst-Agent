import time
from langgraph.types import Command
from langgraph.graph import END
from llm import llm_model
from pydantic import BaseModel,Field
from typing import List,Literal, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from Prompts import supervisor_prompt,PreprocessingPlanner_prompt,cleaner_prompt,Reporter_prompt,VISUALIZATION_PROMPT
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from Toolkit.Tools import python_repl_ast,eda_fact_sheet
from Guardrails.Preprocessing import StructuredPlanOutput
from Guardrails.cleaner import CleaningSummary  
from Guardrails.report import BusinessReport
from Guardrails.visualizer import VisualizationReport
from langchain.output_parsers import PydanticOutputParser,OutputFixingParser


class Router(BaseModel):
    next: Literal["PreprocessingPlanner_node","Cleaner_node","Reporter_node","visualizer_node",END]= Field(description="The next node to route to. Must be one of the available nodes.")
    reasoning: str = Field(description="A short reasoning for the decision made.")

class AgentStateModel(BaseModel):
    messages: Optional[List] = None
    Instructions: Optional[str] = None
    Analysis: Optional[List[dict]] = None
    clean: Optional[List[dict]] = None
    Report: Optional[List[dict]] = None
    Visualizations: Optional[List[dict]] = None
    Path: Optional[str] = None
    next: Optional[str] = None
    current_reasoning: Optional[str] = None   
class DataAnalystAgent:
    def __init__(self):
        self.llm_model = llm_model

    def supervisor_node(self,state:AgentStateModel) -> Command[Literal["PreprocessingPlanner_node","Cleaner_node","Reporter_node","visualizer_node", END]]:

        """
        The central router of the workflow.
        It evaluates the current state and the last message to decide the next action.
        This node is designed to be highly token-efficient by creating a lean summary of the state
        instead of passing the full, verbose state objects to the LLM.
        """

        print("**************************below is my state right after entering****************************")

        print(state)

        print("************************** SUPERVISOR: EVALUATING STATE ****************************")

        state_summary = (
        f"Current Workflow Status:\n"
        f"- Analysis Plan Generated: {'Yes' if state.Analysis else 'No'}\n"
        f"- Cleaning Plan Generated: {'Yes' if state.clean else 'No'}\n"
        f"- Report Generated: {'Yes' if state.Report else 'No'}\n"
        f"- Visualizations Generated: {'Yes' if state.Visualizations else 'No'}\n"
    )

        messages_for_llm = [
            SystemMessage(content=supervisor_prompt),
            HumanMessage(content=state_summary),
        ]

        if state.messages:
            last_message = state.messages[-1]
            # Add a prefix to clearly label the last message for the LLM
            last_message_content = f"Last Event:\nThe last node to run was '{last_message.name}'. It reported the following:\n---\n{last_message.content}\n---"
            messages_for_llm.append(HumanMessage(content=last_message_content))
            print(f"--- Attaching last event from '{last_message.name}' ---")
        else:
            # Handle the very first run where there are no messages
            messages_for_llm.append(HumanMessage(content="Last Event: None. This is the first step of the workflow."))

        messages_for_this_attempt = list(messages_for_llm)

        print("***********************Invoking LLM for routing decision************************")

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
                messages_for_this_attempt.append(HumanMessage(content=f"Previous attempt failed due to: {error_msg}. Please follow the schema strictly: {Router.model_json_schema()}"))

            try:
                response = chain.invoke(messages_for_this_attempt)
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

        parser = PydanticOutputParser(pydantic_object=StructuredPlanOutput)

        fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_model)

        task_prompt = (
        f"Find the instructions given by the user here : {Instructions} and follow this {PreprocessingPlanner_prompt} to the letter.modify in this path:{state.Path}"
    )
        print(f"--- Sending this direct task to the agent ---\n{task_prompt}\n---------------------------------------------")

        system_prompt = ChatPromptTemplate.from_messages([
        
        ("system",
         "You are a DataFrame analyzer. Your primary tool is `eda_fact_sheet`. "
         "First, call the tool to get data insights. Then, based on the tool's output, "
         "provide a final answer formatted as a JSON object containing the preprocessing plan and summaries."),
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
                        "messages": [
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
                # Use an f-string to properly embed all variables
                task_prompt = f"The previous attempt failed with this error: {error_msg}. Please correct your tool usage and try again. Here is the original task:\n---\n{task_prompt}"

        # If all attempts fail, fallback to supervisor with error message
        return Command(
            update={
                "messages": [
                    AIMessage(content="Error: The analysis agent failed to produce a valid preprocessing plan after multiple attempts.", 
                            name="Analyzer_node_Error")
                ],
                "Analysis": [{"error": "Parsing failed after retries"}]
            },
            goto="supervisor",
        )
                               
    def Cleaner_node(self, state: AgentStateModel) -> Command[Literal['supervisor']]:

        print("*****************called cleaner node************")

        Path = state.Path

        preprocessing_plan = state.Analysis[0]['final_answer']['plan']

        batched_plan = []
        current_batch = []

        for column_action in preprocessing_plan:
            # Add the current item to the batch
            current_batch.append(column_action)
            
            # If the batch is now full, add it to our final list and reset it
            if len(current_batch) == 4:
                batched_plan.append(current_batch)
                current_batch = []

            # After the loop, check if there are any leftover items in the last batch
        if current_batch:
          batched_plan.append(current_batch)
    
        parser = PydanticOutputParser(pydantic_object=CleaningSummary)

        fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_model)

        system_prompt = ChatPromptTemplate.from_messages(
        [(
            "system",
            "Follow the instructions here : {cleaner_prompt} and  in the input to the letter and make the necessary changes to the dataframe."
        ),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")        
        ]
        )
        
        Cleaner_agent = create_tool_calling_agent(
            llm=self.llm_model,
            tools=[python_repl_ast],
            prompt=system_prompt
        )

        agent_executor = AgentExecutor(
            agent=Cleaner_agent,
            tools=[python_repl_ast],
            verbose=True,
            # handle_parsing_errors=True,
            # return_intermediate_steps=True
        )

        # 3. Loop Through Batches and Invoke the Agent for Each
        all_batch_results = []
        final_clean_outputs = []

        for i, batch in enumerate(batched_plan, start=1):
            clean_plan_str = str(batch)
            task_prompt = (
                f"Apply the following cleaning plan (batch {i} of {len(batched_plan)}) to the dataset at path: {Path}\n"
                f"Plan details:\n{clean_plan_str}"
            )
            
            print(f"--- Sending task for Batch {i} to the agent ---\n{task_prompt}\n---------------------------------------------")

            max_attempts = 3
            attempt = 0
            batch_success = False
            while attempt < max_attempts:
                attempt += 1
                try:
                    result = agent_executor.invoke({"input": task_prompt,"cleaner_prompt":cleaner_prompt,"agent_scratchpad": [] })

                    final_output_string = result.get("output", "")
                    
                    # Try parsing the output
                    parsed_output: CleaningSummary = fixing_parser.parse(final_output_string)

                    # On success, store results and break the retry loop
                    all_batch_results.append(parsed_output)
                    final_clean_outputs.append({"final_answer": final_output_string})
                    batch_success = True
                    print(f"--- Batch {i} successful on attempt {attempt} ---")
                    time.sleep(10)
                    break

                except Exception as e:
                    error_msg = (
                        f"Attempt {attempt} for batch {i} failed with an error: {str(e)}. "
                        "You MUST provide a final answer that is a valid JSON object. Please review the plan and strictly follow the schema."
                    )
                    print(f"--- Runtime/Parsing error for Batch {i} ---\n{error_msg}")
                    # Inject error context for the next retry attempt on this specific batch
                    task_prompt = f"Your previous attempt failed with this error: {error_msg}\n\nPlease re-execute the original plan:\n{task_prompt}"

            # If a batch fails after all retries, exit the entire node with an error
            if not batch_success:
                error_message = f"Error: The cleaner agent failed to process batch {i} after {max_attempts} attempts."
                return Command(
                    update={
                        "messages": [AIMessage(content=error_message, name="Cleaner_node_Error")],
                        "clean": [{"error": f"Processing failed at batch {i}"}]
                    },
                    goto="supervisor",
                )

        final_summary = "All cleaning batches completed successfully.\n\n"
        for idx, summary in enumerate(all_batch_results, start=1):
            final_summary += f"--- Batch {idx} Summary ---\nSummary: {summary.summary}\nDetails: {summary.details}\n\n"

        return Command(
            update={
                "messages": [AIMessage(content=final_summary.strip(), name="cleaner_node")],
                "clean": final_clean_outputs,
            },
            goto="supervisor",
        )

    def Reporter_node(self, state: AgentStateModel) -> Command[Literal['supervisor']]:
        print("*****************called Reporter node************")

        Instructions = state.Instructions

        df_path = state.Path

        # --- STEP 1: Perform Reconnaissance MANUALLY (and only once) ---
        print("--- Reporter: Performing initial data reconnaissance with eda_fact_sheet ---")
        try:
            recon_result_str = str(eda_fact_sheet.run(path=df_path))
        except Exception as e:
            return Command(update={"messages": [AIMessage(content=f"Error during initial data recon: {e}", name="Reporter_node_Error")]}, goto="supervisor")
        
        print(f"--- Reporter: Condensing {len(recon_result_str)} characters of context... ---")
        condensation_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a data analysis assistant. Summarize the following verbose JSON data profile into a concise, human-readable format for another AI agent to use. Focus on column names, data types, and key stats."),
            ("human", "Please summarize this data profile:\n\n{profile}")
        ])
        summarizer_chain = condensation_prompt | self.llm_model

        condensed_summary = summarizer_chain.invoke({"profile": recon_result_str}).content

        print(f"--- Reporter: Condensed Summary Created ---")

        # 1. Instantiate the parser for our new structured output
        parser = PydanticOutputParser(pydantic_object=BusinessReport)

        fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_model)

        task_prompt = (f"User Instructions: {Instructions}\n\nHere is a condensed summary of the dataset: \n---\n{condensed_summary}\n---\n\nNow, follow your main instructions: {Reporter_prompt}")

        print(f"--- Reporter: Sending condensed task to the main analysis agent ---")

        print(f"--- Sending this direct task to the agent ---\n{task_prompt}\n---------------------------------------------")

        system_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a Business Intelligence consultant. You have access to `eda_fact_sheet(df_path)` for initial recon and `python_repl_ast(query)` for deep-dive analysis. "
             "The CSV is at this path: {df_path}. "
             "Your mission is to autonomously analyze the data and produce a strategic business report. "),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        Reporter_agent = create_tool_calling_agent(
            llm=self.llm_model,
            tools=[python_repl_ast],
            prompt=system_prompt
        )

        agent_executor = AgentExecutor(
            agent=Reporter_agent,
            tools=[python_repl_ast],
            verbose=True,
            # handle_parsing_errors=True,
            # return_intermediate_steps=True
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
                })

                final_output_string = result.get("output", "")

                parsed_output: BusinessReport = fixing_parser.parse(final_output_string)

                summary_str = f"{parsed_output.subject}\n\n{parsed_output.executive_summary}"

                return Command(
                    update={
                        "messages": [AIMessage(content=summary_str, name="Reporter_node")],
                        "Report": [{"final_answer": parsed_output.model_dump()}] 
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
                
                task_prompt =f"here is the error from the previous execution: {error_msg} so process the workflow accordingly" + task_prompt 

        
        return Command(
            update={
                "messages": [
                    AIMessage(content="Error: The reporter agent failed to produce a valid business report after multiple attempts.",
                              name="Reporter_node_Error")
                ],
                "Report": [{"error": "Parsing failed after all retries"}]
            },
            goto="supervisor",
        )


    def visualizer_node(self, state: AgentStateModel) -> Command[Literal['supervisor']]:
        """
        This node directs an agent to perform EDA and generate 10 business-focused visualizations.
        It enforces a structured JSON output for the final report and includes retry logic for parsing.
        """
        print("***************** called Visualizer node ************")

        df_path = state.Path

        print("--- Visualizer: Performing initial data reconnaissance... ---")
        try:
            recon_result_str = str(eda_fact_sheet.run(path=df_path))
        except Exception as e:
            return Command(update={"messages": [AIMessage(content=f"Error during initial data recon: {e}", name="Visualizer_node_Error")]}, goto="supervisor")

        print(f"--- Visualizer: Condensing {len(recon_result_str)} characters of context... ---")
        condensation_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a data analysis assistant. Summarize the following verbose JSON data profile into a concise, human-readable format for another AI agent to use for creating visualizations."),
            ("human", "Please summarize this data profile:\n\n{profile}")
        ])
        summarizer_chain = condensation_prompt | self.llm_model

        condensed_summary = summarizer_chain.invoke({"profile": recon_result_str}).content

        print(f"--- Visualizer: Condensed Summary Created ---")

        parser = PydanticOutputParser(pydantic_object=VisualizationReport)

        fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_model)

        task_prompt = (f"Based on this data summary:\n---\n{condensed_summary}\n---\n\nNow, follow your main instructions to create visualizations: {VISUALIZATION_PROMPT}")
        print(f"--- Visualizer: Sending condensed task to the plotting agent ---")

        print(f"--- Sending this direct task to the agent ---\n{task_prompt}\n---------------------------------------------")

        system_prompt = ChatPromptTemplate.from_messages([
            ("system",
            "You are a Data Visualization Specialist. You have access to `eda_fact_sheet(df_path)` for initial recon and `python_repl_ast(query)` for generating and saving plots. "
            "The CSV is at this path: {df_path}. "
            "Your mission is to autonomously analyze the data and produce a series of 10 visualizations as instructed. "),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        visualizer_agent = create_tool_calling_agent(
            llm=self.llm_model,
            tools=[python_repl_ast],
            prompt=system_prompt
        )

        agent_executor = AgentExecutor(
            agent=visualizer_agent,
            tools=[python_repl_ast],
            verbose=True,
            # handle_parsing_errors=True,
            # return_intermediate_steps=True
        )

        max_attempts = 3
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            try:
                result = agent_executor.invoke({
                    "input": task_prompt,
                    "df_path": df_path,
                })

                final_output_string = result.get("output", "")

                parsed_output: VisualizationReport = fixing_parser.parse(final_output_string)

                summary_str = f"Successfully generated a report with {len(parsed_output.visualizations)} visualizations."

                return Command(
                    update={
                        "messages": [AIMessage(content=summary_str, name="Visualizer_node")],

                        "Visualizations": [{"final_answer": parsed_output.model_dump()}]
                    },
                    goto="supervisor",
                )

            except Exception as e:
                error_msg = (
                    f"Attempt {attempt} failed due to a parsing error: {str(e)}. "
                    f"You MUST provide a final answer that is a valid JSON object. Please strictly follow this schema: "
                    f"{VisualizationReport.model_json_schema()}"
                )
                print(f"--- Runtime/Parsing error encountered ---\n{error_msg}")
          
                task_prompt = f"here is the error from the last execution: {error_msg} so process the workflow accordingly" + "\n\n" + task_prompt

        return Command(
            update={
                "messages":  [
                    AIMessage(content="Error: The visualizer agent failed to produce a valid report after multiple attempts.",
                            name="Visualizer_node_Error")
                ],
                "Visualizations": [{"error": "Parsing failed after all retries"}]
            },
            goto="supervisor",
        )

"Data is about sales,provide the data overview along with the preprocessing steps needed to perform EDA" 

"""
Path = r"D:\Code Assistant\superstore sales.csv"
D:\Code Assistant\superstore sales.csv
"""