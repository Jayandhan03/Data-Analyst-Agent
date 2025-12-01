import time
from langgraph.types import Command
from langgraph.graph import END
from llm import llm_model
from pydantic import BaseModel,Field
from typing import List,Literal, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from Prompts import supervisor_prompt,PreprocessingPlanner_prompt,cleaner_prompt
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from Toolkit.Tools import python_cleaning_tool,eda_fact_sheet
from Guardrails.Preprocessing import StructuredPlanOutput
from Guardrails.cleaner import CleaningSummary  
from langchain.output_parsers import PydanticOutputParser,OutputFixingParser


class Router(BaseModel):
    next: Literal["PreprocessingPlanner_node","Cleaner_node",END]= Field(description="The next node to route to. Must be one of the available nodes.")
    reasoning: str = Field(description="A short reasoning for the decision made.")

class AgentStateModel(BaseModel):
    messages: Optional[List] = None
    Instructions: Optional[str] = None
    Analysis: Optional[List[dict]] = None
    clean: Optional[List[dict]] = None
    batched_plan: Optional[List[List[dict]]] = None 
    Path: Optional[str] = None
    next: Optional[str] = None
    current_reasoning: Optional[str] = None   
    
class DataAnalystAgent:
    def __init__(self):
        self.llm_model = llm_model

    def supervisor_node(self,state:AgentStateModel) -> Command[Literal["PreprocessingPlanner_node","Cleaner_node", END]]:

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
        f"- If Cleaning Plan Generated: {'Yes' if state.clean else 'No'}\n"

    )

        messages_for_llm = [
            SystemMessage(content=supervisor_prompt),
            HumanMessage(content=state_summary),
        ]

        if state.messages:
            last_message = state.messages[-1]
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
        print("***************** called cleaner node ************")
        Path = state.Path
        cleaning_plan = state.Analysis[0]['final_answer']['plan']

        # Batch the determined plan
        batched_plan = [cleaning_plan[i:i + 4] for i in range(0, len(cleaning_plan), 4)]

        # --- Setup agent, parser, and prompt ---
        parser = PydanticOutputParser(pydantic_object=CleaningSummary)
        fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=self.llm_model)
        system_prompt = ChatPromptTemplate.from_messages([
            ("system", "Follow the instructions here : {cleaner_prompt} and in the input to the letter and make the necessary changes to the dataframe."),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        Cleaner_agent = create_tool_calling_agent(llm=self.llm_model, tools=[python_cleaning_tool], prompt=system_prompt)
        agent_executor = AgentExecutor(
            agent=Cleaner_agent,
            tools=[python_cleaning_tool],
            verbose=True,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )

        all_batch_results = []
        final_clean_outputs = []

        # --- Iterate through each batch with retry logic ---
        for i, batch in enumerate(batched_plan, start=1):
            print(f"--- Starting processing for Batch {i} of {len(batched_plan)} ---")
            
            # Initial task prompt for the batch
            task_prompt = (f"Apply the following cleaning plan (batch {i} of {len(batched_plan)}) to the dataset at path: {Path}\nPlan details:\n{str(batch)}")

            max_attempts = 3
            attempt = 0
            batch_successful = False

            while attempt < max_attempts:
                attempt += 1
                print(f"--- Batch {i}, Attempt {attempt} ---")
                
                try:
                    # 1. Invoke the agent
                    result = agent_executor.invoke({
                        "input": task_prompt,
                        "cleaner_prompt": cleaner_prompt,
                    })

                    # 2. Try parsing the final output
                    final_output_string = result.get("output", "")
                    parsed_output: CleaningSummary = fixing_parser.parse(final_output_string)

                    # 3. If successful, store results and break the retry loop
                    all_batch_results.append(parsed_output)
                    final_clean_outputs.append({"final_answer": final_output_string})
                    print(f"--- Batch {i}, Attempt {attempt} successful ---")
                    batch_successful = True
                    time.sleep(5)
                    break # Exit the while loop for this batch

                except Exception as e:
                    # 4. On failure, create an error message for the next attempt
                    error_msg = (
                        f"Attempt {attempt} for batch {i} failed due to error: {str(e)}. "
                        f"Please analyze the error and the plan, then try again. Ensure your final output strictly follows this schema: {CleaningSummary.model_json_schema()}"
                    )
                    print(f"--- Runtime/Parsing error encountered ---\n{error_msg}")
                    
                    # Prepend the error to the prompt for the next retry
                    task_prompt = f"The previous attempt failed with this error: {error_msg}. Please correct your tool usage and try again. Here is the original task for this batch:\n---\n{task_prompt}"

            # 5. If all attempts for this batch fail, exit and report to supervisor
            if not batch_successful:
                error_message = f"Error: Cleaner agent failed on batch {i} after {max_attempts} attempts. Aborting cleaning process."
                return Command(
                    update={
                        "messages": [AIMessage(content=error_message, name="Cleaner_node_Error")],
                        "clean": [{"error": f"Failed on batch {i} after retries"}]
                    },
                    goto="supervisor"
                )

        # --- If all batches succeed, return the final successful result ---
        final_summary = "All cleaning batches completed successfully."
        update_dict = {
            "messages": [AIMessage(content=final_summary, name="cleaner_node")],
            "clean": final_clean_outputs,
            "batched_plan": batched_plan
        }

        return Command(update=update_dict, goto="supervisor")
    