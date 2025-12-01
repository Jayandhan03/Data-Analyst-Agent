from langgraph.graph import START, StateGraph, END
from Cleaner_Agent import DataAnalystAgent, AgentStateModel

agent = DataAnalystAgent()

def main():

    # 2️⃣ Gather individual inputs first
    instructions_input = input("Any instructions about the data: ")
    path_input = input("Path to the data: ")     

    # 1️⃣ Gather input from the userrs
    user_input = {
    "Instructions": instructions_input,
    "Path": path_input,
    "messages": [],
    "Analysis": [],
    "next": "",
    "current_reasoning": ""
}

    initial_state = AgentStateModel(**user_input)

    # 2️⃣ Build the workflow graph
    graph = StateGraph(AgentStateModel)

    # Add actual implemented nodes
    graph.add_node("supervisor", agent.supervisor_node)
    graph.add_node("PreprocessingPlanner_node", agent.PreprocessingPlanner_node)
    graph.add_node("Cleaner_node", agent.Cleaner_node)

    graph.add_edge(START, "supervisor")
    
    # Compile workflow
    app = graph.compile()
    # 3️⃣ Run the workflow from START with the initialized state
    final_state = app.invoke(initial_state)

    # 4️⃣ Print final state
    print("===================================")
    print("Final workflow state:")
    print(final_state)

if __name__ == "__main__":
    main()
