from langgraph.graph import StateGraph, START, END
from schemas import AgentState
from agents import intake_agent, safety_case_agent, coding_agent, duplicate_agent, qc_agent
from database import save_to_evidence_graph, setup_database

def build_orchestrator():
    setup_database()
    
    # 1. Initialize the Orchestrator
    workflow = StateGraph(AgentState)

    # 2. Add Specialized Agents[cite: 1]
    workflow.add_node("Intake", intake_agent)
    workflow.add_node("Case_Extraction", safety_case_agent)
    workflow.add_node("Coding", coding_agent)
    workflow.add_node("Duplicate_Check", duplicate_agent)
    workflow.add_node("QC_Validation", qc_agent)

    # 3. Define the Execution Pattern (Workflow)[cite: 1]
    workflow.add_edge(START, "Intake")
    workflow.add_edge("Intake", "Case_Extraction")
    workflow.add_edge("Case_Extraction", "Coding")
    workflow.add_edge("Coding", "Duplicate_Check")
    workflow.add_edge("Duplicate_Check", "QC_Validation")
    workflow.add_edge("QC_Validation", END)

    return workflow.compile()

def process_document(text_input):
    app = build_orchestrator()
    final_state = app.invoke({
        "raw_text": text_input,
        "extracted_data": {},
        "status": "Processing",
        "audit_trail": []
    })
    
    # Save the final output to the Evidence Graph[cite: 1]
    save_to_evidence_graph(
        final_state["raw_text"], 
        final_state["extracted_data"], 
        "Awaiting Human Approval"
    )
    
    return final_state