import os
import json
import sqlite3
import datetime
from typing import TypedDict, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END

# Import schemas and sub-agents
from schemas import SafetyCasePayload, DuplicateResult, QCResult, RiskAssessment
from pipeline import (
    intake_agent_extract_text,
    extraction_and_coding_agent,
    duplicate_check_agent,
    qc_agent_validate,
    policy_and_risk_agent,
    commit_to_audit_trail
)

# -------------------------------------------------------------------------
# 1. State Definition (Shared Memory Across Agents)
# -------------------------------------------------------------------------
class PharmaAgentState(TypedDict):
    case_id: str
    source_filename: str
    raw_pdf_bytes: Optional[bytes]
    raw_text: str
    extracted_payload: Optional[SafetyCasePayload]
    duplicate_result: Optional[DuplicateResult]
    qc_result: Optional[QCResult]
    risk_assessment: Optional[RiskAssessment]
    workflow_history: List[Dict[str, Any]]
    final_decision: str
    error_message: Optional[str]
    next_step: str

# -------------------------------------------------------------------------
# 2. Agent Node Wrappers
# -------------------------------------------------------------------------
def intake_node(state: PharmaAgentState) -> Dict[str, Any]:
    """Extracts raw text and establishes case ID."""
    history = list(state.get("workflow_history", []))
    try:
        # Fallback to existing text if running without fresh bytes
        raw_text = state.get("raw_text", "")
        history.append({
            "agent": "Intake Agent",
            "status": "Success",
            "timestamp": datetime.datetime.utcnow().isoformat()
        })
        return {
            "raw_text": raw_text,
            "workflow_history": history,
            "next_step": "case_extraction"
        }
    except Exception as exc:
        history.append({"agent": "Intake Agent", "status": f"Failed: {str(exc)}"})
        return {"error_message": str(exc), "workflow_history": history, "next_step": "error"}

def case_and_coding_node(state: PharmaAgentState) -> Dict[str, Any]:
    """Executes entity extraction and MedDRA coding via Groq."""
    history = list(state["workflow_history"])
    try:
        payload = extraction_and_coding_agent(state["raw_text"])
        history.append({
            "agent": "Safety Case & Coding Agent",
            "status": "Success",
            "adverse_events_count": len(payload.coded_adverse_events),
            "timestamp": datetime.datetime.utcnow().isoformat()
        })
        return {
            "extracted_payload": payload,
            "workflow_history": history,
            "next_step": "duplicate_check"
        }
    except Exception as exc:
        history.append({"agent": "Safety Case & Coding Agent", "status": f"Failed: {str(exc)}"})
        return {"error_message": str(exc), "workflow_history": history, "next_step": "error"}

def duplicate_check_node(state: PharmaAgentState) -> Dict[str, Any]:
    """Queries vector database for historical similarity."""
    history = list(state["workflow_history"])
    payload = state["extracted_payload"]
    try:
        query_str = f"{payload.suspect_product} {payload.seriousness_criteria}"
        dup_result = duplicate_check_agent(query_str)
        history.append({
            "agent": "Duplicate Detection Agent",
            "status": "Checked",
            "similarity_score": dup_result.similarity_score,
            "timestamp": datetime.datetime.utcnow().isoformat()
        })
        return {
            "duplicate_result": dup_result,
            "workflow_history": history,
            "next_step": "qc_validation"
        }
    except Exception as exc:
        history.append({"agent": "Duplicate Detection Agent", "status": f"Failed: {str(exc)}"})
        return {"error_message": str(exc), "workflow_history": history, "next_step": "error"}

def qc_validation_node(state: PharmaAgentState) -> Dict[str, Any]:
    """Performs deterministic schema and SOP rule validation."""
    history = list(state["workflow_history"])
    try:
        qc_result = qc_agent_validate(state["extracted_payload"])
        history.append({
            "agent": "QC Agent",
            "status": "Passed" if qc_result.passed else "Failed",
            "flag_count": len(qc_result.flags),
            "timestamp": datetime.datetime.utcnow().isoformat()
        })
        return {
            "qc_result": qc_result,
            "workflow_history": history,
            "next_step": "policy_risk_assessment"
        }
    except Exception as exc:
        history.append({"agent": "QC Agent", "status": f"Failed: {str(exc)}"})
        return {"error_message": str(exc), "workflow_history": history, "next_step": "error"}

def policy_risk_node(state: PharmaAgentState) -> Dict[str, Any]:
    """Evaluates regulatory risk factors to determine human review routing."""
    history = list(state["workflow_history"])
    try:
        risk = policy_and_risk_agent(
            state["extracted_payload"],
            state["qc_result"],
            state["duplicate_result"]
        )
        history.append({
            "agent": "Policy & Risk Engine",
            "risk_level": risk.risk_level,
            "requires_human": risk.requires_human_review,
            "timestamp": datetime.datetime.utcnow().isoformat()
        })
        return {
            "risk_assessment": risk,
            "workflow_history": history
        }
    except Exception as exc:
        history.append({"agent": "Policy & Risk Engine", "status": f"Failed: {str(exc)}"})
        return {"error_message": str(exc), "workflow_history": history, "next_step": "error"}

def human_review_queue_node(state: PharmaAgentState) -> Dict[str, Any]:
    """Routes high-risk or flagged cases into the manual review queue."""
    history = list(state["workflow_history"])
    history.append({
        "agent": "Workflow Router",
        "action": "Escalated to Human Review",
        "timestamp": datetime.datetime.utcnow().isoformat()
    })
    return {
        "final_decision": "PENDING_HUMAN_REVIEW",
        "workflow_history": history
    }

def auto_approve_node(state: PharmaAgentState) -> Dict[str, Any]:
    """Auto-approves low-risk cases that cleared all validation barriers."""
    history = list(state["workflow_history"])
    history.append({
        "agent": "Workflow Router",
        "action": "Fast-Track Auto Approved",
        "timestamp": datetime.datetime.utcnow().isoformat()
    })
    return {
        "final_decision": "AUTO_APPROVED",
        "workflow_history": history
    }

def audit_sink_node(state: PharmaAgentState) -> Dict[str, Any]:
    """Writes immutable execution ledger to the SQLite audit trail."""
    commit_to_audit_trail(
        case_id=state["case_id"],
        filename=state["source_filename"],
        payload=state["extracted_payload"],
        history=state["workflow_history"],
        decision=state["final_decision"],
        reviewer="System_Orchestrator" if state["final_decision"] == "AUTO_APPROVED" else "Pending_Reviewer"
    )
    return state

# -------------------------------------------------------------------------
# 3. Conditional Routing Logic
# -------------------------------------------------------------------------
def risk_router(state: PharmaAgentState) -> str:
    """Evaluates whether to short-circuit to human review or auto-approve."""
    if state.get("error_message"):
        return "route_human"
    
    risk = state.get("risk_assessment")
    if risk and not risk.requires_human_review:
        return "route_auto"
    return "route_human"

# -------------------------------------------------------------------------
# 4. Graph Construction & Compilation
# -------------------------------------------------------------------------
def build_pharma_orchestrator() -> StateGraph:
    workflow = StateGraph(PharmaAgentState)

    # Register Nodes
    workflow.add_node("intake", intake_node)
    workflow.add_node("extraction_and_coding", case_and_coding_node)
    workflow.add_node("duplicate_check", duplicate_check_node)
    workflow.add_node("qc_validation", qc_validation_node)
    workflow.add_node("policy_risk", policy_risk_node)
    workflow.add_node("human_review_queue", human_review_queue_node)
    workflow.add_node("auto_approve", auto_approve_node)
    workflow.add_node("audit_sink", audit_sink_node)

    # Define Linear Backbone
    workflow.set_entry_point("intake")
    workflow.add_edge("intake", "extraction_and_coding")
    workflow.add_edge("extraction_and_coding", "duplicate_check")
    workflow.add_edge("duplicate_check", "qc_validation")
    workflow.add_edge("qc_validation", "policy_risk")

    # Add Dynamic Conditional Branching
    workflow.add_conditional_edges(
        "policy_risk",
        risk_router,
        {
            "route_auto": "auto_approve",
            "route_human": "human_review_queue"
        }
    )

    # Join paths back to Audit Sink
    workflow.add_edge("auto_approve", "audit_sink")
    workflow.add_edge("human_review_queue", "audit_sink")
    workflow.add_edge("audit_sink", END)

    return workflow.compile()

# Singleton compiled application instance
orchestrator_app = build_pharma_orchestrator()