import streamlit as st
from pipeline import (
    init_evidence_db,
    intake_agent_extract_text,
    commit_to_audit_trail,
    send_whatsapp_alert,  
    document_qa_agent     
)
from orchestrator import orchestrator_app # <-- NEW: Imports the LangGraph State Machine

# Initialize Database
init_evidence_db()

st.set_page_config(page_title="PharmaAgentOS Human Review", layout="wide")
st.title("💊 PharmaAgentOS: Safety Case Processing")
st.caption("AI Workforce Execution → Human Governance & Approval")

# Step 1: PDF Upload (UPDATED FOR MULTIPLE FILES)
uploaded_files = st.file_uploader(
    "Upload Safety Report(s) (PDF)", 
    type=["pdf"], 
    accept_multiple_files=True  
)

sample_default_text = """PATIENT ADVERSE EVENT REPORT
Date: 2026-03-12
Patient: Female, 58 years old
Suspect Drug: Metformin 500mg (Lot #99482)
Indication: Type 2 Diabetes
Reaction: Developed acute severe hives (urticaria) and facial swelling 30 minutes after second dose.
Outcome: Admitted to emergency care. Resolved with antihistamines.
Reporter: Dr. Robert Vance, MD"""

# Step 2: Combine text from ALL uploaded files
if uploaded_files:
    raw_text = ""
    filenames = []
    
    for file in uploaded_files:
        raw_text += f"--- Document: {file.name} ---\n"
        raw_text += intake_agent_extract_text(file) + "\n\n"
        filenames.append(file.name)
        
    filename = ", ".join(filenames) 
else:
    raw_text = sample_default_text
    filename = "Hospital_Report_Case_892.pdf"

# --- NEW: LANGGRAPH ORCHESTRATION EXECUTION ---
if st.button("🚀 Run Multi-Agent Workforce"):
    
    # 1. Define the initial State Graph memory
    initial_state = {
        "case_id": "PV-2026-0892",
        "source_filename": filename,
        "raw_pdf_bytes": None,
        "raw_text": raw_text,
        "extracted_payload": None,
        "duplicate_result": None,
        "qc_result": None,
        "risk_assessment": None,
        "workflow_history": [],
        "final_decision": "INITIALIZED",
        "error_message": None,
        "next_step": "start"
    }

    # 2. Invoke the compiled LangGraph Agent Orchestrator
    with st.spinner("LangGraph Orchestrator coordinating AI agents..."):
        final_state = orchestrator_app.invoke(initial_state)

    # 3. Handle Output State
    if final_state.get("error_message"):
        st.error(f"Pipeline failed: {final_state['error_message']}")
    else:
        st.session_state["pipeline_output"] = {
            "case_id": final_state["case_id"],
            "filename": final_state["source_filename"],
            "raw_text": final_state["raw_text"],
            "data": final_state["extracted_payload"],
            "duplicate": final_state["duplicate_result"],
            "qc": final_state["qc_result"],
            "risk": final_state["risk_assessment"],
            "history": final_state["workflow_history"],
            "final_decision": final_state["final_decision"]
        }

# Step 7: Side-by-Side Review
if "pipeline_output" in st.session_state:
    res = st.session_state["pipeline_output"]
    st.divider()
    
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📄 Source Document Evidence")
        st.text_area("Extracted Raw Text", res["raw_text"], height=320, disabled=True)
        st.write(f"**Duplicate Status:** Similarity Score: `{res['duplicate'].similarity_score}` (Is Duplicate: `{res['duplicate'].is_duplicate}`)")
        st.info(f"Duplicate Reason: {res['duplicate'].match_reason}")

    with col2:
        st.subheader("🤖 AI Extracted & Coded Data")
        st.write(f"**Patient:** {res['data'].patient_gender}, {res['data'].patient_age} yrs")
        st.write(f"**Suspect Product:** {res['data'].suspect_product} ({res['data'].dosage})")
        st.write(f"**Seriousness:** {res['data'].seriousness_criteria}")
        st.write(f"**Reporter:** {res['data'].reporter}")

        st.write("**MedDRA Coded Adverse Events:**")
        for item in res["data"].coded_adverse_events:
            st.success(f"• **Reported:** `{item.verbatim_reported}` ➔ **MedDRA PT:** `{item.meddra_pt}` (Code: `{item.meddra_code}`)")

    # Step 8: Human Approval / Rejection
    st.divider()
    st.subheader("⚖️ Regulated Human Review")
    
    # --- NEW: Display Policy & Risk Engine Routing Decision ---
    risk = res.get("risk")
    if risk:
        if risk.requires_human_review:
            st.error(f"🚨 **Action Required (Routed to Human):** {risk.routing_reason}")
        else:
            st.success(f"✅ **Auto-Approval Recommended:** {risk.routing_reason}")
    
    reviewer_name = st.text_input("Reviewer Identifier", value="Safety_Officer_Kavitha")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("✅ Approve Case", type="primary"):
            commit_to_audit_trail(
                res["case_id"], res["filename"], res["data"], 
                res["history"], "APPROVED", reviewer_name
            )
            st.balloons()
            st.success(f"Case {res['case_id']} approved and written to SQLite Audit Trail!")

    with c2:
        if st.button("✍️ Request Edits"):
            commit_to_audit_trail(
                res["case_id"], res["filename"], res["data"], 
                res["history"], "PENDING_EDITS", reviewer_name
            )
            st.warning(f"Case {res['case_id']} flagged for revisions.")

    with c3:
        if st.button("❌ Reject Report"):
            commit_to_audit_trail(
                res["case_id"], res["filename"], res["data"], 
                res["history"], "REJECTED", reviewer_name
            )
            st.error(f"Case {res['case_id']} rejected and recorded.")


# --- FEATURE: AI Chatbot & Notifications ---
st.divider()
st.header("💬 Case Assistant & Notifications")

# 1. Initialize Chat History in Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hello! I am your PharmaAgent. Do you want me to summarize the current case and send it to your WhatsApp, or do you have any questions about the uploaded document?"}
    ]

# 2. Display Chat History
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Chat Input Box
user_input = st.chat_input("Type your message here...")

if user_input:
    # Save and display user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    # 4. Smart Chatbot Logic
    with st.chat_message("assistant"):
        response = ""
        
        # Action 1: Send WhatsApp Alert
        if "whatsapp" in user_input.lower() or "send" in user_input.lower():
            if "pipeline_output" in st.session_state:
                case_id = st.session_state["pipeline_output"]["case_id"]
                data = st.session_state["pipeline_output"]["data"]
                
                # Safely grab the first MedDRA term, or fallback if empty
                event_pt = data.coded_adverse_events[0].meddra_pt if data.coded_adverse_events else "Unknown Event"
                summary = f"Patient: {data.patient_age} yrs, {data.patient_gender}\nDrug: {data.suspect_product}\nEvent: {event_pt}"
                
                # ⚠️ IMPORTANT: UPDATE THIS NUMBER ⚠️
                my_number = "+91YOURNUMBERHERE" # Include your country code!
                
                with st.spinner("Sending WhatsApp alert..."):
                    result = send_whatsapp_alert(case_id, summary, my_number)
                response = f"Done! {result}"
            else:
                response = "Please run a safety case through the pipeline first!"
                
        # Action 2: Document Q&A (Ask questions about the PDF)
        elif "pipeline_output" in st.session_state:
            document_text = st.session_state["pipeline_output"]["raw_text"]
            with st.spinner("Reading documents to find your answer..."):
                response = document_qa_agent(user_input, document_text)
                
        # Fallback if no documents are uploaded yet
        else:
            response = "Please upload and process a safety document first so I have something to read!"
            
        st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})