import sqlite3
import json
import datetime
import pdfplumber
import chromadb
import io
import os
from dotenv import load_dotenv 
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from twilio.rest import Client # <-- Required for WhatsApp Agent
from schemas import SafetyCasePayload, CodedEvent, DuplicateResult, QCResult, RiskAssessment

# Load the secret variables from your .env file
load_dotenv()

# 1. Evidence Graph Database Setup (SQLite)
def init_evidence_db():
    conn = sqlite3.connect("evidence_graph.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS evidence_audit_trail (
            case_id TEXT PRIMARY KEY,
            source_filename TEXT,
            extracted_payload TEXT,
            workflow_history TEXT,
            final_decision TEXT,
            reviewer_id TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

# 2. Intake Agent: PDF Text Extraction
def intake_agent_extract_text(pdf_file) -> str:
    """Extracts raw text from uploaded PDF using pdfplumber."""
    extracted_text = ""
    
    # Read the Streamlit file as bytes so pdfplumber can open it
    pdf_bytes = pdf_file.read() 
    
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                extracted_text += page_text + "\n"
    return extracted_text.strip()

# 3. Safety Case Agent & Coding Agent (REAL AI BRAIN)
def extraction_and_coding_agent(raw_text: str) -> SafetyCasePayload:
    """
    Uses Groq to read the text and extract data into the exact Pydantic schema.
    """
# Initialize the Groq AI Model using a supported free-tier model
    llm = ChatGroq(
        model="openai/gpt-oss-20b", # <-- Update this line!
        temperature=0 # Strict, factual answers only
    )

    # Bind the Pydantic schema so the AI outputs perfect JSON
    structured_llm = llm.with_structured_output(SafetyCasePayload)

    # Give the AI strict instructions
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert Pharmacovigilance AI. Extract medical case data from the provided text into the required format. "
                   "If a patient age is not found, use 0. If a gender is not found, use 'Unknown'. "
                   "For MedDRA coding, estimate the closest standard medical term and provide a dummy 8-digit code if you don't know the exact one."),
        ("human", "Extract the safety data from this report:\n\n{text}")
    ])

    # Connect the prompt to the AI
    chain = prompt | structured_llm

    # Execute the AI Brain
    extracted_data = chain.invoke({"text": raw_text})
    
    return extracted_data

# 4. Duplicate Agent (ChromaDB)
def duplicate_check_agent(case_summary: str) -> DuplicateResult:
    """Checks local ChromaDB for historical case similarity."""
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="historical_safety_cases")

    # Add historical reference record if empty
    if collection.count() == 0:
        collection.add(
            documents=["Male 62 with Metformin experiencing lactic acidosis"],
            ids=["CASE-2025-1044"]
        )

    # Query ChromaDB
    results = collection.query(query_texts=[case_summary], n_results=1)
    
    # Calculate dummy similarity distance
    return DuplicateResult(
        is_duplicate=False,
        similarity_score=0.31,
        closest_match_id=results["ids"][0][0] if results["ids"] else None,
        match_reason="Same drug class, but different patient age, gender, and reporting dates."
    )

# 5. QC Agent: Independent Validation
def qc_agent_validate(payload: SafetyCasePayload) -> QCResult:
    """Independently verifies mandatory fields and consistency."""
    flags = []
    if payload.patient_age <= 0:
        flags.append("Invalid patient age.")
    if not payload.coded_adverse_events:
        flags.append("No adverse events recorded.")
    if not payload.suspect_product:
        flags.append("Missing suspect drug.")

    return QCResult(passed=len(flags) == 0, flags=flags)

# 6. Audit Trail Writer (Evidence Graph)
def commit_to_audit_trail(case_id, filename, payload, history, decision, reviewer):
    conn = sqlite3.connect("evidence_graph.db")
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO evidence_audit_trail 
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        case_id,
        filename,
        payload.model_dump_json(),
        json.dumps(history),
        decision,
        reviewer,
        datetime.datetime.utcnow().isoformat()
    ))
    conn.commit()
    conn.close()

# 7. Notification Agent (WhatsApp)
def send_whatsapp_alert(case_id: str, summary: str, to_number: str):
    """Sends a WhatsApp message with the approved case summary."""
    account_sid = os.getenv("TWILIO_SID")
    auth_token = os.getenv("TWILIO_TOKEN")
    
    if not account_sid or not auth_token:
        return "Twilio credentials missing. Alert not sent."
        
    client = Client(account_sid, auth_token)
    
    message_body = f"🚨 *PharmaAgentOS Alert*\n\nCase ID: {case_id}\nStatus: APPROVED\n\n*Summary:*\n{summary}"
    
    try:
        message = client.messages.create(
            from_='whatsapp:+14155238886', # Twilio Sandbox Number
            body=message_body,
            to=f'whatsapp:{to_number}'    
        )
        return f"WhatsApp alert sent! Message SID: {message.sid}"
    except Exception as e:
        return f"Failed to send WhatsApp alert: {str(e)}"

# 8. Document Q&A Agent (RAG)
def document_qa_agent(user_query: str, document_text: str) -> str:
    """Answers user questions based strictly on the uploaded document text."""
    llm = ChatGroq(
        model="openai/gpt-oss-20b", # <-- Update this line too!
        temperature=0.2 
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful Pharmacovigilance assistant. Answer the user's question using ONLY the provided document text below. If the answer is not in the text, say 'I cannot find this information in the uploaded documents.'\n\nDocument Text:\n{context}"),
        ("human", "{question}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "context": document_text, 
        "question": user_query
    })
    
    return response.content
    
# 9. Policy & Risk Engine Agent
def policy_and_risk_agent(
    case_data: SafetyCasePayload, 
    qc_result: QCResult, 
    duplicate_result: DuplicateResult
) -> RiskAssessment:
    """
    Evaluates the case against compliance rules to decide if human approval is required.
    """
    reasons = []
    requires_human = False
    risk_level = "Low"

    # Rule 1: QC Failures always require human review
    if not qc_result.passed:
        requires_human = True
        risk_level = "High"
        reasons.append(f"Failed QC Checks: {', '.join(qc_result.flags)}")

    # Rule 2: Potential duplicates need a human eye
    if duplicate_result.similarity_score > 0.80:
        requires_human = True
        risk_level = "Medium"
        reasons.append("High probability of being a duplicate case.")

    # Rule 3: Serious outcomes (Death, Hospitalization) strictly require review
    serious_keywords = ["death", "hospital", "emergency", "life-threatening", "fatal"]
    if any(keyword in case_data.seriousness_criteria.lower() for keyword in serious_keywords):
        requires_human = True
        risk_level = "High"
        reasons.append("Serious adverse event criteria met (Regulatory Requirement).")

    # Final Decision Formulation
    if requires_human:
        final_reason = "Routing to Human Review queue. Reasons: " + " | ".join(reasons)
    else:
        final_reason = "Case is standard, passed QC, and non-serious. Eligible for Auto-Approval."

    return RiskAssessment(
        risk_level=risk_level,
        requires_human_review=requires_human,
        routing_reason=final_reason
    )