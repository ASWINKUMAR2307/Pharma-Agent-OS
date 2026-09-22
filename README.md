PharmaAgentOS
An Enterprise-Grade AI Multi-Agent System for Pharmaceutical Pharmacovigilance

PharmaAgentOS is an intelligent, scalable state-machine pipeline designed to automate the extraction, coding, and regulatory routing of adverse event reports. By utilizing a coordinated workforce of AI agents, it reduces manual processing time while ensuring strict regulatory compliance, immutable audit trails, and human-in-the-loop governance.

🌟 Key Features
LangGraph Orchestrator: A deterministic state machine that controls agent routing, enforces schema validation, and prevents workflow step-skipping.

Intake & OCR Pipeline: Automatically extracts unstructured text from single or multiple adverse event PDF reports using pdfplumber.

Medical Extraction & Coding Agent: Powered by Groq (openai/gpt-oss-20b) and LangChain, this agent extracts patient demographics, suspect products, and maps verbatim adverse events to standardized MedDRA codes.

Duplicate Detection Agent: Utilizes ChromaDB vector embeddings to search historical safety cases and flag potential duplicates based on semantic similarity.

Policy & Risk Engine: A dynamic routing module that evaluates AI findings and QC checks. Low-risk cases are flagged for Auto-Approval, while high-risk or fatal cases are routed to the Human Review Queue.

SQLite Audit Sink: Ensures regulatory traceability by logging every AI extraction, risk decision, and human approval/rejection into a local evidence_graph.db.

Streamlit Review Dashboard: A dual-pane interface for Safety Officers to compare source documents against AI-extracted data before making a final approval decision.

Twilio WhatsApp Integration: Pushes instant case summaries and high-risk alerts directly to a reviewer's mobile device.

RAG Document Assistant: A built-in chat interface allowing reviewers to ask specific questions about the uploaded PDF context.

🏗️ Architecture Workflow
Input: PDF Upload (Email/Web Form simulation).

Intake Agent: Parses raw text and assigns a Case ID.

Case & Coding Agent: Extracts entities and translates symptoms to MedDRA terminology.

Duplicate Agent: Performs vector search against historical data to find matches.

QC Agent: Executes deterministic rule and completeness validation.

Policy & Risk Engine: Routes to Auto-Approve or Human Review based on regulatory severity.

Audit Trail: Final decisions (Approved/Rejected/Pending Edits) committed to the SQLite database.
