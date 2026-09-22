def intake_agent(state: dict):
    # Simulates pdfplumber reading a document and triaging
    text = state.get("raw_text", "Patient is 45 years old, experiencing severe headache.")
    state["raw_text"] = text
    state["audit_trail"].append("Intake Agent: OCR and Triage complete.")
    return state

def safety_case_agent(state: dict):
    # Simulates AI extracting patient, event, and seriousness
    state["extracted_data"] = {
        "patient_age": "45", 
        "adverse_event": "headache", 
        "seriousness": "severe"
    }
    state["audit_trail"].append("Safety Case Agent: Data extracted.")
    return state

def coding_agent(state: dict):
    # Simulates matching the event to MedDRA terminology[cite: 1]
    state["extracted_data"]["meddra_code"] = "10019211" # MedDRA code for Headache
    state["audit_trail"].append("Coding Agent: MedDRA code recommended.")
    return state

def duplicate_agent(state: dict):
    # Simulates ChromaDB checking historical cases[cite: 1]
    state["extracted_data"]["is_duplicate"] = False
    state["audit_trail"].append("Duplicate Agent: No similar historical cases found.")
    return state

def qc_agent(state: dict):
    # Simulates independent validation for missing data/contradictions[cite: 1]
    data = state["extracted_data"]
    if data.get("patient_age") and data.get("adverse_event"):
        state["extracted_data"]["qc_passed"] = True
        state["audit_trail"].append("QC Agent: Completeness and coding validated.")
    else:
        state["extracted_data"]["qc_passed"] = False
        state["audit_trail"].append("QC Agent: Validation failed, missing data.")
    return state