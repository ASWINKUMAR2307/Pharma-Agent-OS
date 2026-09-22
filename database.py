import sqlite3
import json

def setup_database():
    conn = sqlite3.connect("evidence_graph.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS audit_trail (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_text TEXT,
            extracted_json TEXT,
            status TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_to_evidence_graph(source_text, extracted_data, status):
    conn = sqlite3.connect("evidence_graph.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO audit_trail (source_text, extracted_json, status) VALUES (?, ?, ?)",
        (source_text, json.dumps(extracted_data), status)
    )
    conn.commit()
    conn.close()