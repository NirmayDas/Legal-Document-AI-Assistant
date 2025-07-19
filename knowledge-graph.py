"""
Knowledge Graph Contract Extractor & CLI

Instructions:
1. Install dependencies:
   pip install sentence-transformers chromadb pypdf tiktoken isodate matplotlib seaborn tqdm pydantic requests
2. Start your Gemma proxy (or set GEMMA_URL to your Cloud Run endpoint).
3. Place your contract .txt files in a folder, e.g., ./contracts/
4. Run this script:
   python knowledge-graph.py

Outputs will be saved in ./output/
"""

import os
import json
import asyncio
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field
from tqdm import tqdm
import isodate
import requests
from sentence_transformers import SentenceTransformer

# ========== CONFIG =============
CONTRACTS_FOLDER = './contracts'  # Change this to your contracts folder
OUTPUT_FOLDER = './output'
RESULTS_JSON = os.path.join(OUTPUT_FOLDER, 'contract_data.json')
EMBEDDINGS_PKL = os.path.join(OUTPUT_FOLDER, 'embedding_outputs.pkl')
GEMMA_URL = os.environ.get('GEMMA_URL', 'http://localhost:9090/api/generate')
GEMMA_MODEL = os.environ.get('GEMMA_MODEL', 'gemma3:1b')

# ========== DATA MODELS =============
CLAUSE_TYPES = [
    "Renewal & Termination",
    "Confidentiality & Non-Disclosure",
    "Non-Compete & Exclusivity",
    "Liability & Indemnification",
    "Service-Level Agreements"
]

class Clause(BaseModel):
    summary: str = Field(..., description="Summary of the clause using no pronouns")
    clause_type: str = Field(..., description="Clause types", enum=CLAUSE_TYPES)

class Location(BaseModel):
    address: Optional[str] = Field(..., description="The street address of the location.Use None if not provided")
    city: Optional[str] = Field(..., description="The city of the location.Use None if not provided")
    state: Optional[str] = Field(..., description="The state or region of the location.Use None if not provided")
    country: str = Field(..., description="The country of the location. Use the two-letter ISO standard.")

class Organization(BaseModel):
    name: str = Field(..., description="The name of the organization.")
    location: Location = Field(..., description="The primary location of the organization.")
    role: str = Field(..., description="The role of the organization in the contract, such as 'provider', 'client', 'supplier', etc.")

CONTRACT_TYPES = [
    "Affiliate Agreement", "Development",
    "Distributor",
    "Endorsement",
    "Franchise",
    "Hosting",
    "IP",
    "Joint Venture",
    "License Agreement",
    "Maintenance",
    "Manufacturing",
    "Marketing",
    "Non Compete/Solicit", "Outsourcing",
    "Promotion",
    "Reseller",
    "Service",
    "Sponsorship",
    "Strategic Alliance",
    "Supply",
    "Transportation",
]

class Contract(BaseModel):
    summary: str = Field(..., description=("High level summary of the contract with relevant facts and details. Include all relevant information to provide full picture. Do no use any pronouns"))
    contract_type: str = Field(..., description="The type of contract being entered into.", enum=CONTRACT_TYPES)
    parties: List[Organization] = Field(..., description="List of parties involved in the contract, with details of each party's role.")
    effective_date: str = Field(..., description=("Enter the date when the contract becomes effective in yyyy-MM-dd format. If only the year (e.g., 2015) is known, use 2015-01-01 as the default date. Always fill in full date"))
    contract_scope: str = Field(..., description="Description of the scope of the contract, including rights, duties, and any limitations.")
    duration: Optional[str] = Field(None, description=("The duration of the agreement, including provisions for renewal or termination. Use ISO 8601 durations standard"))
    end_date: Optional[str] = Field(None, description=("The date when the contract expires. Use yyyy-MM-dd format. If only the year (e.g., 2015) is known, use 2015-01-01 as the default date. Always fill in full date"))
    total_amount: Optional[float] = Field(None, description="Total value of the contract.")
    governing_law: Optional[Location] = Field(None, description="The jurisdiction's laws governing the contract.")
    clauses: Optional[List[Clause]] = Field(None, description=f"""Relevant summaries of clause types. Allowed clause types are {CLAUSE_TYPES}""")

# ========== UTILS =============
def read_txt_files(folder_path):
    data = []
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder '{folder_path}' not found. Please create it and add .txt contract files.")
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_id = os.path.splitext(filename)[0]
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                data.append({"file_id": file_id, "text": text})
    return data

def is_valid_date(date_string):
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return True
    except:
        return False

def add_duration_to_date(date_str, duration_str):
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    duration = isodate.parse_duration(duration_str)
    result_date = date_obj + duration
    return result_date.strftime("%Y-%m-%d")

# ========== GEMMA CALL =============
def call_gemma(prompt: str, model: str = GEMMA_MODEL, url: str = GEMMA_URL, temperature: float = 0.7) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature
    }
    response = requests.post(url, json=payload, stream=True)
    if not response.ok:
        raise RuntimeError(f"Gemma API error: {response.status_code} {response.text}")
    # Gemma streams JSON lines; concatenate 'response' fields
    result = ""
    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line.decode())
                if 'response' in data:
                    result += data['response']
            except Exception:
                continue
    return result.strip()

# ========== MAIN PROCESSING =============
def extract_contract_structured(text: str) -> dict:
    # Prompt Gemma to extract contract info as JSON
    prompt = f"""
Extract the following structured information from this contract. Return your answer as a JSON object with these fields:
- summary (string)
- contract_type (string, one of {CONTRACT_TYPES})
- parties (list of organizations, each with name, location (address, city, state, country), and role)
- effective_date (yyyy-MM-dd)
- contract_scope (string)
- duration (ISO 8601 duration or null)
- end_date (yyyy-MM-dd or null)
- total_amount (float or null)
- governing_law (location or null)
- clauses (list of clause objects, each with summary and clause_type from {CLAUSE_TYPES}, or null)

Contract text:
{text}

Return only the JSON object, no explanation.
"""
    response = call_gemma(prompt)
    try:
        # Try to parse the first JSON object in the response
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            return json.loads(response[json_start:json_end])
        else:
            return {"error": "No JSON found in response", "raw": response}
    except Exception as e:
        return {"error": f"Failed to parse JSON: {e}", "raw": response}

def process_contract(contract, semaphore):
    with semaphore:
        structured_data = extract_contract_structured(contract["text"])
        structured_data["file_id"] = contract["file_id"]
        # Clean dates
        if "effective_date" in structured_data:
            structured_data["effective_date"] = structured_data["effective_date"] if is_valid_date(structured_data["effective_date"]) else None
        if "end_date" in structured_data:
            structured_data["end_date"] = structured_data["end_date"] if is_valid_date(structured_data["end_date"]) else None
        # Infer end date
        if not structured_data.get("end_date") and structured_data.get("effective_date") and structured_data.get("duration"):
            try:
                structured_data["end_date"] = add_duration_to_date(structured_data["effective_date"], structured_data["duration"])
            except:
                pass
        return structured_data

def process_all(contracts, max_workers=5):
    from threading import Semaphore
    semaphore = Semaphore(max_workers)
    results = []
    for contract in tqdm(contracts, desc="Processing contracts"):
        results.append(process_contract(contract, semaphore))
    return results

# ========== MAIN SCRIPT =============
def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"Reading contracts from: {CONTRACTS_FOLDER}")
    contracts = read_txt_files(CONTRACTS_FOLDER)
    print(f"Found {len(contracts)} contracts.")
    print("\nProcessing contracts with Gemma (this may take a while)...")
    results = process_all(contracts)
    with open(RESULTS_JSON, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, indent=2)
    print(f"\nStructured contract data saved to {RESULTS_JSON}")
    print("\nGenerating embeddings with sentence-transformers...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    summaries = [el["summary"] for el in results if "summary" in el]
    embeddings_output = model.encode(summaries)
    import pickle as pkl
    with open(EMBEDDINGS_PKL, 'wb') as f:
        pkl.dump(embeddings_output, f)
    print(f"Embeddings saved to {EMBEDDINGS_PKL}")
    print("\nEntering interactive prompt mode. Type 'quit' to exit.")
    while True:
        query = input("\nPrompt: ").strip()
        if query.lower() == 'quit':
            print("Exiting.")
            break
        # Send the query to Gemma and print the response
        try:
            response = call_gemma(query)
            print(f"Gemma: {response}")
        except Exception as e:
            print(f"[Error] Failed to get response from Gemma: {e}")

if __name__ == "__main__":
    main()


