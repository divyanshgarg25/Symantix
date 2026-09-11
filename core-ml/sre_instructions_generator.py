import json
import os

def generate_sre_instructions(output_path: str, num_samples: int = 100):
    """
    Generates synthetic training data pairing GraphRAG contexts 
    with target deterministic JSON diagnosis schemas.
    """
    prompt_template = "Analyze the following system log trace context and determine the primary root cause, failure blast radius, and recovery steps.\n\nContext:\n{context}"
    
    samples = []
    
    # Template 1: DB Pool Exhaustion
    for _ in range(num_samples // 2):
        context = """[Auth-Service] -> EMITTED -> [TimeoutException: Failed to acquire sql connection, pool exhausted after 3000ms.]
[Auth-Service] -> DEPENDS_ON -> [User-PostgreSQL]
[Payment-API] -> DEPENDS_ON -> [Auth-Service]
[Frontend-Gateway] -> EMITTED -> [500 Internal Server Error]"""
        
        response_json = {
            "root_cause": "PostgreSQL connection pool depletion due to an unindexed query string.",
            "blast_radius": ["Auth-Service", "Payment-API", "Frontend-Gateway"],
            "remediation": "Execute ALTER INDEX on user_id columns and restart the db pool."
        }
        
        samples.append({
            "instruction": prompt_template.format(context=context),
            "response": json.dumps(response_json, indent=2)
        })
        
    # Template 2: Upstream Gateway Timeout
    for _ in range(num_samples // 2):
        context = """[Frontend-Gateway] -> EMITTED -> [504 Gateway Timeout]
[Frontend-Gateway] -> DEPENDS_ON -> [Search-Service]
[Search-Service] -> EMITTED -> [High CPU Usage 99%]"""
        
        response_json = {
            "root_cause": "Search-Service CPU saturated, causing downstream requests to timeout at the Gateway.",
            "blast_radius": ["Frontend-Gateway"],
            "remediation": "Scale up Search-Service replicas and implement request rate limiting on Frontend-Gateway."
        }
        
        samples.append({
            "instruction": prompt_template.format(context=context),
            "response": json.dumps(response_json, indent=2)
        })
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
            
    print(f"Generated {len(samples)} samples at {output_path}")

if __name__ == "__main__":
    generate_sre_instructions("sre_instructions.jsonl")
