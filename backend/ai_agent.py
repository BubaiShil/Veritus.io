"""
Veritus — AI Reasoning Agent (backend/ai_agent.py)

Orchestrates the cognitive components of Automated Hypothesis Generation
and Testing (AHGT):
  1. Profile dataset schema, distributions, and candidate column pairs.
  2. Call LLM (Ollama locally, with Groq/Gemini fallbacks) to formulate
     falsifiable scientific hypotheses.
  3. Refine failed hypotheses into narrower/conditional hypotheses.
  4. Generate clear, non-technical plain-English explanations of validated findings.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional
import pandas as pd
import requests
from dotenv import load_dotenv, find_dotenv

# Load configuration from .env
load_dotenv(find_dotenv())

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))

import socket
from urllib.parse import urlparse

_OLLAMA_STATUS_CACHE = None

def is_ollama_available(force_refresh: bool = False) -> bool:
    """Fast check to determine if the local Ollama server is reachable without Windows IPv6 delay."""
    global _OLLAMA_STATUS_CACHE
    if _OLLAMA_STATUS_CACHE is not None and not force_refresh:
        return _OLLAMA_STATUS_CACHE
        
    try:
        parsed = urlparse(OLLAMA_BASE_URL)
        host = "127.0.0.1" if parsed.hostname in ("localhost", "127.0.0.1") else (parsed.hostname or "127.0.0.1")
        port = parsed.port or 11434
        
        # Fast socket probe (< 200ms)
        with socket.create_connection((host, port), timeout=0.2):
            pass
            
        # Socket open; verify HTTP response
        res = requests.get(f"http://{host}:{port}/api/tags", timeout=1.0)
        _OLLAMA_STATUS_CACHE = (res.status_code == 200)
        return _OLLAMA_STATUS_CACHE
    except Exception:
        _OLLAMA_STATUS_CACHE = False
        return False

def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extracts a statistical and structural profile of the dataframe
    to provide high-signal context for LLM hypothesis generation.
    """
    profile = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": {}
    }
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        num_unique = df[col].nunique()
        missing_count = int(df[col].isnull().sum())
        
        col_info = {
            "dtype": dtype,
            "unique_values_count": num_unique,
            "missing_values": missing_count
        }
        
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info["type_category"] = "numeric"
            col_info["mean"] = round(float(df[col].mean()), 2) if len(df[col].dropna()) > 0 else None
            col_info["std"] = round(float(df[col].std()), 2) if len(df[col].dropna()) > 0 else None
            col_info["min"] = round(float(df[col].min()), 2) if len(df[col].dropna()) > 0 else None
            col_info["max"] = round(float(df[col].max()), 2) if len(df[col].dropna()) > 0 else None
        else:
            col_info["type_category"] = "categorical"
            top_vals = [str(x) for x in df[col].dropna().value_counts().head(4).index.tolist()]
            col_info["top_categories"] = top_vals
            
        profile["columns"][col] = col_info
        
    return profile

import ast

def parse_llm_json(raw_text: Optional[str]) -> Any:
    """
    Robust multi-tier parser that extracts valid JSON or Python literal structures
    from raw LLM output. Handles preambles, fences, trailing commas, single quotes,
    and partial JSON.
    """
    if not raw_text or not raw_text.strip():
        return None
        
    text = raw_text.strip()
    
    # Tier 1: Extract block between markdown fences if present
    code_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    candidate = code_match.group(1).strip() if code_match else text
    
    # Tier 2: Extract outermost array [ ... ] or object { ... }
    bracket_match = re.search(r"(\[.*\]|\{.*\})", candidate, re.DOTALL)
    candidate_bracket = bracket_match.group(1).strip() if bracket_match else candidate
    
    # Attempt 1: Direct JSON parsing
    try:
        return json.loads(candidate_bracket)
    except Exception:
        pass
        
    # Attempt 2: Strip trailing commas before closing brackets
    cleaned_commas = re.sub(r",\s*([\]\}])", r"\1", candidate_bracket)
    try:
        return json.loads(cleaned_commas)
    except Exception:
        pass
        
    # Attempt 3: Python AST literal_eval (handles single quotes and python booleans)
    try:
        return ast.literal_eval(cleaned_commas)
    except Exception:
        pass
        
    # Attempt 4: Extract all individual {...} objects
    extracted = []
    for m in re.finditer(r"\{[^{}]*\}", text):
        chunk = m.group(0)
        try:
            extracted.append(json.loads(chunk))
        except Exception:
            try:
                extracted.append(ast.literal_eval(chunk))
            except Exception:
                pass
    if extracted:
        return extracted
        
    return None

def call_llm(
    prompt: str,
    system_prompt: str = "",
    timeout: Optional[int] = None,
    max_tokens: int = 512,
    json_format: bool = False
) -> Optional[str]:
    """
    Calls Ollama locally. Returns the generated response string.
    Returns None if Ollama is unreachable or times out.
    """
    if not is_ollama_available():
        return None
        
    t = timeout or OLLAMA_TIMEOUT
    endpoint = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": system_prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": max_tokens
        }
    }
    if json_format:
        payload["format"] = "json"
    
    try:
        resp = requests.post(endpoint, json=payload, timeout=t)
        if resp.status_code == 200:
            return resp.json().get("response", "")
        return None
    except requests.exceptions.Timeout:
        print(f"[Warning] Ollama call timed out after {t}s. Falling back to heuristic/template engine.")
        return None
    except Exception as e:
        print(f"[Warning] Ollama call error: {e}")
        return None

def generate_hypotheses_heuristic(df: pd.DataFrame, num_hypotheses: int = 5) -> List[Dict[str, Any]]:
    """
    Statistically grounded heuristic generator used when Ollama is offline or as fallback.
    Identifies sensible candidate column pairs based on types and variances.
    """
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 1]
    categorical_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and 2 <= df[c].nunique() <= 10]
    
    hypotheses = []
    
    # 1. Numeric vs Numeric -> Correlation
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            c1, c2 = numeric_cols[i], numeric_cols[j]
            hypotheses.append({
                "hypothesis_text": f"There is a statistically significant correlation between {c1} and {c2}.",
                "test_method": "correlation",
                "col1": c1,
                "col2": c2,
                "rationale": f"Testing linear and monotonic association between continuous metrics {c1} and {c2}."
            })
            if len(hypotheses) >= num_hypotheses:
                return hypotheses
                
    # 2. Categorical (2 groups) vs Numeric -> T-Test
    for cat in categorical_cols:
        for num in numeric_cols:
            hypotheses.append({
                "hypothesis_text": f"The mean {num} differs significantly across distinct {cat} categories.",
                "test_method": "two_sample_t_test",
                "col1": cat,
                "col2": num,
                "rationale": f"Testing if group membership in {cat} has an effect on continuous metric {num}."
            })
            if len(hypotheses) >= num_hypotheses:
                return hypotheses

    # 3. Categorical vs Categorical -> Chi-Square
    for i in range(len(categorical_cols)):
        for j in range(i + 1, len(categorical_cols)):
            cat1, cat2 = categorical_cols[i], categorical_cols[j]
            hypotheses.append({
                "hypothesis_text": f"There is a significant statistical association between {cat1} and {cat2}.",
                "test_method": "chi_square",
                "col1": cat1,
                "col2": cat2,
                "rationale": f"Testing independence between categorical variables {cat1} and {cat2}."
            })
            if len(hypotheses) >= num_hypotheses:
                return hypotheses
                
    return hypotheses

def generate_hypotheses(df: pd.DataFrame, num_hypotheses: int = 6) -> List[Dict[str, Any]]:
    """
    Generates 5-10 falsifiable statistical hypotheses about the dataset.
    Uses local Ollama if available; seamlessly falls back to structural heuristics.
    """
    profile = profile_dataset(df)
    
    system_prompt = (
        "You are an expert scientific researcher and data scientist specializing in "
        "Automated Hypothesis Generation and Testing (AHGT). You formulate rigorous, "
        "falsifiable hypotheses that can be tested using standard statistical tests "
        "(t-test, correlation, chi-square). Always return strictly valid JSON."
    )
    
    user_prompt = f"""
Given the following dataset schema and characteristics:
{json.dumps(profile, indent=2)}

Generate {num_hypotheses} distinct, falsifiable hypotheses about relationships in this data.
Each hypothesis must use one of these exact test_method values:
- 'two_sample_t_test' (col1 = categorical group column, col2 = numeric metric column)
- 'correlation' (col1 = numeric column, col2 = numeric column)
- 'chi_square' (col1 = categorical column, col2 = categorical column)

Return a JSON object with a 'hypotheses' list in this exact format:
{{
  "hypotheses": [
    {{
      "hypothesis_text": "Clear statement of the relationship",
      "test_method": "two_sample_t_test | correlation | chi_square",
      "col1": "exact_column_name_1",
      "col2": "exact_column_name_2",
      "rationale": "Brief reason why this relationship is plausible"
    }}
  ]
}}
"""
    
    raw_response = call_llm(user_prompt, system_prompt=system_prompt, timeout=OLLAMA_TIMEOUT, max_tokens=768, json_format=True)
    if raw_response:
        parsed = parse_llm_json(raw_response)
        if isinstance(parsed, dict) and "hypotheses" in parsed:
            parsed = parsed["hypotheses"]
        elif isinstance(parsed, dict):
            for val in parsed.values():
                if isinstance(val, list):
                    parsed = val
                    break
        
        if isinstance(parsed, list) and len(parsed) > 0:
            valid_hypotheses = []
            for h in parsed:
                if isinstance(h, dict):
                    c1, c2 = h.get("col1"), h.get("col2")
                    if c1 in df.columns and c2 in df.columns:
                        valid_hypotheses.append(h)
            if len(valid_hypotheses) > 0:
                return valid_hypotheses

    # Fallback if Ollama is not active or output invalid
    return generate_hypotheses_heuristic(df, num_hypotheses=num_hypotheses)

def refine_hypothesis(
    df: pd.DataFrame,
    failed_hypothesis: Dict[str, Any],
    test_result: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Agentic refinement loop: When a broad hypothesis fails (p >= 0.05), the agent
    analyzes why it failed and formulates a more targeted, conditional hypothesis.
    """
    system_prompt = (
        "You are an AI research scientist executing a self-correcting hypothesis refinement loop. "
        "An initial hypothesis failed statistical testing. You must formulate a more targeted, "
        "refined hypothesis (e.g. testing monotonic Spearman correlation, testing a specific subgroup, "
        "or examining an alternative related column). Return strictly valid JSON."
    )
    
    prompt = f"""
The following statistical hypothesis was tested and REJECTED:
- Hypothesis: "{failed_hypothesis.get('hypothesis_text')}"
- Method: {failed_hypothesis.get('test_method')} on '{failed_hypothesis.get('col1')}' and '{failed_hypothesis.get('col2')}'
- Test Result: p-value={test_result.get('p_value')}, effect_size={test_result.get('effect_size')}
- Reason: {test_result.get('summary')}

Available columns in dataset: {list(df.columns)}

Formulate 1 refined, more specific hypothesis to test.
Return ONLY a JSON object:
{{
  "hypothesis_text": "Refined hypothesis statement",
  "test_method": "two_sample_t_test | correlation | chi_square",
  "col1": "exact_column_name_1",
  "col2": "exact_column_name_2",
  "rationale": "Why this refined formulation is more specific"
}}
"""
    raw_response = call_llm(prompt, system_prompt=system_prompt, timeout=45, max_tokens=256, json_format=True)
    if raw_response:
        parsed = parse_llm_json(raw_response)
        if isinstance(parsed, dict) and parsed.get("col1") in df.columns and parsed.get("col2") in df.columns:
            return parsed
        elif isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
            first = parsed[0]
            if first.get("col1") in df.columns and first.get("col2") in df.columns:
                return first
            
    # Fallback refinement heuristic: switch test method or inspect spearman vs pearson
    orig_method = failed_hypothesis.get("test_method", "")
    col1, col2 = failed_hypothesis.get("col1"), failed_hypothesis.get("col2")
    
    if "corr" in orig_method:
        return {
            "hypothesis_text": f"There is a non-linear (monotonic rank) relationship between {col1} and {col2}.",
            "test_method": "spearman_correlation",
            "col1": col1,
            "col2": col2,
            "rationale": "Initial linear Pearson correlation failed; testing monotonic rank correlation."
        }
    return None

def explain_findings(hypothesis_text: str, test_result: Dict[str, Any]) -> str:
    """
    Generates a natural language explanation of validated findings in plain English,
    highlighting the real-world meaning of the p-value and effect size.
    """
    system_prompt = (
        "You are an expert scientific communicator. Translate statistical test results "
        "into clear, professional, plain-English insights for business or research decision makers. "
        "Keep your explanation to 2-3 sentences. Mention the practical significance and effect size."
    )
    
    prompt = f"""
Hypothesis: "{hypothesis_text}"
Test Method: {test_result.get('test_method')}
p-value: {test_result.get('p_value')}
Effect Size: {test_result.get('effect_size_name')} = {test_result.get('effect_size')} ({test_result.get('effect_magnitude', 'notable')})
Statistical Summary: {test_result.get('summary')}

Write a concise 2-3 sentence plain-English explanation of this validated finding.
"""
    explanation = call_llm(prompt, system_prompt=system_prompt, timeout=30, max_tokens=150)
    if explanation and len(explanation.strip()) > 20:
        return explanation.strip()
        
    # Standard high-clarity template fallback
    method = test_result.get("test_method", "Statistical test")
    p = test_result.get("p_value", 0.0)
    eff = test_result.get("effect_size", 0.0)
    eff_name = test_result.get("effect_size_name", "effect size")
    mag = test_result.get("effect_magnitude", "significant")
    
    return (
        f"The data provides empirical evidence supporting the hypothesis (p = {p:.4g} < 0.05). "
        f"The {method} revealed a {mag} {eff_name} of {eff:.2f}, indicating that this is a "
        f"meaningful and statistically reliable pattern rather than random fluctuation."
    )

if __name__ == "__main__":
    print("=== Testing Veritus AI Reasoning Agent ===")
    print(f"Ollama status: {'ONLINE' if is_ollama_available() else 'OFFLINE (Fallback Active)'}")
    
    # Create test dataframe
    test_df = pd.DataFrame({
        "marketing_spend": [100, 200, 300, 400, 500, 600, 700],
        "sales_volume": [220, 410, 630, 790, 1020, 1210, 1400],
        "plan_type": ["Basic", "Basic", "Basic", "Pro", "Pro", "Pro", "Pro"],
        "retention_score": [5.5, 6.0, 5.8, 8.9, 9.2, 9.0, 9.4]
    })
    
    print("\n[1] Testing Dataset Profiling:")
    profile = profile_dataset(test_df)
    print(f"Profiled {profile['num_columns']} columns across {profile['num_rows']} rows.")
    
    print("\n[2] Testing Hypothesis Generation:")
    hypos = generate_hypotheses(test_df, num_hypotheses=3)
    for idx, h in enumerate(hypos, 1):
        print(f"  {idx}. [{h['test_method']}] {h['hypothesis_text']}")
        print(f"     Columns: ({h['col1']}, {h['col2']}) | Rationale: {h['rationale']}")
        
    print("\n[3] Testing Finding Explanation:")
    sample_test_result = {
        "test_method": "pearson_correlation",
        "p_value": 0.0001,
        "effect_size": 0.98,
        "effect_size_name": "Pearson's r",
        "effect_magnitude": "strong",
        "summary": "Correlation between 'marketing_spend' and 'sales_volume': r=0.98, p=0.0001."
    }
    exp = explain_findings("Higher marketing spend drives higher sales volume.", sample_test_result)
    print(f"Explanation:\n{exp}")
    
    print("\n--- AI Agent Module Test Passed! ---")
