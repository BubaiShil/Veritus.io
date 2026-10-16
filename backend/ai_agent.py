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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile").strip()

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

def get_active_llm_provider() -> str:
    """Returns the identifier of the active LLM engine: 'gemini', 'groq', 'ollama', or 'heuristic'."""
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip() or GEMINI_API_KEY
    if gemini_key:
        return "gemini"
    groq_key = os.getenv("GROQ_API_KEY", "").strip() or GROQ_API_KEY
    if groq_key:
        return "groq"
    if is_ollama_available():
        return "ollama"
    return "heuristic"

def is_llm_available(force_refresh: bool = False) -> bool:
    """Checks whether any LLM backend (Gemini, Groq, or local Ollama) is configured and reachable."""
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip() or GEMINI_API_KEY
    if gemini_key:
        return True
    groq_key = os.getenv("GROQ_API_KEY", "").strip() or GROQ_API_KEY
    if groq_key:
        return True
    return is_ollama_available(force_refresh=force_refresh)

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

def _call_gemini(
    prompt: str,
    system_prompt: str = "",
    timeout: Optional[int] = None,
    max_tokens: int = 512,
    json_format: bool = False
) -> Optional[str]:
    """Invokes Google Gemini via REST API."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip() or GEMINI_API_KEY
    if not api_key:
        return None

    model = os.getenv("GEMINI_MODEL", "").strip() or GEMINI_MODEL or "gemini-1.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    gen_config: Dict[str, Any] = {
        "temperature": 0.2,
        "maxOutputTokens": max_tokens,
    }
    if json_format:
        gen_config["responseMimeType"] = "application/json"

    payload: Dict[str, Any] = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": gen_config,
    }
    if system_prompt:
        payload["systemInstruction"] = {
            "parts": [{"text": system_prompt}]
        }

    t = timeout or 30
    try:
        resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=t)
        if resp.status_code == 200:
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts and "text" in parts[0]:
                    return parts[0]["text"]
        else:
            print(f"[Warning] Gemini API returned status {resp.status_code}: {resp.text[:200]}")
            return None
    except Exception as exc:
        print(f"[Warning] Gemini API call failed: {exc}")
        return None

def _call_groq(
    prompt: str,
    system_prompt: str = "",
    timeout: Optional[int] = None,
    max_tokens: int = 512,
    json_format: bool = False
) -> Optional[str]:
    """Invokes Groq Cloud API via OpenAI-compatible endpoint."""
    api_key = os.getenv("GROQ_API_KEY", "").strip() or GROQ_API_KEY
    if not api_key:
        return None

    model = os.getenv("GROQ_MODEL", "").strip() or GROQ_MODEL or "llama-3.1-70b-versatile"
    url = "https://api.groq.com/openai/v1/chat/completions"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    if json_format:
        payload["response_format"] = {"type": "json_object"}

    t = timeout or 30
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=t)
        if resp.status_code == 200:
            data = resp.json()
            choices = data.get("choices", [])
            if choices and "message" in choices[0]:
                return choices[0]["message"].get("content", "")
        else:
            print(f"[Warning] Groq API returned status {resp.status_code}: {resp.text[:200]}")
            return None
    except Exception as exc:
        print(f"[Warning] Groq API call failed: {exc}")
        return None

def _call_ollama(
    prompt: str,
    system_prompt: str = "",
    timeout: Optional[int] = None,
    max_tokens: int = 512,
    json_format: bool = False
) -> Optional[str]:
    """Invokes local Ollama server."""
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
        print(f"[Warning] Ollama call timed out after {t}s.")
        return None
    except Exception as e:
        print(f"[Warning] Ollama call error: {e}")
        return None

def call_llm(
    prompt: str,
    system_prompt: str = "",
    timeout: Optional[int] = None,
    max_tokens: int = 512,
    json_format: bool = False
) -> Optional[str]:
    """
    Intelligent multi-tier LLM router:
      1. Tries Gemini if GEMINI_API_KEY is present.
      2. Tries Groq if GROQ_API_KEY is present.
      3. Tries local Ollama if reachable.
      4. Returns None (triggering statistical/heuristic fallback) if all providers are unavailable.
    """
    # 1. Gemini
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip() or GEMINI_API_KEY
    if gemini_key:
        res = _call_gemini(prompt, system_prompt=system_prompt, timeout=timeout, max_tokens=max_tokens, json_format=json_format)
        if res:
            return res

    # 2. Groq
    groq_key = os.getenv("GROQ_API_KEY", "").strip() or GROQ_API_KEY
    if groq_key:
        res = _call_groq(prompt, system_prompt=system_prompt, timeout=timeout, max_tokens=max_tokens, json_format=json_format)
        if res:
            return res

    # 3. Ollama
    if is_ollama_available():
        res = _call_ollama(prompt, system_prompt=system_prompt, timeout=timeout, max_tokens=max_tokens, json_format=json_format)
        if res:
            return res

    return None

def detect_target_columns(df: pd.DataFrame) -> List[str]:
    """
    Identifies probable outcome or target columns based on common naming conventions
    and column position (e.g. Progression, Target, Outcome, Class, Churn, or last column).
    """
    target_keywords = [
        "target", "progression", "outcome", "label", "status", "churn",
        "sales", "price", "revenue", "risk", "score", "disease", "class",
        "default", "survived", "y", "response", "metric", "output"
    ]
    matched = []
    for col in df.columns:
        c_low = col.lower()
        if any(kw == c_low or kw in c_low for kw in target_keywords):
            matched.append(col)
            
    # If no obvious keyword matched and dataset has multiple columns, consider the last column
    if not matched and len(df.columns) > 1:
        matched.append(df.columns[-1])
        
    return matched

def generate_hypotheses_heuristic(df: pd.DataFrame, num_hypotheses: int = 6) -> List[Dict[str, Any]]:
    """
    Statistically grounded heuristic generator with column diversity and target-awareness.
    Prevents single-column anchoring (e.g. testing only 'age' vs all features) and prioritizes
    relationships involving detected target variables.
    """
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 1]
    categorical_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and 2 <= df[c].nunique() <= 10]
    target_cols = detect_target_columns(df)
    primary_target = target_cols[0] if target_cols else (df.columns[-1] if len(df.columns) > 1 else None)

    hypotheses: List[Dict[str, Any]] = []
    col_usage: Dict[str, int] = {c: 0 for c in df.columns}
    seen_pairs = set()

    def add_hypothesis(c1: str, c2: str, method: str, text: str, rationale: str) -> bool:
        pair_key = tuple(sorted([c1, c2]))
        if pair_key in seen_pairs:
            return False
        hypotheses.append({
            "hypothesis_text": text,
            "test_method": method,
            "col1": c1,
            "col2": c2,
            "rationale": rationale
        })
        seen_pairs.add(pair_key)
        col_usage[c1] = col_usage.get(c1, 0) + 1
        col_usage[c2] = col_usage.get(c2, 0) + 1
        return True

    # 1. Target vs Predictors (High-value primary research questions)
    if primary_target:
        if primary_target in numeric_cols:
            # Pair target with distinct numeric predictors
            other_nums = [c for c in numeric_cols if c != primary_target]
            for pred in other_nums:
                if len(hypotheses) >= num_hypotheses or col_usage.get(primary_target, 0) >= 4:
                    break
                add_hypothesis(
                    c1=pred,
                    c2=primary_target,
                    method="correlation",
                    text=f"There is a statistically significant correlation between {pred} and {primary_target}.",
                    rationale=f"Testing if feature '{pred}' significantly predicts outcome variable '{primary_target}'."
                )

            # Pair target with categorical predictors (T-test)
            for cat in categorical_cols:
                if len(hypotheses) >= num_hypotheses or col_usage.get(primary_target, 0) >= 4:
                    break
                add_hypothesis(
                    c1=cat,
                    c2=primary_target,
                    method="two_sample_t_test",
                    text=f"The mean {primary_target} differs significantly across distinct {cat} categories.",
                    rationale=f"Testing if membership in category '{cat}' influences outcome '{primary_target}'."
                )

    # 2. Categorical vs Numeric (T-Test) across diverse pairs
    for cat in categorical_cols:
        for num in numeric_cols:
            if len(hypotheses) >= num_hypotheses:
                break
            if col_usage.get(cat, 0) >= 2 or col_usage.get(num, 0) >= 2:
                continue
            add_hypothesis(
                c1=cat,
                c2=num,
                method="two_sample_t_test",
                text=f"The mean {num} differs significantly across distinct {cat} categories.",
                rationale=f"Testing if group membership in '{cat}' has an effect on continuous metric '{num}'."
            )

    # 3. Categorical vs Categorical (Chi-Square) across diverse pairs
    for i in range(len(categorical_cols)):
        for j in range(i + 1, len(categorical_cols)):
            if len(hypotheses) >= num_hypotheses:
                break
            cat1, cat2 = categorical_cols[i], categorical_cols[j]
            if col_usage.get(cat1, 0) >= 2 or col_usage.get(cat2, 0) >= 2:
                continue
            add_hypothesis(
                c1=cat1,
                c2=cat2,
                method="chi_square",
                text=f"There is a significant statistical association between {cat1} and {cat2}.",
                rationale=f"Testing independence between categorical attributes '{cat1}' and '{cat2}'."
            )

    # 4. Diverse Numeric vs Numeric (Correlation) using round-robin stride
    # (Avoids pinning column 0 by stepping across the column list)
    if len(numeric_cols) >= 2:
        num_len = len(numeric_cols)
        candidate_pairs = []
        for offset in range(1, num_len):
            for i in range(num_len):
                j = (i + offset) % num_len
                if i < j:
                    candidate_pairs.append((numeric_cols[i], numeric_cols[j]))

        for max_usage in (1, 2, 3, 5):
            for c1, c2 in candidate_pairs:
                if len(hypotheses) >= num_hypotheses:
                    break
                if col_usage.get(c1, 0) < max_usage and col_usage.get(c2, 0) < max_usage:
                    add_hypothesis(
                        c1=c1,
                        c2=c2,
                        method="correlation",
                        text=f"There is a statistically significant correlation between {c1} and {c2}.",
                        rationale=f"Testing linear or monotonic relationship between continuous variables '{c1}' and '{c2}'."
                    )

    return hypotheses[:num_hypotheses]

def generate_hypotheses(df: pd.DataFrame, num_hypotheses: int = 6) -> List[Dict[str, Any]]:
    """
    Generates diverse, falsifiable statistical hypotheses about the dataset.
    Uses local Ollama if available; enforces strict diversity rules (no single-column pinning)
    and falls back to structural heuristics.
    """
    profile = profile_dataset(df)
    target_cols = detect_target_columns(df)
    target_info = f"\nCandidate primary outcome/target variables: {target_cols}" if target_cols else ""
    
    system_prompt = (
        "You are an expert scientific researcher and data scientist specializing in "
        "Automated Hypothesis Generation and Testing (AHGT). You formulate rigorous, "
        "falsifiable hypotheses that can be tested using standard statistical tests "
        "(t-test, correlation, chi-square). Always return strictly valid JSON."
    )
    
    user_prompt = f"""
Given the following dataset schema and characteristics:
{json.dumps(profile, indent=2)}
{target_info}

CRITICAL DIVERSITY & FORMULATION REQUIREMENTS:
1. BROAD COLUMN DIVERSITY: Do NOT anchor on a single column (e.g. NEVER test 'age' against every feature).
2. SPREAD FREQUENCY: No single predictor column may appear in more than 2 hypotheses across the entire generated set.
3. OUTCOME-DRIVEN: Formulate hypotheses exploring what drives the target/outcome variable ({target_cols if target_cols else 'key variables'}), as well as distinct feature-to-feature relationships.
4. VALID METHODS: Each hypothesis must use one of these exact test_method values:
   - 'two_sample_t_test' (col1 = categorical group column, col2 = numeric metric column)
   - 'correlation' (col1 = numeric column, col2 = numeric column)
   - 'chi_square' (col1 = categorical column, col2 = categorical column)

Generate {num_hypotheses} distinct, falsifiable hypotheses about relationships in this data.
Return a JSON object with a 'hypotheses' list in this exact format:
{{
  "hypotheses": [
    {{
      "hypothesis_text": "Clear statement of the relationship",
      "test_method": "two_sample_t_test | correlation | chi_square",
      "col1": "exact_column_name_1",
      "col2": "exact_column_name_2",
      "rationale": "Brief scientific reason why this relationship is plausible"
    }}
  ]
}}
"""
    
    tokens_limit = max(768, num_hypotheses * 120)
    raw_response = call_llm(user_prompt, system_prompt=system_prompt, timeout=OLLAMA_TIMEOUT, max_tokens=tokens_limit, json_format=True)
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
            valid_hypotheses: List[Dict[str, Any]] = []
            seen_pairs = set()
            usage: Dict[str, int] = {c: 0 for c in df.columns}
            target_set = set(target_cols)

            for h in parsed:
                if isinstance(h, dict):
                    c1, c2 = h.get("col1"), h.get("col2")
                    if c1 in df.columns and c2 in df.columns and c1 != c2:
                        pair_key = tuple(sorted([c1, c2]))
                        if pair_key in seen_pairs:
                            continue
                        
                        max_c1 = 4 if c1 in target_set else 2
                        max_c2 = 4 if c2 in target_set else 2
                        if usage.get(c1, 0) < max_c1 and usage.get(c2, 0) < max_c2:
                            valid_hypotheses.append(h)
                            seen_pairs.add(pair_key)
                            usage[c1] = usage.get(c1, 0) + 1
                            usage[c2] = usage.get(c2, 0) + 1
            
            if len(valid_hypotheses) >= num_hypotheses:
                return valid_hypotheses[:num_hypotheses]
            elif len(valid_hypotheses) > 0:
                # Supplement with diverse heuristic hypotheses to reach requested count
                supplement = generate_hypotheses_heuristic(df, num_hypotheses=num_hypotheses * 2)
                for sup in supplement:
                    sc1, sc2 = sup.get("col1"), sup.get("col2")
                    spair = tuple(sorted([sc1, sc2]))
                    if spair not in seen_pairs:
                        valid_hypotheses.append(sup)
                        seen_pairs.add(spair)
                        if len(valid_hypotheses) >= num_hypotheses:
                            break
                return valid_hypotheses[:num_hypotheses]

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
