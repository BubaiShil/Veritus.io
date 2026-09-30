"""
Veritus — Master Pipeline Orchestrator (backend/main.py)

Orchestrates the end-to-end Automated Hypothesis Generation and Testing (AHGT) loop:
  1. Registers uploaded dataset in PostgreSQL (backend/db.py).
  2. Profiles dataset & prompts LLM to generate 5-10 hypotheses (backend/ai_agent.py).
  3. Tests each hypothesis using frequentist statistics (backend/stats_engine.py).
  4. Branches on test verdict:
     - If VALIDATED: generates plain-English explanation & logs finding.
     - If REJECTED: triggers self-correcting refinement loop to formulate
       a more specific child hypothesis, re-tests, and links parent-child lineage.
  5. Saves complete reasoning lineage to PostgreSQL.
  6. Returns structured report for the Streamlit UI (frontend/app.py).
"""

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd

# Support running directly or as a module
backend_dir = Path(__file__).resolve().parent
project_root = backend_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.db import init_db, create_dataset, save_hypothesis, get_hypotheses_by_dataset
from backend.stats_engine import test_hypothesis
from backend.ai_agent import generate_hypotheses, refine_hypothesis, explain_findings

def run_analysis_pipeline(
    df: pd.DataFrame,
    dataset_name: str,
    num_initial_hypotheses: int = 6,
    enable_refinement: bool = True,
    progress_callback: Optional[Callable[[str, float], None]] = None
) -> Dict[str, Any]:
    """
    Executes the complete autonomous scientific discovery loop on a dataframe.
    
    Args:
        df: The pandas DataFrame to analyze.
        dataset_name: Name of the uploaded file or dataset.
        num_initial_hypotheses: Number of initial hypotheses to formulate.
        enable_refinement: Whether to trigger iterative refinement for failed tests.
        progress_callback: Optional callback func(status_message, progress_float_0_to_1).
        
    Returns:
        Dict containing dataset_id, validated findings, rejected hypotheses,
        refinement lineage, and overall summary stats.
    """
    def update_progress(msg: str, frac: float):
        if progress_callback:
            progress_callback(msg, frac)
        else:
            print(f"[{int(frac * 100)}%] {msg}")

    update_progress("Initializing database schema...", 0.05)
    init_db()
    
    update_progress(f"Registering dataset '{dataset_name}' in database...", 0.10)
    dataset_id = create_dataset(dataset_name)
    
    update_progress("AI Agent formulating hypotheses from schema...", 0.20)
    initial_hypotheses = generate_hypotheses(df, num_hypotheses=num_initial_hypotheses)
    
    if not initial_hypotheses:
        update_progress("No testable hypotheses could be generated.", 1.0)
        return {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "validated": [],
            "rejected": [],
            "all_hypotheses": []
        }
        
    total_items = len(initial_hypotheses)
    validated_list: List[Dict[str, Any]] = []
    rejected_list: List[Dict[str, Any]] = []
    all_results: List[Dict[str, Any]] = []
    
    for idx, hypo in enumerate(initial_hypotheses):
        progress_pct = 0.25 + (0.65 * (idx / total_items))
        hypo_text = hypo.get("hypothesis_text", "")
        test_method = hypo.get("test_method", "correlation")
        col1 = hypo.get("col1", "")
        col2 = hypo.get("col2", "")
        
        update_progress(f"Testing hypothesis {idx + 1}/{total_items}: '{hypo_text}'...", progress_pct)
        
        # 1. Run statistical test
        test_res = test_hypothesis(df, test_type=test_method, col1=col1, col2=col2)
        verdict = test_res.get("verdict", "rejected")
        p_val = test_res.get("p_value")
        effect_sz = test_res.get("effect_size")
        
        # 2. Branch based on verdict
        if verdict == "validated":
            explanation = explain_findings(hypo_text, test_res)
            
            # Save to PostgreSQL
            h_id = save_hypothesis(
                dataset_id=dataset_id,
                hypothesis_text=hypo_text,
                test_method=test_res.get("test_method"),
                p_value=p_val,
                effect_size=effect_sz,
                verdict="validated",
                explanation=explanation,
                parent_hypothesis_id=None
            )
            
            entry = {
                "id": h_id,
                "hypothesis_text": hypo_text,
                "test_method": test_res.get("test_method"),
                "col1": col1,
                "col2": col2,
                "p_value": p_val,
                "effect_size": effect_sz,
                "effect_size_name": test_res.get("effect_size_name", ""),
                "verdict": "validated",
                "explanation": explanation,
                "summary": test_res.get("summary", ""),
                "parent_id": None,
                "is_refined": False
            }
            validated_list.append(entry)
            all_results.append(entry)
            
        else:
            # Save rejected hypothesis
            h_id = save_hypothesis(
                dataset_id=dataset_id,
                hypothesis_text=hypo_text,
                test_method=test_res.get("test_method"),
                p_value=p_val,
                effect_size=effect_sz,
                verdict="rejected",
                explanation=test_res.get("summary", "Failed to reach significance threshold (p >= 0.05)."),
                parent_hypothesis_id=None
            )
            
            entry = {
                "id": h_id,
                "hypothesis_text": hypo_text,
                "test_method": test_res.get("test_method"),
                "col1": col1,
                "col2": col2,
                "p_value": p_val,
                "effect_size": effect_sz,
                "effect_size_name": test_res.get("effect_size_name", ""),
                "verdict": "rejected",
                "explanation": test_res.get("summary", "Statistical test failed to reject null hypothesis."),
                "summary": test_res.get("summary", ""),
                "parent_id": None,
                "is_refined": False
            }
            rejected_list.append(entry)
            all_results.append(entry)
            
            # 3. Refinement Branch (Self-Correction)
            if enable_refinement:
                update_progress(f"Refining rejected hypothesis: '{hypo_text}'...", progress_pct + 0.02)
                refined = refine_hypothesis(df, failed_hypothesis=hypo, test_result=test_res)
                
                if refined and refined.get("col1") in df.columns and refined.get("col2") in df.columns:
                    ref_text = refined.get("hypothesis_text", "")
                    ref_method = refined.get("test_method", "correlation")
                    ref_c1, ref_c2 = refined.get("col1"), refined.get("col2")
                    
                    ref_test_res = test_hypothesis(df, test_type=ref_method, col1=ref_c1, col2=ref_c2)
                    ref_verdict = ref_test_res.get("verdict", "rejected")
                    ref_p_val = ref_test_res.get("p_value")
                    ref_effect_sz = ref_test_res.get("effect_size")
                    
                    ref_explanation = (
                        explain_findings(ref_text, ref_test_res)
                        if ref_verdict == "validated"
                        else ref_test_res.get("summary", "Refined hypothesis also failed statistical testing.")
                    )
                    
                    ref_h_id = save_hypothesis(
                        dataset_id=dataset_id,
                        hypothesis_text=ref_text,
                        test_method=ref_test_res.get("test_method"),
                        p_value=ref_p_val,
                        effect_size=ref_effect_sz,
                        verdict=ref_verdict,
                        explanation=ref_explanation,
                        parent_hypothesis_id=h_id  # Provenance link!
                    )
                    
                    ref_entry = {
                        "id": ref_h_id,
                        "hypothesis_text": ref_text,
                        "test_method": ref_test_res.get("test_method"),
                        "col1": ref_c1,
                        "col2": ref_c2,
                        "p_value": ref_p_val,
                        "effect_size": ref_effect_sz,
                        "effect_size_name": ref_test_res.get("effect_size_name", ""),
                        "verdict": ref_verdict,
                        "explanation": ref_explanation,
                        "summary": ref_test_res.get("summary", ""),
                        "parent_id": h_id,
                        "is_refined": True
                    }
                    
                    all_results.append(ref_entry)
                    if ref_verdict == "validated":
                        validated_list.append(ref_entry)
                    else:
                        rejected_list.append(ref_entry)

    update_progress("Analysis pipeline completed successfully!", 1.0)
    
    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "total_hypotheses": len(all_results),
        "validated_count": len(validated_list),
        "rejected_count": len(rejected_list),
        "validated": validated_list,
        "rejected": rejected_list,
        "all_hypotheses": all_results
    }

if __name__ == "__main__":
    print("=== Testing Veritus Master Orchestration Loop ===")
    
    # Generate realistic multi-column synthetic business dataset
    np.random.seed(101)
    n_samples = 150
    
    # Marketing spend strongly correlates with Sales Volume
    ad_spend = np.random.uniform(50, 500, n_samples)
    sales = 3.2 * ad_spend + np.random.normal(0, 100, n_samples)
    
    # Pure random noise column (designed to test rejection)
    random_metric = np.random.normal(100, 20, n_samples)
    
    # Customer Segment binary split (Premium vs Standard)
    segment = np.random.choice(["Premium", "Standard"], size=n_samples, p=[0.45, 0.55])
    # Premium customers have significantly higher Customer Satisfaction
    satisfaction = np.where(segment == "Premium", np.random.normal(8.8, 0.8, n_samples), np.random.normal(6.5, 1.1, n_samples))
    
    # Device vs churn categorical
    device = np.random.choice(["Desktop", "Mobile"], size=n_samples)
    churn = np.where(device == "Mobile", np.random.choice(["Yes", "No"], size=n_samples, p=[0.6, 0.4]),
                     np.random.choice(["Yes", "No"], size=n_samples, p=[0.25, 0.75]))
    
    demo_df = pd.DataFrame({
        "ad_spend": ad_spend,
        "sales_volume": sales,
        "random_noise": random_metric,
        "customer_segment": segment,
        "satisfaction_score": satisfaction,
        "device_type": device,
        "churn_status": churn
    })
    
    # Allow passing custom CSV path via command line
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        csv_path = Path(sys.argv[1])
        demo_df = pd.read_csv(csv_path)
        ds_name = csv_path.name
        print(f"Loaded custom dataset: {ds_name} ({len(demo_df)} rows)")
    else:
        ds_name = "synthetic_ecommerce_demo.csv"
        print(f"Using synthetic dataset: {ds_name} ({len(demo_df)} rows)")
        
    results = run_analysis_pipeline(demo_df, dataset_name=ds_name, num_initial_hypotheses=5)
    
    print("\n=======================================================")
    print(f"  ANALYSIS COMPLETE FOR DATASET ID: {results['dataset_id']}")
    print(f"  Total Hypotheses Tested: {results['total_hypotheses']}")
    print(f"  Validated Discoveries:   {results['validated_count']}")
    print(f"  Rejected Hypotheses:     {results['rejected_count']}")
    print("=======================================================\n")
    
    print("--- [VALIDATED FINDINGS] ---")
    for item in results["validated"]:
        refined_tag = " [REFINED CHILD]" if item.get("is_refined") else ""
        print(f"  * {item['hypothesis_text']}{refined_tag}")
        print(f"    Method: {item['test_method']} | p-val: {item['p_value']:.4g} | effect: {item['effect_size']:.3f} ({item.get('effect_size_name')})")
        print(f"    Explanation: {item['explanation']}\n")
        
    print("--- [REJECTED HYPOTHESES (Graveyard)] ---")
    for item in results["rejected"]:
        print(f"  x {item['hypothesis_text']}")
        print(f"    Method: {item['test_method']} | p-val: {item['p_value']} | verdict: REJECTED\n")
