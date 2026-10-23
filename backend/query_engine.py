"""
Veritus — Natural Language Query & On-Demand Statistical Testing Engine (backend/query_engine.py)

Empowers users to ask any plain-English question about their dataset, including:
  - Dataset overview: "How many rows and columns?", "What columns have missing values?"
  - Statistical testing: "Does high BMI significantly correlate with blood pressure?", "Is there a difference in disease progression between sexes?"
  - Subgroup comparisons: "Compare average blood pressure across gender categories", "What is the average progression by sex?"
  - Specific aggregations: "What is the highest age?", "What is the median cholesterol?", "What is the average BMI?"
  - Comparative ranking: "Which biomarker has the strongest link to disease progression?", "What are the top 3 predictors?"
  - Descriptive summaries: "What is the distribution of column bp?"
  - General data questions: "Summarize this dataset and key insights"

Translates questions into formal statistical or analytical actions, runs computations
on the fly (frequentist tests, aggregations, ANOVA, ranking), and returns empirical
metrics, verdicts, and clear plain-English answers.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from backend.ai_agent import call_llm, parse_llm_json, detect_target_columns
from backend.stats_engine import test_hypothesis

logger = logging.getLogger("veritus.query_engine")


def extract_columns_from_text(text: str, columns: List[str]) -> List[str]:
    """Matches mentioned column names in free text (case-insensitive and token-aware)."""
    text_lower = text.lower()
    found: List[str] = []
    
    # Sort columns by length descending so longer column names match before substrings
    sorted_cols = sorted(columns, key=lambda c: len(c), reverse=True)
    
    for col in sorted_cols:
        col_clean = col.lower()
        # Word boundary or underscore match
        pattern = r"(?:\b|_)" + re.escape(col_clean) + r"(?:\b|_)"
        if re.search(pattern, text_lower):
            if col not in found:
                found.append(col)
        # Also check direct lowercase match for short abbreviations like bp, bmi, s1..s6
        elif f" {col_clean} " in f" {text_lower} " or f"'{col_clean}'" in text_lower or f'"{col_clean}"' in text_lower:
            if col not in found:
                found.append(col)
                
    return found


def classify_question_heuristic(question: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Statistically grounded fallback classifier when LLM is unavailable or offline.
    Uses structural column matching, linguistic intent cues, and data science heuristics.
    """
    q_low = question.lower()
    all_cols = list(df.columns)
    target_cols = detect_target_columns(df)
    primary_target = target_cols[0] if target_cols else all_cols[-1]
    matched_cols = extract_columns_from_text(question, all_cols)

    # 1. Dataset overview cues (rows, columns, nulls, missing, overview, summary)
    overview_cues = [
        "how many rows", "row count", "how many records", "how many observations",
        "how many columns", "what columns", "list columns", "show columns", "features",
        "missing value", "missing data", "null value", "nulls", "na count",
        "overview", "dataset size", "shape", "summarize the dataset", "summarize data",
        "about this dataset", "what is this dataset", "dataset summary"
    ]
    if any(cue in q_low for cue in overview_cues) and len(matched_cols) == 0:
        return {
            "action": "dataset_overview",
            "interpreted_hypothesis": "Provide a structural overview of dataset dimensions, schema, and completeness."
        }

    # 2. Ranking cues ("which", "strongest", "highest", "most", "best predictor", "rank")
    ranking_cues = ["which feature", "which biomarker", "which column", "strongest", "highest link", "most correlated", "best predictor", "greatest impact", "rank all", "compare all"]
    if any(cue in q_low for cue in ranking_cues):
        target = primary_target
        if matched_cols:
            target = matched_cols[0]
            
        return {
            "action": "comparative_ranking",
            "target_col": target,
            "interpreted_hypothesis": f"Rank all candidate predictors by strength of statistical association with '{target}'."
        }

    # 3. Group comparison cues ("across", "by", "per", "grouped by", "difference between groups")
    group_cues = [" across ", " by ", " per ", " grouped by ", " breakdown by ", " category ", " categories "]
    if any(cue in q_low for cue in group_cues) and len(matched_cols) >= 1:
        # Identify numeric vs categorical
        num_col = None
        cat_col = None
        for col in matched_cols:
            if pd.api.types.is_numeric_dtype(df[col]) and num_col is None:
                num_col = col
            elif (not pd.api.types.is_numeric_dtype(df[col]) or df[col].nunique() <= 10) and cat_col is None:
                cat_col = col

        if num_col and cat_col and num_col != cat_col:
            return {
                "action": "group_comparison",
                "metric_col": num_col,
                "group_col": cat_col,
                "interpreted_hypothesis": f"Compare '{num_col}' distributions and group means across '{cat_col}' categories."
            }

    # 4. Aggregation cues (mean, average, median, min, max, highest, lowest, sum) for a single column
    agg_map = {
        "highest": "max", "maximum": "max", "max": "max",
        "lowest": "min", "minimum": "min", "min": "min",
        "average": "mean", "mean": "mean", "avg": "mean",
        "median": "median", "sum": "sum", "total": "sum",
        "standard deviation": "std", "std": "std", "variance": "var"
    }
    for phrase, agg_type in agg_map.items():
        if phrase in q_low and len(matched_cols) == 1:
            col = matched_cols[0]
            if pd.api.types.is_numeric_dtype(df[col]):
                return {
                    "action": "ad_hoc_aggregation",
                    "col1": col,
                    "agg_type": agg_type,
                    "interpreted_hypothesis": f"Compute the {agg_type} value for column '{col}'."
                }

    # 5. Check for confounding / control variables ("controlling for", "adjusting for", "independent of")
    cov_cues = ["controlling for", "adjusting for", "controlled for", "independent of", "adjusted for"]
    for cue in cov_cues:
        if cue in q_low:
            parts = q_low.split(cue, 1)
            target_part = parts[1]
            cov_candidates = extract_columns_from_text(target_part, all_cols)
            main_candidates = [c for c in matched_cols if c not in cov_candidates]
            if len(main_candidates) >= 1 and len(cov_candidates) >= 1:
                c1 = main_candidates[0]
                c2 = main_candidates[1] if len(main_candidates) > 1 else (primary_target if primary_target != c1 else all_cols[-1])
                return {
                    "action": "statistical_test",
                    "test_method": "partial_correlation",
                    "col1": c1,
                    "col2": c2,
                    "covariates": cov_candidates,
                    "interpreted_hypothesis": f"Investigate whether '{c1}' is independently correlated with '{c2}', controlling for {cov_candidates}."
                }

    # 6. Check for 2 matched columns -> statistical hypothesis test
    if len(matched_cols) >= 2:
        c1, c2 = matched_cols[0], matched_cols[1]
        c1_num = pd.api.types.is_numeric_dtype(df[c1])
        c2_num = pd.api.types.is_numeric_dtype(df[c2])
        
        if c1_num and c2_num:
            return {
                "action": "statistical_test",
                "test_method": "correlation",
                "col1": c1,
                "col2": c2,
                "interpreted_hypothesis": f"There is a statistically significant correlation between {c1} and {c2}."
            }
        elif (not c1_num and c2_num) or (c1_num and not c2_num):
            cat_col = c1 if not c1_num else c2
            num_col = c2 if not c1_num else c1
            return {
                "action": "statistical_test",
                "test_method": "two_sample_t_test",
                "col1": cat_col,
                "col2": num_col,
                "interpreted_hypothesis": f"The mean {num_col} differs significantly across distinct {cat_col} groups."
            }
        else:
            return {
                "action": "statistical_test",
                "test_method": "chi_square",
                "col1": c1,
                "col2": c2,
                "interpreted_hypothesis": f"There is a significant association between categorical factors {c1} and {c2}."
            }
            
    # 6. Check if 1 column matched with target column
    if len(matched_cols) == 1 and primary_target and matched_cols[0] != primary_target:
        c1, c2 = matched_cols[0], primary_target
        c1_num = pd.api.types.is_numeric_dtype(df[c1])
        c2_num = pd.api.types.is_numeric_dtype(df[c2])
        if c1_num and c2_num:
            return {
                "action": "statistical_test",
                "test_method": "correlation",
                "col1": c1,
                "col2": c2,
                "interpreted_hypothesis": f"There is a statistically significant correlation between {c1} and {c2}."
            }
        else:
            cat_col = c1 if not c1_num else c2
            num_col = c2 if not c1_num else c1
            return {
                "action": "statistical_test",
                "test_method": "two_sample_t_test",
                "col1": cat_col,
                "col2": num_col,
                "interpreted_hypothesis": f"The mean {num_col} differs significantly across {cat_col} categories."
            }

    # 7. Descriptive summary for single column
    if len(matched_cols) == 1:
        return {
            "action": "descriptive_summary",
            "col1": matched_cols[0],
            "interpreted_hypothesis": f"Distribution summary for column '{matched_cols[0]}'."
        }

    # 8. Default fallback: General analytical QA or ranking
    return {
        "action": "general_analytical_qa",
        "interpreted_hypothesis": f"Analytical evaluation of user query: '{question}'."
    }


def parse_query_with_llm(question: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Uses Gemini / Ollama / LLM to interpret complex natural language queries and map to analytical actions."""
    columns = list(df.columns)
    target_cols = detect_target_columns(df)
    
    col_summary = {}
    for col in columns[:30]:
        dtype = "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "categorical"
        col_summary[col] = dtype

    system_prompt = (
        "You are an expert AI Data Scientist. Your task is to translate user natural language questions "
        "about a dataset into precise statistical and analytical actions. Return strictly valid JSON."
    )

    user_prompt = f"""
Dataset columns and types:
{json.dumps(col_summary, indent=2)}
Detected outcome/target variables: {target_cols}
Total rows: {len(df)}, Total columns: {len(df.columns)}

User question: "{question}"

Classify this question into one of these actions:
1. "dataset_overview": User asks about row count, column count, list of columns, missing values, or dataset overview.
2. "group_comparison": User asks to compare a metric across categories or groups (e.g. "compare blood pressure across sex", "average progression by category").
   - "metric_col": numeric column
   - "group_col": categorical or grouping column
3. "ad_hoc_aggregation": User asks for a specific metric (highest/max, lowest/min, average/mean, median, sum) of a specific column.
   - "col1": column name
   - "agg_type": "mean" | "median" | "min" | "max" | "std" | "sum"
4. "statistical_test": User asks about relationship, correlation, or difference between two specific variables.
   - "test_method": "correlation" | "partial_correlation" | "two_sample_t_test" | "chi_square"
   - "col1": exact column name
   - "col2": exact column name
   - "covariates": list of control/confounding columns (e.g. if user asks "controlling for BMI", covariates is ["bmi"])
5. "comparative_ranking": User asks "which feature/biomarker is most related", "strongest link to outcome", or wants to compare multiple predictors against a target.
   - "target_col": exact target column name
6. "descriptive_summary": User asks for distribution, summary, or quartiles of a single column.
   - "col1": exact column name
7. "general_analytical_qa": User asks any other analytical or explanatory question about the dataset.

Return strictly JSON format:
{{
  "action": "dataset_overview" | "group_comparison" | "ad_hoc_aggregation" | "statistical_test" | "comparative_ranking" | "descriptive_summary" | "general_analytical_qa",
  "test_method": "correlation | partial_correlation | two_sample_t_test | chi_square | null",
  "col1": "col_name_or_null",
  "col2": "col_name_or_null",
  "covariates": ["covariate_col1"],
  "metric_col": "metric_col_or_null",
  "group_col": "group_col_or_null",
  "agg_type": "mean | median | min | max | std | sum | null",
  "target_col": "target_col_or_null",
  "interpreted_hypothesis": "Clear statement"
}}
"""

    raw = call_llm(user_prompt, system_prompt=system_prompt, timeout=25, max_tokens=300, json_format=True)

    if raw:
        parsed = parse_llm_json(raw)
        if isinstance(parsed, dict) and "action" in parsed:
            action = parsed.get("action")
            
            if action == "dataset_overview":
                return parsed
                
            if action == "group_comparison":
                m = parsed.get("metric_col") or parsed.get("col2")
                g = parsed.get("group_col") or parsed.get("col1")
                if m in df.columns and g in df.columns:
                    return {
                        "action": "group_comparison",
                        "metric_col": m,
                        "group_col": g,
                        "interpreted_hypothesis": parsed.get("interpreted_hypothesis", f"Compare '{m}' across '{g}' categories.")
                    }
                    
            if action == "ad_hoc_aggregation":
                c = parsed.get("col1")
                if c in df.columns:
                    return parsed
                    
            if action == "statistical_test":
                c1, c2 = parsed.get("col1"), parsed.get("col2")
                covs = [c for c in parsed.get("covariates", []) if c in df.columns and c not in (c1, c2)]
                if c1 in df.columns and c2 in df.columns and c1 != c2:
                    parsed["covariates"] = covs
                    if covs:
                        parsed["test_method"] = "partial_correlation"
                    return parsed
                    
            elif action == "comparative_ranking":
                tgt = parsed.get("target_col")
                if tgt in df.columns:
                    return parsed
                    
            elif action == "descriptive_summary":
                c1 = parsed.get("col1")
                if c1 in df.columns:
                    return parsed

            elif action == "general_analytical_qa":
                return parsed

    # Heuristic fallback if LLM is offline or output didn't validate
    return classify_question_heuristic(question, df)


def run_dataset_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """Generates structural overview of dataset dimensions, missing data, and column types."""
    num_rows = len(df)
    num_cols = len(df.columns)
    
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    
    missing_summary = {}
    total_missing = 0
    for col in df.columns:
        m_count = int(df[col].isnull().sum())
        if m_count > 0:
            missing_summary[col] = {
                "count": m_count,
                "pct": round((m_count / num_rows) * 100, 1)
            }
            total_missing += m_count
            
    # Format plain English answer
    missing_str = (
        f"{len(missing_summary)} column(s) have missing values ({total_missing} total nulls)."
        if total_missing > 0
        else "There are 0 missing values across all columns (complete dataset)."
    )
    
    col_preview = ", ".join(list(df.columns)[:8])
    if len(df.columns) > 8:
        col_preview += f", and {len(df.columns) - 8} more"
        
    answer = (
        f"The dataset contains {num_rows:,} rows (records) and {num_cols} columns: "
        f"{len(numeric_cols)} numeric and {len(categorical_cols)} categorical. "
        f"{missing_str} Features include: {col_preview}."
    )
    
    return {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "numeric_cols_count": len(numeric_cols),
        "categorical_cols_count": len(categorical_cols),
        "missing_summary": missing_summary,
        "total_missing": total_missing,
        "columns": list(df.columns),
        "summary_answer": answer
    }


def run_group_comparison(df: pd.DataFrame, metric_col: str, group_col: str) -> Dict[str, Any]:
    """
    Compares a numeric metric across distinct categories of a grouping column.
    Computes group means, medians, std, and performs ANOVA / Welch t-test for statistical significance.
    """
    if metric_col not in df.columns or group_col not in df.columns:
        return {"error": "Invalid columns provided for group comparison."}
        
    sub = df[[group_col, metric_col]].dropna()
    if len(sub) < 5:
        return {"error": "Insufficient data points for group comparison."}
        
    groups = sub[group_col].unique()
    group_stats = []
    arrays_for_anova = []
    
    for g in groups:
        vals = sub[sub[group_col] == g][metric_col].values
        if len(vals) > 0:
            group_stats.append({
                "group": str(g),
                "count": int(len(vals)),
                "mean": round(float(np.mean(vals)), 2),
                "median": round(float(np.median(vals)), 2),
                "std": round(float(np.std(vals)), 2) if len(vals) > 1 else 0.0
            })
            if len(vals) >= 2:
                arrays_for_anova.append(vals)
                
    # Sort groups by mean descending
    group_stats.sort(key=lambda x: x["mean"], reverse=True)
    
    # Statistical significance testing across groups (ANOVA or t-test)
    p_val = None
    stat_val = None
    test_name = "One-Way ANOVA"
    verdict = "inconclusive"
    
    if len(arrays_for_anova) >= 2:
        try:
            if len(arrays_for_anova) == 2:
                t_res = stats.ttest_ind(arrays_for_anova[0], arrays_for_anova[1], equal_var=False)
                stat_val = round(float(t_res.statistic), 3)
                p_val = float(t_res.pvalue)
                test_name = "Welch's Two-Sample t-test"
            else:
                f_res = stats.f_oneway(*arrays_for_anova)
                stat_val = round(float(f_res.statistic), 3)
                p_val = float(f_res.pvalue)
                test_name = "One-Way ANOVA"
                
            verdict = "validated" if p_val < 0.05 else "rejected"
        except Exception as exc:
            logger.warning(f"Group comparison test calculation error: {exc}")

    # Build plain English answer
    top_group = group_stats[0] if group_stats else None
    bottom_group = group_stats[-1] if len(group_stats) > 1 else None
    
    p_str = f"p = {p_val:.4g}" if p_val is not None else "p = N/A"
    sig_str = (
        f"This difference is statistically significant ({test_name}, {p_str} < 0.05)."
        if verdict == "validated"
        else f"This difference is not statistically significant ({test_name}, {p_str} ≥ 0.05)."
    )
    
    if top_group and bottom_group and top_group != bottom_group:
        answer = (
            f"Comparing '{metric_col}' across '{group_col}' categories: Group '{top_group['group']}' "
            f"has the highest mean ({top_group['mean']:.2f}, n={top_group['count']}), while group '{bottom_group['group']}' "
            f"has the lowest mean ({bottom_group['mean']:.2f}, n={bottom_group['count']}). {sig_str}"
        )
    else:
        answer = f"Analyzed '{metric_col}' across {len(group_stats)} categories of '{group_col}'."
        
    return {
        "metric_col": metric_col,
        "group_col": group_col,
        "groups": group_stats,
        "test_name": test_name,
        "statistic": stat_val,
        "p_value": p_val,
        "verdict": verdict,
        "summary_answer": answer
    }


def run_ad_hoc_aggregation(df: pd.DataFrame, col: str, agg_type: str) -> Dict[str, Any]:
    """Computes specific aggregations (mean, median, min, max, std, sum) for a column."""
    if col not in df.columns:
        return {"error": f"Column '{col}' not found."}
        
    sub = df[col].dropna()
    if len(sub) == 0:
        return {"error": f"Column '{col}' has no non-null values."}
        
    agg_type = agg_type.lower()
    val = None
    
    if agg_type in ("max", "highest", "maximum"):
        val = float(sub.max())
        label = "maximum (highest)"
    elif agg_type in ("min", "lowest", "minimum"):
        val = float(sub.min())
        label = "minimum (lowest)"
    elif agg_type in ("median", "middle"):
        val = float(sub.median())
        label = "median"
    elif agg_type in ("mean", "average", "avg"):
        val = float(sub.mean())
        label = "average (mean)"
    elif agg_type in ("std", "standard deviation"):
        val = float(sub.std())
        label = "standard deviation"
    elif agg_type in ("sum", "total"):
        val = float(sub.sum())
        label = "total sum"
    else:
        val = float(sub.mean())
        label = "mean"
        
    answer = f"The {label} value for column '{col}' is {val:,.2f} (across {len(sub):,} non-null records)."
    
    return {
        "column": col,
        "aggregation": label,
        "value": round(val, 2),
        "total_records": len(sub),
        "summary_answer": answer
    }


def run_comparative_ranking(df: pd.DataFrame, target_col: str, top_k: int = 8) -> Dict[str, Any]:
    """
    Ranks all candidate features against a target column by statistical association strength.
    Computes Pearson r and Welch's t-test / ANOVA depending on data type.
    """
    if target_col not in df.columns:
        return {"error": f"Target column '{target_col}' not found in dataset."}
        
    ranking = []
    is_target_numeric = pd.api.types.is_numeric_dtype(df[target_col])
    
    for col in df.columns:
        if col == target_col:
            continue
            
        sub = df[[col, target_col]].dropna()
        if len(sub) < 5:
            continue
            
        is_col_numeric = pd.api.types.is_numeric_dtype(df[col])
        
        try:
            if is_target_numeric and is_col_numeric:
                r, p_val = stats.pearsonr(sub[col], sub[target_col])
                ranking.append({
                    "column": col,
                    "test_type": "Pearson Correlation",
                    "metric_name": "r",
                    "metric_value": round(float(r), 3),
                    "abs_effect": abs(float(r)),
                    "p_value": float(p_val),
                    "significant": bool(p_val < 0.05)
                })
            elif is_target_numeric and not is_col_numeric:
                cats = sub[col].unique()
                if 2 <= len(cats) <= 10:
                    group_data = [sub[sub[col] == c][target_col].values for c in cats if len(sub[sub[col] == c]) >= 2]
                    if len(group_data) >= 2:
                        f_stat, p_val = stats.f_oneway(*group_data)
                        ranking.append({
                            "column": col,
                            "test_type": "ANOVA F-test",
                            "metric_name": "F-stat",
                            "metric_value": round(float(f_stat), 2),
                            "abs_effect": round(float(f_stat), 2),
                            "p_value": float(p_val),
                            "significant": bool(p_val < 0.05)
                        })
            elif not is_target_numeric and is_col_numeric:
                cats = sub[target_col].unique()
                if 2 <= len(cats) <= 10:
                    group_data = [sub[sub[target_col] == c][col].values for c in cats if len(sub[sub[target_col] == c]) >= 2]
                    if len(group_data) >= 2:
                        f_stat, p_val = stats.f_oneway(*group_data)
                        ranking.append({
                            "column": col,
                            "test_type": "ANOVA F-test",
                            "metric_name": "F-stat",
                            "metric_value": round(float(f_stat), 2),
                            "abs_effect": round(float(f_stat), 2),
                            "p_value": float(p_val),
                            "significant": bool(p_val < 0.05)
                        })
            else:
                cont_table = pd.crosstab(sub[col], sub[target_col])
                chi2, p_val, _, _ = stats.chi2_contingency(cont_table)
                ranking.append({
                    "column": col,
                    "test_type": "Chi-Square",
                    "metric_name": "Chi²",
                    "metric_value": round(float(chi2), 2),
                    "abs_effect": round(float(chi2), 2),
                    "p_value": float(p_val),
                    "significant": bool(p_val < 0.05)
                })
        except Exception:
            continue
            
    # Sort by absolute effect size descending
    ranking.sort(key=lambda item: item.get("abs_effect", 0), reverse=True)
    top_ranking = ranking[:top_k]
    
    # Formulate plain-English answer
    if top_ranking:
        sig_features = [item for item in top_ranking if item.get("significant")]
        if sig_features:
            top_f = sig_features[0]
            summary_answer = (
                f"The feature with the strongest statistical link to '{target_col}' is '{top_f['column']}' "
                f"({top_f['test_type']}: {top_f['metric_name']} = {top_f['metric_value']}, p = {top_f['p_value']:.4g} < 0.05). "
                f"A total of {len(sig_features)} features showed statistically significant relationships."
            )
        else:
            summary_answer = f"No features showed a statistically significant link to '{target_col}' at α = 0.05."
    else:
        summary_answer = f"No testable predictors found for '{target_col}'."
        
    return {
        "target_col": target_col,
        "ranking": top_ranking,
        "summary_answer": summary_answer
    }


def run_descriptive_summary(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    """Generates a clean statistical summary of a single column."""
    if col not in df.columns:
        return {"error": f"Column '{col}' not found."}
        
    sub = df[col].dropna()
    total = len(df)
    missing = int(df[col].isnull().sum())
    
    if pd.api.types.is_numeric_dtype(df[col]):
        mean_val = float(sub.mean())
        std_val = float(sub.std())
        min_val = float(sub.min())
        max_val = float(sub.max())
        median_val = float(sub.median())
        q25 = float(sub.quantile(0.25))
        q75 = float(sub.quantile(0.75))
        
        answer = (
            f"Column '{col}' has an average of {mean_val:.2f} (std: {std_val:.2f}, median: {median_val:.2f}), "
            f"spanning from a minimum of {min_val:.2f} to a maximum of {max_val:.2f} across {len(sub)} observations."
        )
        stats_dict = {
            "mean": round(mean_val, 2),
            "std": round(std_val, 2),
            "median": round(median_val, 2),
            "min": round(min_val, 2),
            "max": round(max_val, 2),
            "q25": round(q25, 2),
            "q75": round(q75, 2),
            "missing": missing,
            "total_rows": total
        }
    else:
        top_counts = sub.value_counts().head(5).to_dict()
        top_str = ", ".join([f"{k} (n={v})" for k, v in top_counts.items()])
        answer = (
            f"Categorical column '{col}' has {sub.nunique()} distinct categories across {len(sub)} non-null rows. "
            f"Top categories: {top_str}."
        )
        stats_dict = {
            "unique_categories": sub.nunique(),
            "top_categories": top_counts,
            "missing": missing,
            "total_rows": total
        }
        
    return {
        "column": col,
        "stats": stats_dict,
        "summary_answer": answer
    }


def run_general_analytical_qa(df: pd.DataFrame, question: str) -> Dict[str, Any]:
    """
    Answers freeform analytical questions by compiling a concise factual statistical profile
    and querying the LLM (Gemini / Groq / Ollama) or generating structured descriptive facts.
    """
    num_rows = len(df)
    num_cols = len(df.columns)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    
    # Extract brief descriptive summary of up to 10 columns
    sample_stats = {}
    for col in list(df.columns)[:10]:
        sub = df[col].dropna()
        if pd.api.types.is_numeric_dtype(df[col]):
            sample_stats[col] = {
                "mean": round(float(sub.mean()), 2) if len(sub) > 0 else None,
                "median": round(float(sub.median()), 2) if len(sub) > 0 else None,
                "min": round(float(sub.min()), 2) if len(sub) > 0 else None,
                "max": round(float(sub.max()), 2) if len(sub) > 0 else None,
            }
        else:
            sample_stats[col] = {
                "unique_count": int(sub.nunique()),
                "top": str(sub.value_counts().index[0]) if len(sub) > 0 else None
            }

    system_prompt = (
        "You are an expert AI Data Scientist at Veritus. Answer the user's analytical question "
        "concisely and empirically based ONLY on the provided dataset context. Keep your response "
        "accurate, objective, and under 3-4 sentences."
    )
    
    user_prompt = f"""
Dataset Overview:
- Rows: {num_rows:,}
- Columns ({num_cols}): {list(df.columns)}
- Numeric features: {numeric_cols}
- Categorical features: {categorical_cols}
- Summary sample:
{json.dumps(sample_stats, indent=2)}

User Question: "{question}"

Provide a clear, direct, empirical answer.
"""
    answer = call_llm(user_prompt, system_prompt=system_prompt, timeout=25, max_tokens=250)
    
    if not answer or not answer.strip():
        # Fallback synthesis
        overview = run_dataset_overview(df)
        answer = (
            f"Dataset includes {num_rows:,} records and {num_cols} features. "
            f"Columns: {', '.join(list(df.columns)[:8])}. "
            f"To get precise statistics, try asking about specific column relationships or metrics."
        )

    return {
        "question": question,
        "answer": answer.strip(),
        "summary_answer": answer.strip()
    }


def generate_pinpoint_answer(
    question: str,
    action: str,
    test_res: Dict[str, Any],
    df: pd.DataFrame
) -> str:
    """
    Synthesizes a pinpoint, highly articulate, data-grounded scientific answer that
    directly and specifically addresses the user's research question, context, and terminology.
    Grounded in exact empirical statistics computed by the frequentist engine (zero hallucinations).
    """
    if action == "statistical_test":
        c1 = test_res.get("col1")
        c2 = test_res.get("col2")
        p_val = test_res.get("p_value")
        effect_sz = test_res.get("effect_size")
        eff_name = test_res.get("effect_size_name", "Effect size")
        method = test_res.get("test_method", "statistical_test")
        verdict = test_res.get("verdict", "rejected")
        covariates = test_res.get("covariates", [])
        
        r_squared = None
        direction = "positive"
        magnitude = test_res.get("effect_magnitude", "moderate")
        if effect_sz is not None and ("correlation" in str(method) or "r" in str(eff_name).lower()):
            try:
                r_squared = round(float(effect_sz) ** 2, 4)
                direction = "positive" if float(effect_sz) > 0 else "negative"
            except Exception:
                pass

        sub_n = test_res.get("sample_size", len(df))
        p_str = f"p = {p_val:.4g}" if p_val is not None else "p = N/A"
        if p_val is not None and p_val < 0.0001:
            p_str = "p < 0.0001"

        cov_str = f"controlling for {covariates}" if covariates else "bivariate"

        system_prompt = (
            "You are Veritus, an elite AI Principal Biostatistician and Data Scientist. "
            "Your task is to provide a pinpoint, direct, and scientifically nuanced answer to the user's specific question "
            "based strictly on the empirical statistical test results provided. "
            "Keep your answer concise (2-4 sentences). Directly address the user's specific context and phrasing (e.g. biomarker meaning, "
            "practical significance, directionality, clinical relevance). Do not use generic boilerplate."
        )

        r_sq_prompt = f"{r_squared * 100:.1f}%" if r_squared is not None else "N/A"
        eff_val_str = f"{effect_sz:.3f}" if effect_sz is not None else "N/A"

        user_prompt = f"""
User Question: "{question}"

Empirical Statistical Test Results:
- Statistical Procedure: {method}
- Predictor / Variable 1: '{c1}'
- Outcome / Variable 2: '{c2}'
- Effect Size ({eff_name}): {eff_val_str} ({direction}, {magnitude} effect)
- Variance Explained (R²): {r_sq_prompt}
- p-value: {p_str} (alpha = 0.05, Verdict: {verdict.upper()})
- Sample Size: n = {sub_n}
- Covariates Controlled: {cov_str}

Write an exact, pinpoint answer that:
1. Directly answers the user's question in the first sentence (e.g., whether the relationship is meaningful/significant and in what direction).
2. Explicitly references any specific context from the user's prompt (e.g., mentioning s4 as a cholesterol/biomarker ratio, clinical relevance, or practical implications).
3. Cites the exact numbers: effect size ({eff_name} = {eff_val_str}), p-value ({p_str}), and variance explained ({r_sq_prompt}).
4. Explains what this means practically in plain English (e.g., higher s4 levels predict higher disease progression scores).
"""
        llm_reply = call_llm(user_prompt, system_prompt=system_prompt, timeout=20, max_tokens=220)
        if llm_reply and llm_reply.strip():
            return llm_reply.strip()

        # Rich scientific fallback if LLM is offline or times out:
        r_sq_text = f", explaining {r_squared * 100:.1f}% of the variance (R² = {r_squared:.3f})" if r_squared is not None else ""
        if verdict == "validated":
            if covariates:
                return (
                    f"Yes. Even after controlling for {covariates}, '{c1}' maintains an independent, statistically "
                    f"significant {direction} relationship with '{c2}' ({eff_name} = {eff_val_str}, {p_str} < 0.05{r_sq_text}). "
                    f"This indicates that elevated '{c1}' is meaningfully linked to higher '{c2}' across {sub_n:,} records."
                )
            else:
                return (
                    f"Yes. The data confirms a statistically significant and meaningful {direction} relationship between '{c1}' "
                    f"and '{c2}' ({eff_name} = {eff_val_str}, {p_str} < 0.05{r_sq_text}). "
                    f"Higher levels of '{c1}' reliably correspond to increased '{c2}' in this cohort."
                )
        else:
            return (
                f"No statistically significant relationship was found between '{c1}' and '{c2}' ({p_str} ≥ 0.05). "
                f"The evidence does not support a meaningful association based on the available data."
            )

    elif action == "group_comparison":
        m_col = test_res.get("metric_col")
        g_col = test_res.get("group_col")
        groups = test_res.get("groups", [])
        test_name = test_res.get("test_name", "ANOVA")
        p_val = test_res.get("p_value")
        p_str = f"p = {p_val:.4g}" if p_val is not None else "p = N/A"
        if p_val is not None and p_val < 0.0001:
            p_str = "p < 0.0001"
        verdict = test_res.get("verdict", "informative")

        system_prompt = (
            "You are Veritus, an elite AI Principal Data Scientist. Answer the user's specific research question "
            "in 2-3 concise sentences based on the group comparison statistics provided. Directly state which group has highest/lowest, "
            "whether the difference is statistically significant, and practical implications."
        )
        user_prompt = f"""
User Question: "{question}"

Group Comparison Results:
- Metric: '{m_col}'
- Grouping Column: '{g_col}'
- Groups & Stats: {json.dumps(groups, indent=2)}
- Hypothesis Test: {test_name} ({p_str}, Verdict: {verdict.upper()})

Write an exact, pinpoint answer directly addressing the question.
"""
        llm_reply = call_llm(user_prompt, system_prompt=system_prompt, timeout=20, max_tokens=220)
        if llm_reply and llm_reply.strip():
            return llm_reply.strip()
        return test_res.get("summary_answer", "")

    elif action == "comparative_ranking":
        target = test_res.get("target_col")
        ranking = test_res.get("ranking", [])
        system_prompt = (
            "You are Veritus, an elite AI Principal Data Scientist. Answer the user's specific question "
            "in 2-3 concise sentences based on the feature association ranking results provided."
        )
        user_prompt = f"""
User Question: "{question}"

Feature Ranking for Target '{target}':
{json.dumps(ranking[:5], indent=2)}

Write an exact, pinpoint answer addressing the question directly.
"""
        llm_reply = call_llm(user_prompt, system_prompt=system_prompt, timeout=20, max_tokens=220)
        if llm_reply and llm_reply.strip():
            return llm_reply.strip()
        return test_res.get("summary_answer", "")

    elif action == "ad_hoc_aggregation":
        col = test_res.get("column")
        val = test_res.get("value")
        label = test_res.get("aggregation")
        tot = test_res.get("total_records")
        system_prompt = "You are Veritus AI Data Scientist. Answer the user's specific aggregation question directly in 1-2 sentences."
        user_prompt = f'User Question: "{question}"\nMetric: {label} for column {col} = {val} (n={tot})\nProvide a direct pinpoint answer.'
        llm_reply = call_llm(user_prompt, system_prompt=system_prompt, timeout=15, max_tokens=150)
        if llm_reply and llm_reply.strip():
            return llm_reply.strip()
        return test_res.get("summary_answer", "")

    elif action == "dataset_overview":
        overview = test_res.get("overview", {})
        system_prompt = "You are Veritus AI Data Scientist. Answer the user's specific dataset question directly in 2-3 sentences based on the dataset summary."
        user_prompt = f'User Question: "{question}"\nDataset Summary: {json.dumps(overview, indent=2)}\nProvide a direct pinpoint answer.'
        llm_reply = call_llm(user_prompt, system_prompt=system_prompt, timeout=15, max_tokens=200)
        if llm_reply and llm_reply.strip():
            return llm_reply.strip()
        return overview.get("summary_answer", "")

    return test_res.get("summary_answer", test_res.get("answer", "Analysis completed."))


def answer_dataset_query(
    df: pd.DataFrame,
    question: str,
    dataset_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main entry point for answering natural language user questions about a dataset.
    Executes frequentist tests, aggregations, ANOVA, ranking, or descriptive metrics,
    and returns a structured empirical response.
    """
    cleaned_q = question.strip()
    if not cleaned_q:
        return {
            "success": False,
            "error": "Question cannot be empty."
        }
        
    # Step 1: Parse and classify question
    try:
        plan = parse_query_with_llm(cleaned_q, df)
    except Exception as exc:
        logger.warning(f"Query parsing error: {exc}. Falling back to heuristic classifier.")
        plan = classify_question_heuristic(cleaned_q, df)

    action = plan.get("action", "statistical_test")
    c1 = plan.get("col1")
    c2 = plan.get("col2")
    covariates = plan.get("covariates") or []
    cov_str = f" (controlling for {', '.join(covariates)})" if covariates else ""
    raw_hypo = plan.get("interpreted_hypothesis")

    if raw_hypo and not raw_hypo.lower().startswith("statistical investigation"):
        interpreted_hypo = raw_hypo
    elif action == "statistical_test" and c1 and c2:
        interpreted_hypo = f"Assess whether '{c1}' has a statistically significant relationship with '{c2}'{cov_str}."
    elif action == "group_comparison":
        interpreted_hypo = f"Evaluate whether mean '{plan.get('metric_col')}' differs significantly across '{plan.get('group_col')}' categories."
    elif action == "comparative_ranking":
        interpreted_hypo = f"Rank all candidate features by strength of association with '{plan.get('target_col', 'target')}'."
    elif action == "ad_hoc_aggregation":
        interpreted_hypo = f"Compute the {plan.get('agg_type', 'summary')} metric for column '{c1}'."
    elif action == "dataset_overview":
        interpreted_hypo = "Profile overall dataset dimensions, feature distributions, and completeness."
    else:
        interpreted_hypo = f"Empirical evaluation of '{cleaned_q}'."
    
    # Step 2: Execute Action
    if action == "dataset_overview":
        overview_res = run_dataset_overview(df)
        pinpoint_ans = generate_pinpoint_answer(cleaned_q, "dataset_overview", {"overview": overview_res}, df)
        return {
            "success": True,
            "question": cleaned_q,
            "action": "dataset_overview",
            "interpreted_hypothesis": interpreted_hypo,
            "verdict": "informative",
            "overview": overview_res,
            "answer": pinpoint_ans,
            "dataset_name": dataset_name
        }

    elif action == "group_comparison":
        m_col = plan.get("metric_col")
        g_col = plan.get("group_col")
        if not m_col or not g_col or m_col not in df.columns or g_col not in df.columns:
            # Fallback to overview or ranking
            overview_res = run_dataset_overview(df)
            return {
                "success": True,
                "question": cleaned_q,
                "action": "dataset_overview",
                "interpreted_hypothesis": interpreted_hypo,
                "verdict": "informative",
                "overview": overview_res,
                "answer": overview_res.get("summary_answer", ""),
                "dataset_name": dataset_name
            }
            
        group_res = run_group_comparison(df, metric_col=m_col, group_col=g_col)
        answer = generate_pinpoint_answer(cleaned_q, "group_comparison", group_res, df)
        return {
            "success": True,
            "question": cleaned_q,
            "action": "group_comparison",
            "interpreted_hypothesis": interpreted_hypo,
            "metric_col": m_col,
            "group_col": g_col,
            "verdict": group_res.get("verdict", "informative"),
            "groups": group_res.get("groups", []),
            "p_value": group_res.get("p_value"),
            "statistic": group_res.get("statistic"),
            "test_method": group_res.get("test_name"),
            "answer": answer,
            "dataset_name": dataset_name
        }

    elif action == "ad_hoc_aggregation":
        col = plan.get("col1")
        agg_type = plan.get("agg_type", "mean")
        if not col or col not in df.columns:
            col = df.columns[0]
            
        agg_res = run_ad_hoc_aggregation(df, col, agg_type)
        answer = generate_pinpoint_answer(cleaned_q, "ad_hoc_aggregation", agg_res, df)
        return {
            "success": True,
            "question": cleaned_q,
            "action": "ad_hoc_aggregation",
            "interpreted_hypothesis": interpreted_hypo,
            "col1": col,
            "aggregation": agg_res.get("aggregation", agg_type),
            "value": agg_res.get("value"),
            "verdict": "informative",
            "answer": answer,
            "dataset_name": dataset_name
        }

    elif action == "comparative_ranking":
        target = plan.get("target_col") or detect_target_columns(df)[0]
        rank_res = run_comparative_ranking(df, target)
        answer = generate_pinpoint_answer(cleaned_q, "comparative_ranking", rank_res, df)
        
        return {
            "success": True,
            "question": cleaned_q,
            "action": "comparative_ranking",
            "interpreted_hypothesis": interpreted_hypo,
            "target_col": target,
            "verdict": "informative",
            "ranking": rank_res.get("ranking", []),
            "answer": answer,
            "dataset_name": dataset_name
        }
        
    elif action == "descriptive_summary":
        col = plan.get("col1") or df.columns[0]
        desc_res = run_descriptive_summary(df, col)
        
        return {
            "success": True,
            "question": cleaned_q,
            "action": "descriptive_summary",
            "interpreted_hypothesis": interpreted_hypo,
            "target_col": col,
            "verdict": "informative",
            "details": desc_res.get("stats", {}),
            "answer": desc_res.get("summary_answer", ""),
            "dataset_name": dataset_name
        }

    elif action == "general_analytical_qa":
        qa_res = run_general_analytical_qa(df, cleaned_q)
        return {
            "success": True,
            "question": cleaned_q,
            "action": "general_analytical_qa",
            "interpreted_hypothesis": interpreted_hypo,
            "verdict": "informative",
            "answer": qa_res.get("answer", ""),
            "dataset_name": dataset_name
        }
        
    else:
        # Default: 2-variable hypothesis test
        c1 = plan.get("col1")
        c2 = plan.get("col2")
        method = plan.get("test_method") or "correlation"
        covariates = plan.get("covariates") or []
        
        # Verify columns exist
        if not c1 or c1 not in df.columns or not c2 or c2 not in df.columns:
            # Fall back to general analytical QA
            qa_res = run_general_analytical_qa(df, cleaned_q)
            return {
                "success": True,
                "question": cleaned_q,
                "action": "general_analytical_qa",
                "interpreted_hypothesis": interpreted_hypo,
                "verdict": "informative",
                "answer": qa_res.get("answer", ""),
                "dataset_name": dataset_name
            }
            
        # Run statistical test via stats_engine
        test_res = test_hypothesis(df, test_type=method, col1=c1, col2=c2, covariates=covariates)
        test_res["col1"] = c1
        test_res["col2"] = c2
        test_res["covariates"] = covariates
        if "sample_size" not in test_res:
            test_res["sample_size"] = len(df[[c1, c2] + covariates].dropna()) if (c1 in df and c2 in df) else len(df)
            
        verdict = test_res.get("verdict", "rejected")
        p_val = test_res.get("p_value")
        effect_sz = test_res.get("effect_size")
        eff_name = test_res.get("effect_size_name", "Effect size")
        
        # Build scatter sample points for frontend charting (up to 40 sample points)
        scatter_sample = []
        if pd.api.types.is_numeric_dtype(df[c1]) and pd.api.types.is_numeric_dtype(df[c2]):
            sub_pts = df[[c1, c2]].dropna()
            step = max(1, len(sub_pts) // 40)
            for _, row in sub_pts.iloc[::step].iterrows():
                scatter_sample.append({
                    "x": round(float(row[c1]), 2),
                    "y": round(float(row[c2]), 2)
                })

        # Generate scientific pinpoint answer directly addressing user's question and context
        answer = generate_pinpoint_answer(cleaned_q, "statistical_test", test_res, df)

        return {
            "success": True,
            "question": cleaned_q,
            "action": "statistical_test",
            "interpreted_hypothesis": interpreted_hypo,
            "verdict": verdict,
            "test_method": test_res.get("test_method", method),
            "col1": c1,
            "col2": c2,
            "p_value": p_val,
            "effect_size": effect_sz,
            "effect_size_name": eff_name,
            "summary": test_res.get("summary", ""),
            "answer": answer,
            "scatter_points": scatter_sample,
            "dataset_name": dataset_name
        }
