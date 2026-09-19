"""
Veritus — Statistical Engine (backend/stats_engine.py)

Performs rigorous statistical hypothesis testing on pandas DataFrames.
Does not rely on LLM guessing — uses scientific frequentist testing:
  1. Two-sample T-test (Welch's t-test) + Cohen's d effect size
  2. Correlation test (Pearson & Spearman) + r effect size
  3. Chi-square test of independence + Cramér's V effect size

Every test returns an empirical p-value, effect size, and definitive verdict
('validated' if p < alpha, else 'rejected').
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

ALPHA_DEFAULT = 0.05

def calculate_cohens_d(group1: pd.Series, group2: pd.Series) -> float:
    """
    Computes Cohen's d effect size between two independent groups.
    Formula: (mean1 - mean2) / pooled_standard_deviation
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    
    if n1 + n2 <= 2:
        return 0.0
        
    # Pooled standard deviation
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_se == 0 or np.isnan(pooled_se):
        return 0.0
        
    return float((group1.mean() - group2.mean()) / pooled_se)

def calculate_cramers_v(contingency_table: pd.DataFrame) -> float:
    """
    Computes Cramér's V effect size for a contingency table.
    Formula: sqrt(chi2 / (n * min(rows - 1, cols - 1)))
    """
    chi2 = stats.chi2_contingency(contingency_table, correction=False)[0]
    n = contingency_table.values.sum()
    r, k = contingency_table.shape
    min_dim = min(r - 1, k - 1)
    
    if n == 0 or min_dim <= 0:
        return 0.0
        
    v = np.sqrt(chi2 / (n * min_dim))
    return float(v)

def run_t_test(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    alpha: float = ALPHA_DEFAULT,
    group_values: Optional[Tuple[Any, Any]] = None
) -> Dict[str, Any]:
    """
    Performs a Two-Sample Welch's T-Test comparing metric_col across two groups in group_col.
    """
    if group_col not in df.columns or metric_col not in df.columns:
        return {
            "test_method": "two_sample_t_test",
            "verdict": "inconclusive",
            "error": f"Columns '{group_col}' or '{metric_col}' not found in dataframe."
        }
    
    # Drop missing values in the relevant columns
    sub_df = df[[group_col, metric_col]].dropna()
    
    # Ensure numeric metric column
    sub_df[metric_col] = pd.to_numeric(sub_df[metric_col], errors="coerce")
    sub_df = sub_df.dropna(subset=[metric_col])
    
    unique_groups = sub_df[group_col].unique()
    if len(unique_groups) < 2:
        return {
            "test_method": "two_sample_t_test",
            "verdict": "inconclusive",
            "error": f"Column '{group_col}' must contain at least 2 distinct groups (found {len(unique_groups)})."
        }
    
    # Select the two groups to compare
    if group_values and len(group_values) == 2:
        g1_val, g2_val = group_values
    else:
        g1_val, g2_val = unique_groups[0], unique_groups[1]
        
    g1 = sub_df[sub_df[group_col] == g1_val][metric_col]
    g2 = sub_df[sub_df[group_col] == g2_val][metric_col]
    
    if len(g1) < 3 or len(g2) < 3:
        return {
            "test_method": "two_sample_t_test",
            "verdict": "inconclusive",
            "error": f"Insufficient samples: Group '{g1_val}' (n={len(g1)}), Group '{g2_val}' (n={len(g2)}). Minimum 3 required."
        }
        
    # Run Welch's t-test (equal_var=False is more reliable for real-world variance differences)
    t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False, nan_policy="omit")
    
    if np.isnan(p_val) or np.isnan(t_stat):
        return {
            "test_method": "two_sample_t_test",
            "verdict": "inconclusive",
            "error": "T-test produced NaN (likely zero variance within groups)."
        }
        
    cohens_d = calculate_cohens_d(g1, g2)
    verdict = "validated" if p_val < alpha else "rejected"
    
    # Magnitude descriptor for Cohen's d
    abs_d = abs(cohens_d)
    d_mag = "large" if abs_d >= 0.8 else "medium" if abs_d >= 0.5 else "small" if abs_d >= 0.2 else "negligible"
    
    summary = (
        f"Comparing '{metric_col}' between '{group_col}={g1_val}' (mean={g1.mean():.2f}, n={len(g1)}) "
        f"and '{group_col}={g2_val}' (mean={g2.mean():.2f}, n={len(g2)}): "
        f"t={t_stat:.3f}, p-value={p_val:.4g}, Cohen's d={cohens_d:.3f} ({d_mag} effect). "
        f"Verdict: {verdict.upper()} at alpha={alpha}."
    )
    
    return {
        "test_method": "two_sample_t_test",
        "group_col": group_col,
        "metric_col": metric_col,
        "group_values": (str(g1_val), str(g2_val)),
        "group1_mean": float(g1.mean()),
        "group2_mean": float(g2.mean()),
        "test_statistic": float(t_stat),
        "p_value": float(p_val),
        "effect_size": float(cohens_d),
        "effect_size_name": "Cohen's d",
        "effect_magnitude": d_mag,
        "alpha": alpha,
        "verdict": verdict,
        "summary": summary
    }

def run_correlation(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    method: str = "pearson",
    alpha: float = ALPHA_DEFAULT
) -> Dict[str, Any]:
    """
    Computes Pearson (linear) or Spearman (rank) correlation between two numeric continuous columns.
    """
    if col1 not in df.columns or col2 not in df.columns:
        return {
            "test_method": f"{method}_correlation",
            "verdict": "inconclusive",
            "error": f"Columns '{col1}' or '{col2}' not found in dataframe."
        }
        
    sub_df = df[[col1, col2]].dropna()
    sub_df[col1] = pd.to_numeric(sub_df[col1], errors="coerce")
    sub_df[col2] = pd.to_numeric(sub_df[col2], errors="coerce")
    sub_df = sub_df.dropna()
    
    if len(sub_df) < 4:
        return {
            "test_method": f"{method}_correlation",
            "verdict": "inconclusive",
            "error": f"Sample size too small (n={len(sub_df)}). Minimum 4 required."
        }
        
    # Check for zero variance
    if sub_df[col1].std() == 0 or sub_df[col2].std() == 0:
        return {
            "test_method": f"{method}_correlation",
            "verdict": "inconclusive",
            "error": "One or both columns have zero variance (all identical values)."
        }
        
    if method.lower() == "spearman":
        corr, p_val = stats.spearmanr(sub_df[col1], sub_df[col2])
        test_method_name = "spearman_correlation"
        effect_name = "Spearman's rho"
    else:
        corr, p_val = stats.pearsonr(sub_df[col1], sub_df[col2])
        test_method_name = "pearson_correlation"
        effect_name = "Pearson's r"
        
    if np.isnan(corr) or np.isnan(p_val):
        return {
            "test_method": test_method_name,
            "verdict": "inconclusive",
            "error": "Correlation calculation resulted in NaN."
        }
        
    verdict = "validated" if p_val < alpha else "rejected"
    
    abs_r = abs(corr)
    r_mag = "strong" if abs_r >= 0.5 else "moderate" if abs_r >= 0.3 else "weak" if abs_r >= 0.1 else "negligible"
    direction = "positive" if corr > 0 else "negative"
    
    summary = (
        f"Correlation between '{col1}' and '{col2}' (n={len(sub_df)}): "
        f"{effect_name}={corr:.3f} ({direction}, {r_mag}), p-value={p_val:.4g}. "
        f"Verdict: {verdict.upper()} at alpha={alpha}."
    )
    
    return {
        "test_method": test_method_name,
        "col1": col1,
        "col2": col2,
        "sample_size": len(sub_df),
        "test_statistic": float(corr),
        "p_value": float(p_val),
        "effect_size": float(corr),
        "effect_size_name": effect_name,
        "effect_magnitude": r_mag,
        "alpha": alpha,
        "verdict": verdict,
        "summary": summary
    }

def run_chi_square(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    alpha: float = ALPHA_DEFAULT
) -> Dict[str, Any]:
    """
    Performs Chi-Square Test of Independence between two categorical columns.
    """
    if col1 not in df.columns or col2 not in df.columns:
        return {
            "test_method": "chi_square",
            "verdict": "inconclusive",
            "error": f"Columns '{col1}' or '{col2}' not found in dataframe."
        }
        
    sub_df = df[[col1, col2]].dropna()
    contingency_table = pd.crosstab(sub_df[col1], sub_df[col2])
    
    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        return {
            "test_method": "chi_square",
            "verdict": "inconclusive",
            "error": f"Contingency table dimension must be at least 2x2. Current shape: {contingency_table.shape}."
        }
        
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
    cramers_v = calculate_cramers_v(contingency_table)
    
    verdict = "validated" if p_val < alpha else "rejected"
    
    v_mag = "large" if cramers_v >= 0.35 else "medium" if cramers_v >= 0.20 else "small" if cramers_v >= 0.10 else "negligible"
    
    summary = (
        f"Chi-square test of independence between '{col1}' and '{col2}' (dof={dof}): "
        f"chi2={chi2:.3f}, p-value={p_val:.4g}, Cramér's V={cramers_v:.3f} ({v_mag} association). "
        f"Verdict: {verdict.upper()} at alpha={alpha}."
    )
    
    return {
        "test_method": "chi_square",
        "col1": col1,
        "col2": col2,
        "degrees_of_freedom": int(dof),
        "test_statistic": float(chi2),
        "p_value": float(p_val),
        "effect_size": float(cramers_v),
        "effect_size_name": "Cramér's V",
        "effect_magnitude": v_mag,
        "alpha": alpha,
        "verdict": verdict,
        "summary": summary
    }

def test_hypothesis(
    df: pd.DataFrame,
    test_type: str,
    col1: str,
    col2: str,
    alpha: float = ALPHA_DEFAULT,
    **kwargs
) -> Dict[str, Any]:
    """
    Universal dispatcher that takes a hypothesis test specification and executes
    the appropriate statistical procedure on the dataframe.
    """
    test_type_clean = test_type.lower().strip()
    
    if "t_test" in test_type_clean or "t-test" in test_type_clean:
        return run_t_test(df, group_col=col1, metric_col=col2, alpha=alpha, **kwargs)
    elif "spearman" in test_type_clean:
        return run_correlation(df, col1=col1, col2=col2, method="spearman", alpha=alpha)
    elif "corr" in test_type_clean or "pearson" in test_type_clean:
        return run_correlation(df, col1=col1, col2=col2, method="pearson", alpha=alpha)
    elif "chi" in test_type_clean:
        return run_chi_square(df, col1=col1, col2=col2, alpha=alpha)
    else:
        # Fallback automatic heuristic based on column types
        if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
            return run_correlation(df, col1=col1, col2=col2, method="pearson", alpha=alpha)
        elif pd.api.types.is_numeric_dtype(df[col2]):
            return run_t_test(df, group_col=col1, metric_col=col2, alpha=alpha)
        elif pd.api.types.is_numeric_dtype(df[col1]):
            return run_t_test(df, group_col=col2, metric_col=col1, alpha=alpha)
        else:
            return run_chi_square(df, col1=col1, col2=col2, alpha=alpha)

if __name__ == "__main__":
    print("=== Testing Veritus Statistics Engine ===")
    
    # Create realistic synthetic experiment dataset
    np.random.seed(42)
    n = 120
    
    # 1. True correlation: ad spend drives revenue
    ad_spend = np.random.uniform(100, 1000, n)
    revenue = 2.5 * ad_spend + np.random.normal(0, 150, n)
    
    # 2. Pure random noise (should be REJECTED)
    noise1 = np.random.normal(50, 10, n)
    noise2 = np.random.normal(50, 10, n)
    
    # 3. Two distinct groups (significant t-test)
    tier = np.random.choice(["Premium", "Basic"], size=n)
    satisfaction = np.where(tier == "Premium", np.random.normal(8.5, 1.0, n), np.random.normal(6.2, 1.2, n))
    
    # 4. Categorical association (Chi-Square)
    device = np.random.choice(["Mobile", "Desktop"], size=n, p=[0.6, 0.4])
    converted = np.where(device == "Desktop", np.random.choice(["Yes", "No"], size=n, p=[0.7, 0.3]),
                         np.random.choice(["Yes", "No"], size=n, p=[0.3, 0.7]))
    
    sample_df = pd.DataFrame({
        "ad_spend": ad_spend,
        "revenue": revenue,
        "noise1": noise1,
        "noise2": noise2,
        "tier": tier,
        "satisfaction": satisfaction,
        "device": device,
        "converted": converted
    })
    
    print("\n[Test 1: True Correlation] ad_spend vs revenue")
    res1 = run_correlation(sample_df, "ad_spend", "revenue")
    print(res1["summary"])
    assert res1["verdict"] == "validated", "Expected true correlation to validate!"
    
    print("\n[Test 2: False Correlation (Noise)] noise1 vs noise2")
    res2 = run_correlation(sample_df, "noise1", "noise2")
    print(res2["summary"])
    assert res2["verdict"] == "rejected", "Expected noise correlation to be rejected!"
    
    print("\n[Test 3: Two-sample T-Test] tier vs satisfaction")
    res3 = run_t_test(sample_df, "tier", "satisfaction")
    print(res3["summary"])
    assert res3["verdict"] == "validated", "Expected significant difference in satisfaction!"
    
    print("\n[Test 4: Chi-Square Test] device vs converted")
    res4 = run_chi_square(sample_df, "device", "converted")
    print(res4["summary"])
    assert res4["verdict"] == "validated", "Expected categorical association to validate!"
    
    print("\n--- All Statistical Engine Tests Passed Flawlessly! ---")
