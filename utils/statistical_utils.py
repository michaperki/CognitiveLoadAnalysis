import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy import stats
import logging


def calculate_confidence_interval(values: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for a set of values.
    
    Args:
        values: Array of values
        confidence: Confidence level (0-1)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    # Calculate mean and standard error
    mean = np.mean(values)
    se = stats.sem(values)
    
    # Calculate confidence interval
    h = se * stats.t.ppf((1 + confidence) / 2, len(values) - 1)
    
    return mean - h, mean + h


def bootstrap_statistic(values: np.ndarray, statistic: callable, 
                       n_bootstrap: int = 1000, confidence: float = 0.95,
                       random_state: int = 42) -> Dict[str, Any]:
    """
    Calculate bootstrapped statistics with confidence intervals.
    
    Args:
        values: Array of values
        statistic: Function to calculate statistic (e.g., np.mean, np.median)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (0-1)
        random_state: Random seed
        
    Returns:
        Dictionary with mean, std, and confidence interval
    """
    np.random.seed(random_state)
    
    # Calculate bootstrap samples
    bootstrap_samples = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_samples.append(statistic(sample))
    
    # Calculate statistics
    mean = np.mean(bootstrap_samples)
    std = np.std(bootstrap_samples)
    
    # Calculate confidence interval
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_samples, alpha/2 * 100)
    upper = np.percentile(bootstrap_samples, (1 - alpha/2) * 100)
    
    return {
        'mean': mean,
        'std': std,
        'ci_lower': lower,
        'ci_upper': upper,
        'samples': bootstrap_samples
    }


def paired_bootstrap_test(values1: np.ndarray, values2: np.ndarray,
                        statistic: callable, n_bootstrap: int = 1000,
                        alpha: float = 0.05, random_state: int = 42) -> Dict[str, Any]:
    """
    Perform paired bootstrap test for comparing two sets of values.
    
    Args:
        values1: First array of values
        values2: Second array of values
        statistic: Function to calculate statistic (e.g., np.mean, np.median)
        n_bootstrap: Number of bootstrap samples
        alpha: Significance level
        random_state: Random seed
        
    Returns:
        Dictionary with test results
    """
    if len(values1) != len(values2):
        raise ValueError("Arrays must have the same length for paired bootstrap")
    
    np.random.seed(random_state)
    
    # Calculate observed difference
    observed_diff = statistic(values1) - statistic(values2)
    
    # Calculate bootstrap distribution of differences
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        # Generate paired samples
        indices = np.random.choice(len(values1), size=len(values1), replace=True)
        bootstrap_diffs.append(statistic(values1[indices]) - statistic(values2[indices]))
    
    # Calculate p-value (two-sided test)
    if observed_diff >= 0:
        p_value = np.mean(bootstrap_diffs <= 0) * 2
    else:
        p_value = np.mean(bootstrap_diffs >= 0) * 2
    
    # Ensure p-value is within valid range
    p_value = min(p_value, 1.0)
    
    # Calculate confidence interval
    ci_lower = np.percentile(bootstrap_diffs, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_diffs, (1 - alpha/2) * 100)
    
    # Determine significance
    significant = p_value < alpha
    
    return {
        'observed_diff': observed_diff,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
