"""
Statistical Analysis Module

Advanced statistical analysis utilities for model evaluation,
significance testing, and performance comparison.
"""

import numpy as np
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import pandas as pd
from collections import defaultdict

from .logger import get_logger

logger = get_logger(__name__)

@dataclass
class StatisticalTest:
    """Container for statistical test results"""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: Optional[str] = None

class StatisticalAnalyzer:
    """Advanced statistical analysis for model comparison and evaluation"""
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize statistical analyzer.
        
        Args:
            alpha: Significance level for hypothesis testing
        """
        self.alpha = alpha
    
    def paired_t_test(self, 
                     baseline_scores: List[float], 
                     treatment_scores: List[float]) -> StatisticalTest:
        """
        Perform paired t-test for comparing two related samples.
        
        Args:
            baseline_scores: Baseline model scores
            treatment_scores: Treatment model scores
            
        Returns:
            StatisticalTest result
        """
        if len(baseline_scores) != len(treatment_scores):
            raise ValueError("Sample sizes must be equal for paired t-test")
        
        baseline_arr = np.array(baseline_scores)
        treatment_arr = np.array(treatment_scores)
        
        # Perform paired t-test
        t_stat, p_value = stats.ttest_rel(treatment_arr, baseline_arr)
        
        # Calculate effect size (Cohen's d)
        differences = treatment_arr - baseline_arr
        pooled_std = np.sqrt((np.var(baseline_arr, ddof=1) + np.var(treatment_arr, ddof=1)) / 2)
        cohens_d = np.mean(differences) / pooled_std if pooled_std > 0 else 0
        
        # Calculate confidence interval for the difference
        n = len(differences)
        se_diff = stats.sem(differences)
        t_critical = stats.t.ppf(1 - self.alpha/2, n - 1)
        margin_error = t_critical * se_diff
        mean_diff = np.mean(differences)
        ci = (mean_diff - margin_error, mean_diff + margin_error)
        
        # Interpretation
        interpretation = self._interpret_cohens_d(abs(cohens_d))
        
        return StatisticalTest(
            test_name="Paired t-test",
            statistic=t_stat,
            p_value=p_value,
            significant=p_value < self.alpha,
            effect_size=cohens_d,
            confidence_interval=ci,
            interpretation=interpretation
        )
    
    def wilcoxon_signed_rank_test(self, 
                                baseline_scores: List[float], 
                                treatment_scores: List[float]) -> StatisticalTest:
        """
        Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).
        
        Args:
            baseline_scores: Baseline model scores
            treatment_scores: Treatment model scores
            
        Returns:
            StatisticalTest result
        """
        if len(baseline_scores) != len(treatment_scores):
            raise ValueError("Sample sizes must be equal for Wilcoxon signed-rank test")
        
        baseline_arr = np.array(baseline_scores)
        treatment_arr = np.array(treatment_scores)
        
        # Perform Wilcoxon signed-rank test
        try:
            w_stat, p_value = stats.wilcoxon(treatment_arr, baseline_arr, alternative='two-sided')
        except ValueError as e:
            # Handle case where all differences are zero
            logger.warning(f"Wilcoxon test failed: {e}")
            return StatisticalTest(
                test_name="Wilcoxon signed-rank test",
                statistic=np.nan,
                p_value=1.0,
                significant=False,
                interpretation="Test could not be performed (likely no differences)"
            )
        
        # Calculate effect size (r)
        n = len(baseline_scores)
        z_score = (w_stat - n*(n+1)/4) / np.sqrt(n*(n+1)*(2*n+1)/24)
        effect_size_r = z_score / np.sqrt(n)
        
        return StatisticalTest(
            test_name="Wilcoxon signed-rank test",
            statistic=w_stat,
            p_value=p_value,
            significant=p_value < self.alpha,
            effect_size=effect_size_r,
            interpretation=self._interpret_effect_size_r(abs(effect_size_r))
        )
    
    def mann_whitney_u_test(self, 
                           baseline_scores: List[float], 
                           treatment_scores: List[float]) -> StatisticalTest:
        """
        Perform Mann-Whitney U test for independent samples.
        
        Args:
            baseline_scores: Baseline model scores
            treatment_scores: Treatment model scores
            
        Returns:
            StatisticalTest result
        """
        baseline_arr = np.array(baseline_scores)
        treatment_arr = np.array(treatment_scores)
        
        # Perform Mann-Whitney U test
        u_stat, p_value = stats.mannwhitneyu(treatment_arr, baseline_arr, alternative='two-sided')
        
        # Calculate effect size (rank-biserial correlation)
        n1, n2 = len(baseline_scores), len(treatment_scores)
        effect_size = (2 * u_stat) / (n1 * n2) - 1
        
        return StatisticalTest(
            test_name="Mann-Whitney U test",
            statistic=u_stat,
            p_value=p_value,
            significant=p_value < self.alpha,
            effect_size=effect_size,
            interpretation=self._interpret_effect_size_r(abs(effect_size))
        )
    
    def bootstrap_confidence_interval(self, 
                                    scores: List[float], 
                                    statistic_func=np.mean,
                                    n_bootstrap: int = 10000,
                                    confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval for a statistic.
        
        Args:
            scores: Data points
            statistic_func: Function to calculate statistic (default: mean)
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (default: 0.95)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        scores_arr = np.array(scores)
        n = len(scores_arr)
        
        # Generate bootstrap samples
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(scores_arr, size=n, replace=True)
            bootstrap_stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(bootstrap_stat)
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)
        
        return (ci_lower, ci_upper)
    
    def permutation_test(self, 
                        baseline_scores: List[float], 
                        treatment_scores: List[float],
                        n_permutations: int = 10000) -> StatisticalTest:
        """
        Perform permutation test for comparing two groups.
        
        Args:
            baseline_scores: Baseline model scores
            treatment_scores: Treatment model scores
            n_permutations: Number of permutations
            
        Returns:
            StatisticalTest result
        """
        baseline_arr = np.array(baseline_scores)
        treatment_arr = np.array(treatment_scores)
        
        # Observed difference in means
        observed_diff = np.mean(treatment_arr) - np.mean(baseline_arr)
        
        # Combine all scores
        all_scores = np.concatenate([baseline_arr, treatment_arr])
        n_baseline = len(baseline_arr)
        n_total = len(all_scores)
        
        # Perform permutations
        permutation_diffs = []
        for _ in range(n_permutations):
            # Randomly shuffle and split
            np.random.shuffle(all_scores)
            perm_baseline = all_scores[:n_baseline]
            perm_treatment = all_scores[n_baseline:]
            
            # Calculate difference
            perm_diff = np.mean(perm_treatment) - np.mean(perm_baseline)
            permutation_diffs.append(perm_diff)
        
        # Calculate p-value (two-tailed)
        permutation_diffs = np.array(permutation_diffs)
        p_value = np.mean(np.abs(permutation_diffs) >= np.abs(observed_diff))
        
        return StatisticalTest(
            test_name="Permutation test",
            statistic=observed_diff,
            p_value=p_value,
            significant=p_value < self.alpha,
            interpretation=f"Observed difference: {observed_diff:.4f}"
        )
    
    def bayesian_t_test(self, 
                       baseline_scores: List[float], 
                       treatment_scores: List[float]) -> Dict[str, Any]:
        """
        Perform Bayesian t-test for model comparison.
        
        Args:
            baseline_scores: Baseline model scores
            treatment_scores: Treatment model scores
            
        Returns:
            Dictionary with Bayesian analysis results
        """
        try:
            from scipy.stats import bayes_mvs
        except ImportError:
            logger.warning("Bayesian analysis requires scipy.stats.bayes_mvs")
            return {"error": "Bayesian analysis not available"}
        
        baseline_arr = np.array(baseline_scores)
        treatment_arr = np.array(treatment_scores)
        differences = treatment_arr - baseline_arr
        
        # Bayesian estimation of mean difference
        mean_estimate, var_estimate, std_estimate = bayes_mvs(differences, alpha=0.05)
        
        # Calculate probability that treatment is better
        prob_better = np.mean(differences > 0)
        
        return {
            "mean_difference": {
                "estimate": mean_estimate.statistic,
                "confidence_interval": mean_estimate.minmax
            },
            "probability_treatment_better": prob_better,
            "variance_estimate": {
                "estimate": var_estimate.statistic,
                "confidence_interval": var_estimate.minmax
            }
        }
    
    def comprehensive_comparison(self, 
                               baseline_scores: List[float], 
                               treatment_scores: List[float]) -> Dict[str, Any]:
        """
        Perform comprehensive statistical comparison between two models.
        
        Args:
            baseline_scores: Baseline model scores
            treatment_scores: Treatment model scores
            
        Returns:
            Dictionary with all comparison results
        """
        logger.info("Performing comprehensive statistical comparison")
        
        results = {
            "descriptive_statistics": self._calculate_descriptive_stats(baseline_scores, treatment_scores),
            "normality_tests": self._test_normality(baseline_scores, treatment_scores),
        }
        
        # Parametric tests
        if len(baseline_scores) == len(treatment_scores):
            results["paired_t_test"] = self.paired_t_test(baseline_scores, treatment_scores)
            results["wilcoxon_test"] = self.wilcoxon_signed_rank_test(baseline_scores, treatment_scores)
        else:
            logger.warning("Unequal sample sizes, using independent sample tests")
            results["independent_t_test"] = self._independent_t_test(baseline_scores, treatment_scores)
        
        # Non-parametric tests
        results["mann_whitney_test"] = self.mann_whitney_u_test(baseline_scores, treatment_scores)
        
        # Resampling methods
        results["permutation_test"] = self.permutation_test(baseline_scores, treatment_scores)
        
        # Bootstrap confidence intervals
        results["bootstrap_ci_baseline"] = self.bootstrap_confidence_interval(baseline_scores)
        results["bootstrap_ci_treatment"] = self.bootstrap_confidence_interval(treatment_scores)
        
        # Bayesian analysis
        if len(baseline_scores) == len(treatment_scores):
            results["bayesian_analysis"] = self.bayesian_t_test(baseline_scores, treatment_scores)
        
        return results
    
    def _calculate_descriptive_stats(self, 
                                   baseline_scores: List[float], 
                                   treatment_scores: List[float]) -> Dict[str, Any]:
        """Calculate descriptive statistics for both groups"""
        
        def stats_for_group(scores):
            arr = np.array(scores)
            return {
                "n": len(arr),
                "mean": np.mean(arr),
                "median": np.median(arr),
                "std": np.std(arr, ddof=1),
                "min": np.min(arr),
                "max": np.max(arr),
                "q25": np.percentile(arr, 25),
                "q75": np.percentile(arr, 75),
                "skewness": stats.skew(arr),
                "kurtosis": stats.kurtosis(arr)
            }
        
        return {
            "baseline": stats_for_group(baseline_scores),
            "treatment": stats_for_group(treatment_scores)
        }
    
    def _test_normality(self, 
                       baseline_scores: List[float], 
                       treatment_scores: List[float]) -> Dict[str, Any]:
        """Test normality of distributions"""
        
        def normality_for_group(scores, group_name):
            arr = np.array(scores)
            
            # Shapiro-Wilk test (better for small samples)
            if len(arr) <= 5000:
                sw_stat, sw_p = stats.shapiro(arr)
            else:
                sw_stat, sw_p = np.nan, np.nan
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_p = stats.kstest(arr, 'norm', args=(np.mean(arr), np.std(arr)))
            
            return {
                f"{group_name}_shapiro_wilk": {
                    "statistic": sw_stat,
                    "p_value": sw_p,
                    "normal": sw_p > self.alpha if not np.isnan(sw_p) else None
                },
                f"{group_name}_kolmogorov_smirnov": {
                    "statistic": ks_stat,
                    "p_value": ks_p,
                    "normal": ks_p > self.alpha
                }
            }
        
        baseline_tests = normality_for_group(baseline_scores, "baseline")
        treatment_tests = normality_for_group(treatment_scores, "treatment")
        
        return {**baseline_tests, **treatment_tests}
    
    def _independent_t_test(self, 
                          baseline_scores: List[float], 
                          treatment_scores: List[float]) -> StatisticalTest:
        """Perform independent samples t-test"""
        baseline_arr = np.array(baseline_scores)
        treatment_arr = np.array(treatment_scores)
        
        # Perform independent t-test (assuming equal variances)
        t_stat, p_value = stats.ttest_ind(treatment_arr, baseline_arr)
        
        # Calculate Cohen's d for independent samples
        pooled_std = np.sqrt(((len(baseline_arr) - 1) * np.var(baseline_arr, ddof=1) + 
                             (len(treatment_arr) - 1) * np.var(treatment_arr, ddof=1)) / 
                            (len(baseline_arr) + len(treatment_arr) - 2))
        
        cohens_d = (np.mean(treatment_arr) - np.mean(baseline_arr)) / pooled_std if pooled_std > 0 else 0
        
        return StatisticalTest(
            test_name="Independent t-test",
            statistic=t_stat,
            p_value=p_value,
            significant=p_value < self.alpha,
            effect_size=cohens_d,
            interpretation=self._interpret_cohens_d(abs(cohens_d))
        )
    
    def _interpret_cohens_d(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""
        if cohens_d < 0.2:
            return "negligible effect"
        elif cohens_d < 0.5:
            return "small effect"
        elif cohens_d < 0.8:
            return "medium effect"
        else:
            return "large effect"
    
    def _interpret_effect_size_r(self, r: float) -> str:
        """Interpret correlation-based effect size"""
        if r < 0.1:
            return "negligible effect"
        elif r < 0.3:
            return "small effect"
        elif r < 0.5:
            return "medium effect"
        else:
            return "large effect"
    
    def create_statistical_report(self, comparison_results: Dict[str, Any]) -> str:
        """
        Create a comprehensive statistical report.
        
        Args:
            comparison_results: Results from comprehensive_comparison
            
        Returns:
            Formatted report string
        """
        report = "Statistical Comparison Report\n"
        report += "=" * 35 + "\n\n"
        
        # Descriptive statistics
        if "descriptive_statistics" in comparison_results:
            stats_data = comparison_results["descriptive_statistics"]
            report += "Descriptive Statistics:\n"
            report += "-" * 21 + "\n"
            
            for group in ["baseline", "treatment"]:
                if group in stats_data:
                    s = stats_data[group]
                    report += f"\n{group.capitalize()}:\n"
                    report += f"  N: {s['n']}\n"
                    report += f"  Mean: {s['mean']:.4f} ± {s['std']:.4f}\n"
                    report += f"  Median: {s['median']:.4f}\n"
                    report += f"  Range: [{s['min']:.4f}, {s['max']:.4f}]\n"
        
        # Statistical tests
        report += "\n\nStatistical Tests:\n"
        report += "-" * 17 + "\n"
        
        test_keys = ["paired_t_test", "independent_t_test", "wilcoxon_test", 
                    "mann_whitney_test", "permutation_test"]
        
        for test_key in test_keys:
            if test_key in comparison_results:
                test_result = comparison_results[test_key]
                if isinstance(test_result, StatisticalTest):
                    report += f"\n{test_result.test_name}:\n"
                    report += f"  Statistic: {test_result.statistic:.4f}\n"
                    report += f"  P-value: {test_result.p_value:.6f}\n"
                    report += f"  Significant: {'Yes' if test_result.significant else 'No'}\n"
                    
                    if test_result.effect_size is not None:
                        report += f"  Effect size: {test_result.effect_size:.4f}\n"
                    
                    if test_result.interpretation:
                        report += f"  Interpretation: {test_result.interpretation}\n"
        
        # Bayesian analysis
        if "bayesian_analysis" in comparison_results:
            bayes = comparison_results["bayesian_analysis"]
            if "error" not in bayes:
                report += "\nBayesian Analysis:\n"
                report += "-" * 17 + "\n"
                report += f"  Probability treatment is better: {bayes['probability_treatment_better']:.3f}\n"
                
                mean_diff = bayes["mean_difference"]
                report += f"  Mean difference: {mean_diff['estimate']:.4f}\n"
                report += f"  95% CI: [{mean_diff['confidence_interval'][0]:.4f}, "
                report += f"{mean_diff['confidence_interval'][1]:.4f}]\n"
        
        return report