"""
Comprehensive Evaluation Module

Advanced evaluation framework for financial domain LLM fine-tuning
with multiple metrics, statistical analysis, and comparative assessment.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time
from concurrent.futures import ThreadPoolExecutor
import warnings

# NLP and ML evaluation libraries
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sacrebleu import BLEU
import evaluate
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import scipy.stats as stats

# HuggingFace
from transformers import pipeline, Pipeline
from datasets import Dataset

from ..core.config import EvaluationConfig
from ..models.llama_adapter import LlamaFinancialAdapter
from ..utils.logger import get_logger
from ..utils.statistics import StatisticalAnalyzer

logger = get_logger(__name__)

@dataclass
class EvaluationResult:
    """Container for individual evaluation result"""
    sample_id: str
    ground_truth: str
    prediction: str
    metrics: Dict[str, float]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class AggregatedResults:
    """Container for aggregated evaluation results"""
    total_samples: int
    metrics_summary: Dict[str, Dict[str, float]]  # metric_name -> {mean, std, min, max}
    comparative_analysis: Optional[Dict[str, Any]] = None
    statistical_significance: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class MetricCalculator:
    """Advanced metric calculation with financial domain considerations"""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bleu_scorer = BLEU()
        self.perplexity_evaluator = evaluate.load("perplexity", module_type="metric")
        
        # Initialize sentence transformer for semantic similarity
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.warning(f"Failed to load sentence transformer: {e}")
            self.sentence_model = None
    
    def calculate_all_metrics(self, 
                            ground_truth: str, 
                            prediction: str,
                            model_outputs: Optional[Dict] = None) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for a single prediction.
        
        Args:
            ground_truth: Reference text
            prediction: Model prediction
            model_outputs: Optional model outputs for additional metrics
            
        Returns:
            Dictionary of metric scores
        """
        metrics = {}
        
        # Basic text metrics
        metrics.update(self._calculate_rouge_scores(ground_truth, prediction))
        metrics.update(self._calculate_bleu_score(ground_truth, prediction))
        metrics.update(self._calculate_word_overlap(ground_truth, prediction))
        
        # Semantic metrics
        if self.sentence_model:
            metrics.update(self._calculate_semantic_similarity(ground_truth, prediction))
        
        # Financial domain specific metrics
        metrics.update(self._calculate_financial_metrics(ground_truth, prediction))
        
        # BERTScore (more computationally expensive)
        try:
            metrics.update(self._calculate_bert_score(ground_truth, prediction))
        except Exception as e:
            logger.warning(f"BERTScore calculation failed: {e}")
        
        return metrics
    
    def _calculate_rouge_scores(self, ground_truth: str, prediction: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        scores = self.rouge_scorer.score(ground_truth, prediction)
        return {
            'rouge1_f': scores['rouge1'].fmeasure,
            'rouge1_p': scores['rouge1'].precision,
            'rouge1_r': scores['rouge1'].recall,
            'rouge2_f': scores['rouge2'].fmeasure,
            'rougeL_f': scores['rougeL'].fmeasure,
        }
    
    def _calculate_bleu_score(self, ground_truth: str, prediction: str) -> Dict[str, float]:
        """Calculate BLEU score"""
        try:
            bleu = self.bleu_scorer.sentence_score(prediction, [ground_truth])
            return {
                'bleu': bleu.score / 100.0,  # Normalize to 0-1
                'bleu_1': bleu.precisions[0] if len(bleu.precisions) > 0 else 0.0,
                'bleu_2': bleu.precisions[1] if len(bleu.precisions) > 1 else 0.0,
            }
        except:
            return {'bleu': 0.0, 'bleu_1': 0.0, 'bleu_2': 0.0}
    
    def _calculate_word_overlap(self, ground_truth: str, prediction: str) -> Dict[str, float]:
        """Calculate word-level overlap metrics"""
        gt_words = set(ground_truth.lower().split())
        pred_words = set(prediction.lower().split())
        
        intersection = gt_words & pred_words
        union = gt_words | pred_words
        
        precision = len(intersection) / len(pred_words) if pred_words else 0.0
        recall = len(intersection) / len(gt_words) if gt_words else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        jaccard = len(intersection) / len(union) if union else 0.0
        
        return {
            'word_overlap_precision': precision,
            'word_overlap_recall': recall,
            'word_overlap_f1': f1,
            'jaccard_similarity': jaccard
        }
    
    def _calculate_semantic_similarity(self, ground_truth: str, prediction: str) -> Dict[str, float]:
        """Calculate semantic similarity using sentence embeddings"""
        try:
            embeddings = self.sentence_model.encode([ground_truth, prediction])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return {'semantic_similarity': float(similarity)}
        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {e}")
            return {'semantic_similarity': 0.0}
    
    def _calculate_bert_score(self, ground_truth: str, prediction: str) -> Dict[str, float]:
        """Calculate BERTScore"""
        try:
            P, R, F1 = bert_score([prediction], [ground_truth], lang="en", verbose=False)
            return {
                'bert_score_precision': float(P[0]),
                'bert_score_recall': float(R[0]),
                'bert_score_f1': float(F1[0])
            }
        except Exception as e:
            logger.warning(f"BERTScore calculation failed: {e}")
            return {'bert_score_precision': 0.0, 'bert_score_recall': 0.0, 'bert_score_f1': 0.0}
    
    def _calculate_financial_metrics(self, ground_truth: str, prediction: str) -> Dict[str, float]:
        """Calculate financial domain-specific metrics"""
        # Financial keywords
        financial_terms = {
            'revenue', 'sales', 'income', 'earnings', 'profit', 'loss',
            'investment', 'capital', 'funding', 'buyback', 'dividend',
            'market', 'growth', 'expansion', 'performance', 'metrics',
            'clients', 'customers', 'franchise', 'outstanding', 'shares'
        }
        
        gt_lower = ground_truth.lower()
        pred_lower = prediction.lower()
        
        # Count financial terms
        gt_financial_terms = sum(1 for term in financial_terms if term in gt_lower)
        pred_financial_terms = sum(1 for term in financial_terms if term in pred_lower)
        
        # Financial term coverage
        if gt_financial_terms > 0:
            financial_recall = pred_financial_terms / gt_financial_terms
        else:
            financial_recall = 1.0 if pred_financial_terms == 0 else 0.0
        
        # Number preservation (important for financial data)
        gt_numbers = self._extract_numbers(ground_truth)
        pred_numbers = self._extract_numbers(prediction)
        
        number_accuracy = len(set(gt_numbers) & set(pred_numbers)) / max(len(set(gt_numbers)), 1)
        
        return {
            'financial_term_recall': financial_recall,
            'financial_term_count_ratio': pred_financial_terms / max(gt_financial_terms, 1),
            'number_accuracy': number_accuracy
        }
    
    def _extract_numbers(self, text: str) -> List[str]:
        """Extract numbers from text"""
        import re
        # Extract numbers (including percentages, decimals)
        pattern = r'\b\d+(?:\.\d+)?(?:%|\b)'
        return re.findall(pattern, text)

class FinancialLLMEvaluator:
    """
    Comprehensive evaluator for financial domain LLM models.
    
    Features:
    - Multiple evaluation metrics (ROUGE, BLEU, BERTScore, semantic similarity)
    - Financial domain-specific metrics
    - Comparative analysis between models
    - Statistical significance testing
    - Comprehensive reporting and visualization
    """
    
    def __init__(self, config: EvaluationConfig):
        """Initialize evaluator with configuration."""
        self.config = config
        self.metric_calculator = MetricCalculator()
        self.statistical_analyzer = StatisticalAnalyzer()
        
        logger.info("Initialized FinancialLLMEvaluator")
    
    def evaluate_model(self, 
                      model_adapter: LlamaFinancialAdapter,
                      eval_dataset: Dataset,
                      model_name: str = "model") -> Tuple[List[EvaluationResult], AggregatedResults]:
        """
        Evaluate a single model on the given dataset.
        
        Args:
            model_adapter: Model adapter for inference
            eval_dataset: Evaluation dataset
            model_name: Name for the model (for logging)
            
        Returns:
            Tuple of (individual_results, aggregated_results)
        """
        logger.info(f"Evaluating model '{model_name}' on {len(eval_dataset)} samples")
        
        # Limit samples if configured
        eval_samples = min(len(eval_dataset), self.config.full_eval_samples)
        eval_subset = eval_dataset.select(range(eval_samples))
        
        # Generate predictions
        predictions = self._generate_predictions(model_adapter, eval_subset)
        
        # Calculate metrics for each sample
        individual_results = []
        
        for i, (sample, prediction) in enumerate(zip(eval_subset, predictions)):
            ground_truth = sample.get('output', sample.get('transcript', ''))
            
            # Calculate metrics
            metrics = self.metric_calculator.calculate_all_metrics(ground_truth, prediction)
            
            # Create result
            result = EvaluationResult(
                sample_id=f"{model_name}_{i}",
                ground_truth=ground_truth,
                prediction=prediction,
                metrics=metrics,
                metadata={
                    'model_name': model_name,
                    'sample_index': i,
                    'ground_truth_length': len(ground_truth.split()),
                    'prediction_length': len(prediction.split())
                }
            )
            
            individual_results.append(result)
        
        # Aggregate results
        aggregated_results = self._aggregate_results(individual_results)
        
        logger.info(f"Evaluation complete for '{model_name}'")
        return individual_results, aggregated_results
    
    def compare_models(self,
                      baseline_results: Tuple[List[EvaluationResult], AggregatedResults],
                      finetuned_results: Tuple[List[EvaluationResult], AggregatedResults],
                      output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare baseline and fine-tuned models.
        
        Args:
            baseline_results: Results from baseline model
            finetuned_results: Results from fine-tuned model
            output_dir: Optional directory to save comparison results
            
        Returns:
            Comprehensive comparison analysis
        """
        logger.info("Performing comparative analysis...")
        
        baseline_individual, baseline_agg = baseline_results
        finetuned_individual, finetuned_agg = finetuned_results
        
        # Statistical comparison
        comparison = self._perform_statistical_comparison(
            baseline_individual, finetuned_individual
        )
        
        # Generate improvement metrics
        improvements = self._calculate_improvements(baseline_agg, finetuned_agg)
        
        # Qualitative analysis
        qualitative_analysis = self._perform_qualitative_analysis(
            baseline_individual, finetuned_individual
        )
        
        # Create comprehensive comparison report
        comparison_report = {
            'summary': {
                'baseline_samples': len(baseline_individual),
                'finetuned_samples': len(finetuned_individual),
                'evaluation_timestamp': time.time(),
            },
            'statistical_comparison': comparison,
            'improvements': improvements,
            'qualitative_analysis': qualitative_analysis,
            'baseline_metrics': baseline_agg.to_dict(),
            'finetuned_metrics': finetuned_agg.to_dict()
        }
        
        # Save results if output directory provided
        if output_dir:
            self._save_comparison_results(comparison_report, output_dir)
            self._create_visualizations(comparison_report, output_dir)
        
        logger.info("Comparative analysis complete")
        return comparison_report
    
    def _generate_predictions(self, 
                            model_adapter: LlamaFinancialAdapter, 
                            dataset: Dataset) -> List[str]:
        """Generate predictions for the dataset."""
        logger.info("Generating predictions...")
        
        # Get model and tokenizer
        model = model_adapter.get_model_for_training()
        tokenizer = model_adapter.tokenizer
        
        # Create generation pipeline
        if model_adapter.device.type == "cuda":
            device_arg = model_adapter.device.index if hasattr(model_adapter.device, 'index') else 0
            torch_dtype = torch.float16
        elif model_adapter.device.type == "mps":
            device_arg = -1  # Use CPU for pipeline, model is already on MPS
            torch_dtype = torch.float32
        else:
            device_arg = -1
            torch_dtype = torch.float32
            
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device_arg,
            torch_dtype=torch_dtype
        )
        
        predictions = []
        
        for sample in dataset:
            # Format input prompt
            instruction = sample.get('instruction', 'What was said in this business call excerpt?')
            input_text = sample.get('input', '')
            
            if input_text:
                prompt = f"{instruction}\n\n{input_text}\n\nResponse:"
            else:
                prompt = f"{instruction}\n\nResponse:"
            
            try:
                # Generate prediction
                outputs = generator(
                    prompt,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=self.config.do_sample,
                    repetition_penalty=self.config.repetition_penalty,
                    pad_token_id=tokenizer.eos_token_id,
                    return_full_text=False
                )
                
                prediction = outputs[0]['generated_text'].strip()
                
            except Exception as e:
                logger.warning(f"Prediction generation failed: {e}")
                prediction = ""
            
            predictions.append(prediction)
        
        logger.info(f"Generated {len(predictions)} predictions")
        return predictions
    
    def _aggregate_results(self, results: List[EvaluationResult]) -> AggregatedResults:
        """Aggregate individual results into summary statistics."""
        if not results:
            return AggregatedResults(0, {})
        
        # Collect all metrics
        all_metrics = defaultdict(list)
        for result in results:
            for metric_name, value in result.metrics.items():
                all_metrics[metric_name].append(value)
        
        # Calculate summary statistics
        metrics_summary = {}
        for metric_name, values in all_metrics.items():
            values_array = np.array(values)
            metrics_summary[metric_name] = {
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array)),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'median': float(np.median(values_array)),
                'q25': float(np.percentile(values_array, 25)),
                'q75': float(np.percentile(values_array, 75))
            }
        
        return AggregatedResults(
            total_samples=len(results),
            metrics_summary=metrics_summary
        )
    
    def _perform_statistical_comparison(self, 
                                      baseline_results: List[EvaluationResult],
                                      finetuned_results: List[EvaluationResult]) -> Dict[str, Any]:
        """Perform statistical significance testing."""
        logger.info("Performing statistical significance testing...")
        
        # Ensure same number of samples
        min_samples = min(len(baseline_results), len(finetuned_results))
        baseline_subset = baseline_results[:min_samples]
        finetuned_subset = finetuned_results[:min_samples]
        
        # Collect metrics for comparison
        comparison_results = {}
        
        # Get all metric names
        metric_names = set()
        for result in baseline_subset + finetuned_subset:
            metric_names.update(result.metrics.keys())
        
        for metric_name in metric_names:
            baseline_values = [r.metrics.get(metric_name, 0.0) for r in baseline_subset]
            finetuned_values = [r.metrics.get(metric_name, 0.0) for r in finetuned_subset]
            
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(finetuned_values, baseline_values)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(baseline_values) + np.var(finetuned_values)) / 2)
            cohens_d = (np.mean(finetuned_values) - np.mean(baseline_values)) / pooled_std if pooled_std > 0 else 0
            
            # Wilcoxon signed-rank test (non-parametric alternative)
            try:
                wilcoxon_stat, wilcoxon_p = stats.wilcoxon(finetuned_values, baseline_values)
            except:
                wilcoxon_stat, wilcoxon_p = None, None
            
            comparison_results[metric_name] = {
                'baseline_mean': np.mean(baseline_values),
                'finetuned_mean': np.mean(finetuned_values),
                'difference': np.mean(finetuned_values) - np.mean(baseline_values),
                'relative_improvement': ((np.mean(finetuned_values) - np.mean(baseline_values)) / 
                                       max(np.mean(baseline_values), 1e-8)) * 100,
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'cohens_d': float(cohens_d),
                'effect_size': self._interpret_effect_size(abs(cohens_d)),
                'wilcoxon_p': float(wilcoxon_p) if wilcoxon_p is not None else None
            }
        
        return comparison_results
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""
        if cohens_d < 0.2:
            return "negligible"
        elif cohens_d < 0.5:
            return "small"
        elif cohens_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _calculate_improvements(self, 
                              baseline_agg: AggregatedResults,
                              finetuned_agg: AggregatedResults) -> Dict[str, float]:
        """Calculate percentage improvements."""
        improvements = {}
        
        for metric_name in baseline_agg.metrics_summary:
            if metric_name in finetuned_agg.metrics_summary:
                baseline_mean = baseline_agg.metrics_summary[metric_name]['mean']
                finetuned_mean = finetuned_agg.metrics_summary[metric_name]['mean']
                
                if baseline_mean > 0:
                    improvement = ((finetuned_mean - baseline_mean) / baseline_mean) * 100
                    improvements[metric_name] = improvement
        
        return improvements
    
    def _perform_qualitative_analysis(self,
                                    baseline_results: List[EvaluationResult],
                                    finetuned_results: List[EvaluationResult]) -> Dict[str, Any]:
        """Perform qualitative analysis of improvements."""
        # Sample analysis on subset
        analysis_samples = min(self.config.qualitative_eval_samples, len(baseline_results))
        
        qualitative_examples = []
        for i in range(analysis_samples):
            baseline_result = baseline_results[i]
            finetuned_result = finetuned_results[i]
            
            # Calculate improvement scores
            improvements = {}
            for metric in baseline_result.metrics:
                if metric in finetuned_result.metrics:
                    baseline_score = baseline_result.metrics[metric]
                    finetuned_score = finetuned_result.metrics[metric]
                    improvements[metric] = finetuned_score - baseline_score
            
            qualitative_examples.append({
                'sample_id': i,
                'ground_truth': baseline_result.ground_truth,
                'baseline_prediction': baseline_result.prediction,
                'finetuned_prediction': finetuned_result.prediction,
                'improvements': improvements,
                'overall_improvement': np.mean(list(improvements.values()))
            })
        
        # Sort by overall improvement
        qualitative_examples.sort(key=lambda x: x['overall_improvement'], reverse=True)
        
        return {
            'total_analyzed': analysis_samples,
            'best_improvements': qualitative_examples[:5],
            'worst_improvements': qualitative_examples[-5:],
            'examples': qualitative_examples
        }
    
    def _save_comparison_results(self, comparison_report: Dict[str, Any], output_dir: str):
        """Save comparison results to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save full report
        with open(output_path / "comparison_report.json", 'w') as f:
            json.dump(comparison_report, f, indent=2, default=str)
        
        # Save summary table
        summary_data = []
        for metric, stats in comparison_report['statistical_comparison'].items():
            summary_data.append({
                'metric': metric,
                'baseline_mean': stats['baseline_mean'],
                'finetuned_mean': stats['finetuned_mean'],
                'improvement_%': stats['relative_improvement'],
                'p_value': stats['p_value'],
                'significant': stats['significant'],
                'effect_size': stats['effect_size']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_path / "metrics_comparison.csv", index=False)
        
        logger.info(f"Comparison results saved to {output_path}")
    
    def _create_visualizations(self, comparison_report: Dict[str, Any], output_dir: str):
        """Create visualization plots for the comparison."""
        output_path = Path(output_dir)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Metrics improvement bar chart
        self._plot_metrics_comparison(comparison_report, output_path)
        
        # 2. Statistical significance heatmap
        self._plot_significance_heatmap(comparison_report, output_path)
        
        # 3. Distribution comparison for key metrics
        self._plot_metric_distributions(comparison_report, output_path)
        
        logger.info(f"Visualizations saved to {output_path}")
    
    def _plot_metrics_comparison(self, comparison_report: Dict[str, Any], output_path: Path):
        """Plot metrics comparison bar chart."""
        stats = comparison_report['statistical_comparison']
        
        metrics = list(stats.keys())
        improvements = [stats[m]['relative_improvement'] for m in metrics]
        significant = [stats[m]['significant'] for m in metrics]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Color bars based on significance
        colors = ['green' if sig else 'orange' for sig in significant]
        bars = ax.bar(metrics, improvements, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5 if height >= 0 else height - 1,
                   f'{imp:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        ax.set_title('Model Performance Improvement by Metric', fontsize=16, fontweight='bold')
        ax.set_ylabel('Relative Improvement (%)', fontsize=12)
        ax.set_xlabel('Metrics', fontsize=12)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.7, label='Statistically Significant'),
                          Patch(facecolor='orange', alpha=0.7, label='Not Significant')]
        ax.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.savefig(output_path / "metrics_improvement.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_significance_heatmap(self, comparison_report: Dict[str, Any], output_path: Path):
        """Plot statistical significance heatmap."""
        stats = comparison_report['statistical_comparison']
        
        # Create data for heatmap
        metrics = list(stats.keys())
        p_values = [stats[m]['p_value'] for m in metrics]
        effect_sizes = [abs(stats[m]['cohens_d']) for m in metrics]
        
        # Create significance matrix
        data = np.array([p_values, effect_sizes]).T
        
        fig, ax = plt.subplots(figsize=(8, 10))
        
        # Create heatmap
        sns.heatmap(data, 
                   xticklabels=['P-value', 'Effect Size (|Cohen\'s d|)'],
                   yticklabels=metrics,
                   annot=True, 
                   fmt='.3f',
                   cmap='RdYlBu_r',
                   ax=ax)
        
        ax.set_title('Statistical Significance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / "significance_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metric_distributions(self, comparison_report: Dict[str, Any], output_path: Path):
        """Plot distribution comparison for key metrics."""
        # This would require access to individual results, which we'd need to modify the structure for
        # For now, create a placeholder plot
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Metric Distribution Plots\n(Implementation pending individual results access)', 
               ha='center', va='center', fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.savefig(output_path / "metric_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def quick_evaluate(self, 
                      model_adapter: LlamaFinancialAdapter,
                      eval_dataset: Dataset) -> Dict[str, float]:
        """Quick evaluation with limited samples for fast feedback."""
        logger.info(f"Running quick evaluation with {self.config.quick_eval_samples} samples")
        
        # Limit to quick eval samples
        eval_subset = eval_dataset.select(range(min(self.config.quick_eval_samples, len(eval_dataset))))
        
        # Generate predictions
        predictions = self._generate_predictions(model_adapter, eval_subset)
        
        # Calculate basic metrics
        total_metrics = defaultdict(list)
        
        for sample, prediction in zip(eval_subset, predictions):
            ground_truth = sample.get('output', sample.get('transcript', ''))
            metrics = self.metric_calculator.calculate_all_metrics(ground_truth, prediction)
            
            for metric_name, value in metrics.items():
                total_metrics[metric_name].append(value)
        
        # Average metrics
        avg_metrics = {
            metric_name: np.mean(values) 
            for metric_name, values in total_metrics.items()
        }
        
        logger.info("Quick evaluation complete")
        return avg_metrics