import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class ICDEvaluator:
    """
    Comprehensive ICD-10 code evaluation module with exact match evaluation metrics
    and analysis functions.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the evaluator with a DataFrame containing predictions and references.
        
        Args:
            df: DataFrame with columns 'entries' (predictions) and 'reference_answer' (ground truth)
        """
        self.df = df.copy()
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Preprocess the data to extract codes and handle empty entries."""
        self.processed_data = []
        
        for index, row in self.df.iterrows():
            try:
                # Check if entries is empty
                entries_eval = ast.literal_eval(row['entries'])
                if len(entries_eval) == 0:
                    self.processed_data.append({
                        'index': index,
                        'predicted_codes': [],
                        'reference_codes': [],
                        'is_empty': True
                    })
                    continue
                
                # Extract predicted and reference codes
                predicted_codes = self._get_pred_codes(row['entries'])
                reference_codes = self._get_reference_codes(row['reference_answer'])
                
                self.processed_data.append({
                    'index': index,
                    'predicted_codes': predicted_codes,
                    'reference_codes': reference_codes,
                    'is_empty': False
                })
                
            except Exception as e:
                print(f"Error processing row {index}: {e}")
                self.processed_data.append({
                    'index': index,
                    'predicted_codes': [],
                    'reference_codes': [],
                    'is_empty': True
                })
    
    def _get_pred_codes(self, entries_string: str) -> List[str]:
        """Extract ICD codes from prediction entries."""
        pred_codes = []
        try:
            cond = ast.literal_eval(entries_string)
            for item in cond:
                pred_codes.append(item["icd10_code"])
        except:
            pass
        return pred_codes
    
    def _get_reference_codes(self, reference_string: str) -> List[str]:
        """Extract ICD codes from reference answer."""
        reference_codes = []
        try:
            lines = reference_string.split("\n")
            for line in lines:
                if line.strip():
                    reference_codes.append(line.split(" ")[0])
        except:
            pass
        return reference_codes

    def compute_precision_recall_f1(self, predicted_codes: List[str], 
                                   reference_codes: List[str]) -> Tuple[float, float, float]:
        """
        Compute precision, recall, and F1 score for ICD-10 code predictions.
        
        Args:
            predicted_codes: List of predicted ICD-10 codes
            reference_codes: List of reference/ground truth ICD-10 codes
        
        Returns:
            tuple: (precision, recall, f1_score)
        """
        pred_set = set(predicted_codes)
        ref_set = set(reference_codes)
        
        # Calculate true positives
        true_positives = len(pred_set.intersection(ref_set))
        
        # Calculate precision: TP / (TP + FP) = TP / total_predicted
        precision = true_positives / len(pred_set) if len(pred_set) > 0 else 0
        
        # Calculate recall: TP / (TP + FN) = TP / total_reference
        recall = true_positives / len(ref_set) if len(ref_set) > 0 else 0
        
        # Calculate F1 score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1_score
    
    def evaluate_per_prediction(self) -> Dict[str, Any]:
        """
        Evaluate precision, recall, F1 for each prediction (exact match).
        
        Returns:
            Dictionary with evaluation results and individual metrics
        """
        all_precisions = []
        all_recalls = []
        all_f1_scores = []
        detailed_results = []
        
        for data in self.processed_data:
            if data['is_empty']:
                continue
                
            precision, recall, f1 = self.compute_precision_recall_f1(
                data['predicted_codes'], data['reference_codes']
            )
            
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1_scores.append(f1)
            
            # Store detailed results
            pred_set = set(data['predicted_codes'])
            ref_set = set(data['reference_codes'])
            matched = pred_set.intersection(ref_set)
            
            detailed_results.append({
                'index': data['index'],
                'predicted_codes': data['predicted_codes'],
                'reference_codes': data['reference_codes'],
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'matched_codes': list(matched),
                'missed_codes': list(ref_set - pred_set),
                'extra_codes': list(pred_set - ref_set)
            })
        
        return {
            'avg_precision': np.mean(all_precisions) if all_precisions else 0,
            'avg_recall': np.mean(all_recalls) if all_recalls else 0,
            'avg_f1': np.mean(all_f1_scores) if all_f1_scores else 0,
            'individual_precisions': all_precisions,
            'individual_recalls': all_recalls,
            'individual_f1_scores': all_f1_scores,
            'detailed_results': detailed_results,
            'processed_count': len([d for d in self.processed_data if not d['is_empty']]),
            'empty_count': len([d for d in self.processed_data if d['is_empty']])
        }

    def evaluate_top_frequent_conditions(self, top_k: int = 10) -> Dict[str, Any]:
        """
        3. Evaluate performance on the top-k most frequent conditions.
        
        Args:
            top_k: Number of top frequent conditions to analyze
        
        Returns:
            Dictionary with evaluation results for top frequent conditions
        """
        # Collect all reference codes to find most frequent
        all_reference_codes = []
        for data in self.processed_data:
            if not data['is_empty']:
                all_reference_codes.extend(data['reference_codes'])
        
        # Get top-k most frequent codes
        code_counts = Counter(all_reference_codes)
        top_codes = [code for code, count in code_counts.most_common(top_k)]
        
        # Evaluate performance for each top code
        code_performances = {}
        for code in top_codes:
            # Collect all examples where this code appears in reference
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            total_occurrences = 0
            
            for data in self.processed_data:
                if data['is_empty']:
                    continue
                    
                # Check if this code appears in reference and/or predictions
                in_reference = code in data['reference_codes']
                in_prediction = code in data['predicted_codes']
                
                if in_reference:
                    total_occurrences += 1
                    if in_prediction:
                        true_positives += 1
                    else:
                        false_negatives += 1
                elif in_prediction:
                    false_positives += 1
            
            # Calculate metrics for this specific code
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            code_performances[code] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'occurrences': total_occurrences,
                'frequency_rank': top_codes.index(code) + 1,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives
            }
        
        return {
            'top_codes': top_codes,
            'code_counts': dict(code_counts.most_common(top_k)),
            'code_performances': code_performances,
            'total_reference_codes': len(all_reference_codes),
            'unique_reference_codes': len(set(all_reference_codes))
        }
    
    def analyze_example_by_index(self, index: int) -> Dict[str, Any]:
        """
        4. Analyze a specific example by index.
        
        Args:
            index: Index of the example to analyze
        
        Returns:
            Dictionary with detailed analysis of the example
        """
        if index >= len(self.processed_data):
            return {'error': f'Index {index} out of range'}
        
        data = self.processed_data[index]
        
        if data['is_empty']:
            return {
                'index': index,
                'error': 'Empty prediction/reference',
                'predicted_codes': [],
                'reference_codes': []
            }
        
        # Exact match evaluation
        exact_precision, exact_recall, exact_f1 = self.compute_precision_recall_f1(
            data['predicted_codes'], data['reference_codes']
        )
        
        pred_set = set(data['predicted_codes'])
        ref_set = set(data['reference_codes'])
        
        return {
            'index': index,
            'predicted_codes': data['predicted_codes'],
            'reference_codes': data['reference_codes'],
            'exact_match': {
                'precision': exact_precision,
                'recall': exact_recall,
                'f1': exact_f1,
                'matched_codes': list(pred_set.intersection(ref_set)),
                'missed_codes': list(ref_set - pred_set),
                'extra_codes': list(pred_set - ref_set)
            }
        }
    
    def analyze_example_by_icd_code(self, icd_code: str) -> Dict[str, Any]:
        """
        5. Analyze examples containing a specific ICD code.
        
        Args:
            icd_code: ICD code to search for
        
        Returns:
            Dictionary with analysis of examples containing the specified code
        """
        matching_examples = []
        code_in_predictions = 0
        code_in_references = 0
        correct_predictions = 0
        
        for data in self.processed_data:
            if data['is_empty']:
                continue
            
            # Check if code appears in reference or predictions
            in_reference = icd_code in data['reference_codes']
            in_prediction = icd_code in data['predicted_codes']
            
            if in_reference or in_prediction:
                # Analyze this example
                precision, recall, f1 = self.compute_precision_recall_f1(
                    data['predicted_codes'], data['reference_codes']
                )
                
                example_analysis = {
                    'index': data['index'],
                    'predicted_codes': data['predicted_codes'],
                    'reference_codes': data['reference_codes'],
                    'code_in_reference': in_reference,
                    'code_in_prediction': in_prediction,
                    'code_correctly_predicted': in_reference and in_prediction,
                    'overall_precision': precision,
                    'overall_recall': recall,
                    'overall_f1': f1
                }
                
                matching_examples.append(example_analysis)
                
                # Update counters
                if in_reference:
                    code_in_references += 1
                if in_prediction:
                    code_in_predictions += 1
                if in_reference and in_prediction:
                    correct_predictions += 1
        
        # Calculate code-specific metrics
        precision_for_code = correct_predictions / code_in_predictions if code_in_predictions > 0 else 0
        recall_for_code = correct_predictions / code_in_references if code_in_references > 0 else 0
        f1_for_code = 2 * (precision_for_code * recall_for_code) / (precision_for_code + recall_for_code) if (precision_for_code + recall_for_code) > 0 else 0
        
        return {
            'icd_code': icd_code,
            'total_examples_found': len(matching_examples),
            'examples': matching_examples,
            'summary': {
                'code_in_references': code_in_references,
                'code_in_predictions': code_in_predictions,
                'correct_predictions': correct_predictions,
                'precision_for_code': precision_for_code,
                'recall_for_code': recall_for_code,
                'f1_for_code': f1_for_code
            }
        }
    
    def plot_evaluation_results(self, results: Dict[str, Any], title: str = "Evaluation Results"):
        """
        Plot evaluation results with precision, recall, and F1 scores.
        
        Args:
            results: Results dictionary from evaluate_per_prediction()
            title: Title for the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Average metrics bar plot
        metrics = ['Precision', 'Recall', 'F1']
        values = [results['avg_precision'], results['avg_recall'], results['avg_f1']]
        colors = ['skyblue', 'lightgreen', 'lightcoral']
        
        bars = axes[0, 0].bar(metrics, values, color=colors, alpha=0.8, edgecolor='black')
        axes[0, 0].set_title('Average Metrics', fontweight='bold')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Distribution of individual F1 scores
        f1_scores = results['individual_f1_scores']
        axes[0, 1].hist(f1_scores, bins=20, color='lightcoral', alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Distribution of F1 Scores', fontweight='bold')
        axes[0, 1].set_xlabel('F1 Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(np.mean(f1_scores), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(f1_scores):.3f}')
        axes[0, 1].legend()
        
        # 3. Precision vs Recall scatter plot
        precisions = results['individual_precisions']
        recalls = results['individual_recalls']
        scatter = axes[1, 0].scatter(recalls, precisions, alpha=0.6, c=f1_scores, 
                                   cmap='viridis', s=30)
        axes[1, 0].set_title('Precision vs Recall', fontweight='bold')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].set_ylim(0, 1)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=axes[1, 0])
        cbar.set_label('F1 Score')
        
        # 4. Summary statistics
        axes[1, 1].axis('off')
        summary_text = f"""
        Summary Statistics:
        
        Total Processed: {results['processed_count']}
        Empty Entries: {results['empty_count']}
        
        Average Precision: {results['avg_precision']:.3f}
        Average Recall: {results['avg_recall']:.3f}
        Average F1: {results['avg_f1']:.3f}
        
        Std Precision: {np.std(precisions):.3f}
        Std Recall: {np.std(recalls):.3f}
        Std F1: {np.std(f1_scores):.3f}
        
        Max F1: {max(f1_scores) if f1_scores else 0:.3f}
        Min F1: {min(f1_scores) if f1_scores else 0:.3f}
        """
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def plot_top_frequent_conditions_scores(self, results: Dict[str, Any], top_k: int = 10):
        """
        Plot performance scores for the most frequent ICD conditions.
        
        Args:
            results: Results from evaluate_top_frequent_conditions()
            top_k: Number of top conditions to display
        """
        code_performances = results['code_performances']
        
        if not code_performances:
            print("No performance data available for frequent conditions.")
            return
        
        # Prepare data for plotting
        codes = list(code_performances.keys())[:top_k]
        precisions = [code_performances[code]['precision'] for code in codes]
        recalls = [code_performances[code]['recall'] for code in codes]
        f1_scores = [code_performances[code]['f1'] for code in codes]
        occurrences = [code_performances[code]['occurrences'] for code in codes]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Performance Analysis: Top {len(codes)} Most Frequent ICD Conditions', 
                     fontsize=16, fontweight='bold')
        
        # 1. Precision, Recall, F1 bar plot
        x = np.arange(len(codes))
        width = 0.25
        
        bars1 = axes[0, 0].bar(x - width, precisions, width, label='Precision', 
                              color='skyblue', alpha=0.8)
        bars2 = axes[0, 0].bar(x, recalls, width, label='Recall', 
                              color='lightgreen', alpha=0.8)
        bars3 = axes[0, 0].bar(x + width, f1_scores, width, label='F1', 
                              color='lightcoral', alpha=0.8)
        
        axes[0, 0].set_title('Precision, Recall, and F1 Scores by ICD Code', fontweight='bold')
        axes[0, 0].set_xlabel('ICD Code')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(codes, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, max(max(precisions), max(recalls), max(f1_scores)) * 1.1)
        
        # 2. Frequency (occurrences) bar plot
        bars = axes[0, 1].bar(codes, occurrences, color='gold', alpha=0.8, edgecolor='black')
        axes[0, 1].set_title('Frequency of ICD Codes in Dataset', fontweight='bold')
        axes[0, 1].set_xlabel('ICD Code')
        axes[0, 1].set_ylabel('Number of Occurrences')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, occurrences):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           str(value), ha='center', va='bottom', fontweight='bold')
        
        # 3. F1 Score vs Frequency scatter plot
        scatter = axes[1, 0].scatter(occurrences, f1_scores, s=100, alpha=0.7, 
                                   c=range(len(codes)), cmap='viridis')
        axes[1, 0].set_title('F1 Score vs Frequency', fontweight='bold')
        axes[1, 0].set_xlabel('Number of Occurrences')
        axes[1, 0].set_ylabel('F1 Score')
        
        # Add code labels to points
        for i, code in enumerate(codes):
            axes[1, 0].annotate(code, (occurrences[i], f1_scores[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 4. Performance heatmap
        performance_matrix = np.array([precisions, recalls, f1_scores])
        im = axes[1, 1].imshow(performance_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        axes[1, 1].set_title('Performance Heatmap', fontweight='bold')
        axes[1, 1].set_xticks(range(len(codes)))
        axes[1, 1].set_xticklabels(codes, rotation=45, ha='right')
        axes[1, 1].set_yticks([0, 1, 2])
        axes[1, 1].set_yticklabels(['Precision', 'Recall', 'F1'])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[1, 1])
        cbar.set_label('Score')
        
        # Add text annotations to heatmap
        for i in range(3):
            for j in range(len(codes)):
                text = axes[1, 1].text(j, i, f'{performance_matrix[i, j]:.2f}',
                                     ha="center", va="center", color="black", fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def print_top_conditions_analysis(self, results: Dict[str, Any], top_k: int = 10):
        """
        Print detailed analysis of top frequent conditions.
        
        Args:
            results: Results from evaluate_top_frequent_conditions()
            top_k: Number of top conditions to display
        """
        print("\n" + "="*80)
        print("PERFORMANCE INSIGHTS FOR MOST FREQUENT ICD CATEGORIES")
        print("="*80)
        
        code_performances = results['code_performances']
        
        if not code_performances:
            print("No performance data available.")
            return
        
        # Sort by frequency (occurrences)
        sorted_codes = sorted(code_performances.keys(), 
                            key=lambda x: code_performances[x]['occurrences'], reverse=True)[:top_k]
        
        # Calculate summary statistics
        all_precisions = [code_performances[code]['precision'] for code in sorted_codes]
        all_recalls = [code_performances[code]['recall'] for code in sorted_codes]
        all_f1s = [code_performances[code]['f1'] for code in sorted_codes]
        
        perfect_precision_count = sum(1 for p in all_precisions if p == 1.0)
        zero_performance_count = sum(1 for f1 in all_f1s if f1 == 0.0)
        high_recall_count = sum(1 for r in all_recalls if r > 1.0)
        
        best_f1_code = max(sorted_codes, key=lambda x: code_performances[x]['f1'])
        worst_f1_code = min(sorted_codes, key=lambda x: code_performances[x]['f1'])
        
        total_occurrences = sum(code_performances[code]['occurrences'] for code in sorted_codes)
        most_frequent = max(sorted_codes, key=lambda x: code_performances[x]['occurrences'])
        least_frequent = min(sorted_codes, key=lambda x: code_performances[x]['occurrences'])
        
        print("\nKey Observations:")
        print(f"• {perfect_precision_count}/{len(sorted_codes)} categories show perfect precision (1.000)")
        print(f"• {zero_performance_count}/{len(sorted_codes)} categories show zero performance across all metrics")
        print(f"• {high_recall_count}/{len(sorted_codes)} categories show recall > 1.0 (model over-predicting)")
        print(f"• Best performing category: {best_f1_code} (F1: {code_performances[best_f1_code]['f1']:.3f})")
        print(f"• Worst performing category: {worst_f1_code} (F1: {code_performances[worst_f1_code]['f1']:.3f})")
        print(f"• Top {len(sorted_codes)} categories represent {total_occurrences} total occurrences")
        print(f"• Most frequent: {most_frequent} ({code_performances[most_frequent]['occurrences']} occurrences)")
        print(f"• Least frequent in top {len(sorted_codes)}: {least_frequent} ({code_performances[least_frequent]['occurrences']} occurrences)")
    
    def print_example_analysis(self, analysis: Dict[str, Any]):
        """
        Print detailed analysis of a specific example.
        
        Args:
            analysis: Results from analyze_example_by_index()
        """
        if 'error' in analysis:
            print(f"Error analyzing example {analysis.get('index', 'unknown')}: {analysis['error']}")
            return
        
        print("\n" + "="*80)
        print(f"DETAILED ANALYSIS FOR EXAMPLE {analysis['index']}")
        print("="*80)
        
        print(f"\nPredicted Codes ({len(analysis['predicted_codes'])}): {analysis['predicted_codes']}")
        print(f"Reference Codes ({len(analysis['reference_codes'])}): {analysis['reference_codes']}")
        
        # Exact Match Analysis
        exact = analysis['exact_match']
        print(f"\n--- EXACT MATCH EVALUATION ---")
        print(f"Precision: {exact['precision']:.3f}")
        print(f"Recall: {exact['recall']:.3f}")
        print(f"F1 Score: {exact['f1']:.3f}")
        print(f"Correctly Predicted: {exact['matched_codes']}")
        print(f"Missed (False Negatives): {exact['missed_codes']}")
        print(f"Extra (False Positives): {exact['extra_codes']}")
    
    def analyze_confidence_patterns(self) -> Dict[str, Any]:
        """
        Analyze confidence patterns in predictions.
        
        Returns:
            Dictionary with confidence analysis results
        """
        print("Analyzing confidence patterns in predictions...")
        
        confidence_data = []
        
        for data in self.processed_data:
            if data['is_empty']:
                continue
            
            try:
                # Get original entries to extract confidence scores
                row = self.df.iloc[data['index']]
                entries = ast.literal_eval(row['entries'])
                
                for entry in entries:
                    if 'confidence' in entry:
                        confidence_data.append({
                            'index': data['index'],
                            'icd_code': entry['icd10_code'],
                            'confidence': entry['confidence'],
                            'is_correct': entry['icd10_code'] in data['reference_codes']
                        })
            except Exception as e:
                print(f"Error processing confidence for index {data['index']}: {e}")
                continue
        
        if not confidence_data:
            return {'error': 'No confidence data found'}
        
        # Convert to DataFrame for easier analysis
        conf_df = pd.DataFrame(confidence_data)
        
        # Analyze confidence distributions by category
        confident_correct = len(conf_df[(conf_df['confidence'] == 'confident') & (conf_df['is_correct'] == True)])
        confident_incorrect = len(conf_df[(conf_df['confidence'] == 'confident') & (conf_df['is_correct'] == False)])
        review_correct = len(conf_df[(conf_df['confidence'] == 'requires_human_review') & (conf_df['is_correct'] == True)])
        review_incorrect = len(conf_df[(conf_df['confidence'] == 'requires_human_review') & (conf_df['is_correct'] == False)])
        
        # Calculate accuracy by confidence category
        confident_total = confident_correct + confident_incorrect
        review_total = review_correct + review_incorrect
        
        confident_accuracy = confident_correct / confident_total if confident_total > 0 else 0
        review_accuracy = review_correct / review_total if review_total > 0 else 0
        
        # Analyze by confidence category
        confidence_category_accuracy = conf_df.groupby('confidence')['is_correct'].agg(['mean', 'count']).reset_index()
        
        return {
            'total_predictions': len(conf_df),
            'correct_predictions': len(conf_df[conf_df['is_correct'] == True]),
            'incorrect_predictions': len(conf_df[conf_df['is_correct'] == False]),
            'confident_total': confident_total,
            'confident_correct': confident_correct,
            'confident_incorrect': confident_incorrect,
            'confident_accuracy': confident_accuracy,
            'review_total': review_total,
            'review_correct': review_correct,
            'review_incorrect': review_incorrect,
            'review_accuracy': review_accuracy,
            'confidence_data': conf_df,
            'confidence_category_accuracy': confidence_category_accuracy
        }
    
    def evaluate_by_confidence_level(self) -> Dict[str, Any]:
        """
        Evaluate model performance by confidence levels.
        
        Returns:
            Dictionary with confidence-based evaluation results
        """
        # First get the confidence patterns analysis
        confidence_patterns = self.analyze_confidence_patterns()
        
        if 'error' in confidence_patterns:
            return confidence_patterns
        
        # Group predictions by encounter (index)
        predictions_by_encounter = defaultdict(list)
        
        for data in self.processed_data:
            if data['is_empty']:
                continue
            
            try:
                # Get original entries to extract confidence scores
                row = self.df.iloc[data['index']]
                entries = ast.literal_eval(row['entries'])
                
                for entry in entries:
                    if 'confidence' in entry:
                        predictions_by_encounter[data['index']].append({
                            'icd_code': entry['icd10_code'],
                            'confidence': entry['confidence'],
                            'reference_codes': data['reference_codes']
                        })
            except Exception as e:
                print(f"Error processing confidence for index {data['index']}: {e}")
                continue
        
        # Evaluate different confidence categories
        confident_results = self._evaluate_subset_by_confidence_category(predictions_by_encounter, 'confident')
        review_results = self._evaluate_subset_by_confidence_category(predictions_by_encounter, 'requires_human_review')
        
        # Return the expected keys for the notebook
        return {
            'confident': confident_results,
            'requires_human_review': review_results,
            'total_encounters': len(predictions_by_encounter),
            # Add the expected keys from confidence patterns
            'total_predictions': confidence_patterns['total_predictions'],
            'correct_predictions': confidence_patterns['correct_predictions'],
            'confident_accuracy': confidence_patterns['confident_accuracy'],
            'review_accuracy': confidence_patterns['review_accuracy'],
            'confidence_data': confidence_patterns['confidence_data'],
            'confidence_category_accuracy': confidence_patterns['confidence_category_accuracy']
        }
    
    def _evaluate_subset_by_confidence_category(self, predictions_by_encounter: Dict[int, List[str]], 
                                                category: str) -> Dict[str, Any]:
        """
        Helper method to evaluate a subset of predictions based on confidence category.
        
        Args:
            predictions_by_encounter: Dictionary mapping encounter index to predictions
            category: 'confident' or 'requires_human_review'
        
        Returns:
            Dictionary with evaluation results for the confidence subset
        """
        subset_precisions = []
        subset_recalls = []
        subset_f1_scores = []
        encounter_count = 0
        total_predictions_in_category = 0
        
        for encounter_idx, predictions in predictions_by_encounter.items():
            # Filter predictions by confidence category
            filtered_preds = [p for p in predictions if p['confidence'] == category]
            
            if not filtered_preds:
                continue
            
            encounter_count += 1
            total_predictions_in_category += len(filtered_preds)
            
            # Extract codes for evaluation - ONLY evaluate the filtered predictions against reference
            pred_codes = [p['icd_code'] for p in filtered_preds]
            ref_codes = filtered_preds[0]['reference_codes']  # Same for all predictions in encounter
            
            precision, recall, f1 = self.compute_precision_recall_f1(pred_codes, ref_codes)
            
            subset_precisions.append(precision)
            subset_recalls.append(recall)
            subset_f1_scores.append(f1)
        
        return {
            'avg_precision': np.mean(subset_precisions) if subset_precisions else 0,
            'avg_recall': np.mean(subset_recalls) if subset_recalls else 0,
            'avg_f1': np.mean(subset_f1_scores) if subset_f1_scores else 0,
            'encounter_count': encounter_count,
            'total_predictions': total_predictions_in_category,
            'confidence_category': category
        }
    
    def plot_confidence_analysis(self, confidence_results: Dict[str, Any]):
        """
        Plot confidence analysis results for categorical confidence values.
        
        Args:
            confidence_results: Results from analyze_confidence_patterns()
        """
        if 'error' in confidence_results:
            print(f"Error: {confidence_results['error']}")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Confidence Analysis: Confident vs Requires Human Review', fontsize=16, fontweight='bold')
        
        # 1. Accuracy by confidence category
        categories = ['confident', 'requires_human_review']
        accuracies = [
            confidence_results['confident_accuracy'],
            confidence_results['review_accuracy']
        ]
        counts = [
            confidence_results['confident_total'],
            confidence_results['review_total']
        ]
        
        bars = axes[0, 0].bar(categories, accuracies, color=['green', 'orange'], alpha=0.7)
        axes[0, 0].set_title('Accuracy by Confidence Category')
        axes[0, 0].set_xlabel('Confidence Category')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f'n={count}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 2. Distribution of predictions by confidence category
        conf_df = confidence_results['confidence_data']
        category_counts = conf_df['confidence'].value_counts()
        
        colors = ['green' if cat == 'confident' else 'orange' for cat in category_counts.index]
        bars = axes[0, 1].bar(category_counts.index, category_counts.values, color=colors, alpha=0.7)
        axes[0, 1].set_title('Distribution of Predictions by Confidence')
        axes[0, 1].set_xlabel('Confidence Category')
        axes[0, 1].set_ylabel('Number of Predictions')
        
        # Add value labels on bars
        for bar, value in zip(bars, category_counts.values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           str(value), ha='center', va='bottom', fontweight='bold')
        
        # 3. Correctness breakdown by category
        confident_correct = confidence_results['confident_correct']
        confident_incorrect = confidence_results['confident_incorrect']
        review_correct = confidence_results['review_correct']
        review_incorrect = confidence_results['review_incorrect']
        
        categories = ['Confident', 'Requires Review']
        correct_counts = [confident_correct, review_correct]
        incorrect_counts = [confident_incorrect, review_incorrect]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = axes[1, 0].bar(x - width/2, correct_counts, width, label='Correct', color='green', alpha=0.7)
        bars2 = axes[1, 0].bar(x + width/2, incorrect_counts, width, label='Incorrect', color='red', alpha=0.7)
        
        axes[1, 0].set_title('Correct vs Incorrect by Confidence Category')
        axes[1, 0].set_xlabel('Confidence Category')
        axes[1, 0].set_ylabel('Number of Predictions')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(categories)
        axes[1, 0].legend()
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                               f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Summary statistics
        axes[1, 1].axis('off')
        
        total_preds = confidence_results['total_predictions']
        correct_preds = confidence_results['correct_predictions']
        overall_accuracy = correct_preds / total_preds if total_preds > 0 else 0
        
        summary_text = f"""
        Confidence Analysis Summary:
        
        Total Predictions: {total_preds}
        Overall Accuracy: {overall_accuracy:.3f}
        
        CONFIDENT Predictions:
        • Total: {confident_correct + confident_incorrect}
        • Correct: {confident_correct}
        • Incorrect: {confident_incorrect}
        • Accuracy: {confidence_results['confident_accuracy']:.3f}
        
        REQUIRES REVIEW Predictions:
        • Total: {review_correct + review_incorrect}
        • Correct: {review_correct}
        • Incorrect: {review_incorrect}
        • Accuracy: {confidence_results['review_accuracy']:.3f}
        
        Model Calibration:
        {'✓ Well calibrated' if confidence_results['confident_accuracy'] > confidence_results['review_accuracy'] else '✗ Poorly calibrated'}
        (Confident should be more accurate)
        """
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def plot_confidence_performance_metrics(self, confidence_results: Dict[str, Any]):
        """
        Plot precision, recall, F1 scores by confidence level and pie chart distribution.
        
        Args:
            confidence_results: Results from evaluate_by_confidence_level()
        """
        if 'error' in confidence_results:
            print(f"Error: {confidence_results['error']}")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance Metrics by Confidence Level', fontsize=16, fontweight='bold')
        
        # Extract data for confident and requires_human_review
        confident_data = confidence_results['confident']
        review_data = confidence_results['requires_human_review']
        
        # 1. Precision, Recall, F1 bar plot
        categories = ['Confident', 'Requires Review']
        precisions = [confident_data['avg_precision'], review_data['avg_precision']]
        recalls = [confident_data['avg_recall'], review_data['avg_recall']]
        f1_scores = [confident_data['avg_f1'], review_data['avg_f1']]
        
        x = np.arange(len(categories))
        width = 0.25
        
        bars1 = axes[0, 0].bar(x - width, precisions, width, label='Precision', 
                              color='skyblue', alpha=0.8)
        bars2 = axes[0, 0].bar(x, recalls, width, label='Recall', 
                              color='lightgreen', alpha=0.8)
        bars3 = axes[0, 0].bar(x + width, f1_scores, width, label='F1', 
                              color='lightcoral', alpha=0.8)
        
        axes[0, 0].set_title('Precision, Recall, and F1 Scores by Confidence Level', fontweight='bold')
        axes[0, 0].set_xlabel('Confidence Level')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(categories)
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0, max(max(precisions), max(recalls), max(f1_scores)) * 1.1)
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Pie chart of confidence distribution
        confident_total = confidence_results['confident']['total_predictions']
        review_total = confidence_results['requires_human_review']['total_predictions']
        
        sizes = [confident_total, review_total]
        labels = ['Confident', 'Requires Review']
        colors = ['lightgreen', 'lightcoral']
        explode = (0.05, 0.05)  # explode both slices slightly
        
        wedges, texts, autotexts = axes[0, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                                 explode=explode, shadow=True, startangle=90)
        axes[0, 1].set_title('Distribution of Predictions by Confidence Level', fontweight='bold')
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
        
        # 3. Encounter counts bar plot
        encounter_counts = [confident_data['encounter_count'], review_data['encounter_count']]
        bars = axes[1, 0].bar(categories, encounter_counts, color=['lightgreen', 'lightcoral'], alpha=0.8)
        axes[1, 0].set_title('Number of Encounters by Confidence Level', fontweight='bold')
        axes[1, 0].set_xlabel('Confidence Level')
        axes[1, 0].set_ylabel('Number of Encounters')
        
        # Add value labels on bars
        for bar, count in zip(bars, encounter_counts):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           str(count), ha='center', va='bottom', fontweight='bold')
        
        # 4. Summary statistics
        axes[1, 1].axis('off')
        
        total_encounters = confidence_results['total_encounters']
        confident_accuracy = confidence_results.get('confident_accuracy', 0)
        review_accuracy = confidence_results.get('review_accuracy', 0)
        
        summary_text = f"""
        Performance Summary by Confidence:
        
        CONFIDENT Predictions:
        • Encounters: {confident_data['encounter_count']}
        • Precision: {confident_data['avg_precision']:.3f}
        • Recall: {confident_data['avg_recall']:.3f}
        • F1 Score: {confident_data['avg_f1']:.3f}
        • Accuracy: {confident_accuracy:.3f}
        
        REQUIRES REVIEW Predictions:
        • Encounters: {review_data['encounter_count']}
        • Precision: {review_data['avg_precision']:.3f}
        • Recall: {review_data['avg_recall']:.3f}
        • F1 Score: {review_data['avg_f1']:.3f}
        • Accuracy: {review_accuracy:.3f}
        
        Total Encounters: {total_encounters}
        
        Model Calibration:
        {'✓ Well calibrated' if confident_data['avg_f1'] > review_data['avg_f1'] else '✗ Poorly calibrated'}
        """
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def comprehensive_confidence_evaluation(self) -> Dict[str, Any]:
        """
        Comprehensive confidence evaluation with clear separation of prediction-level and encounter-level metrics.
        
        Returns:
            Dictionary with both prediction-level accuracy and encounter-level precision/recall/F1 metrics
        """
        print("Performing comprehensive confidence evaluation...")
        
        # Get prediction-level analysis
        prediction_level = self.analyze_confidence_patterns()
        if 'error' in prediction_level:
            return prediction_level
        
        # Get encounter-level analysis
        encounter_level = self.evaluate_by_confidence_level()
        if 'error' in encounter_level:
            return encounter_level
        
        # Calculate additional summary statistics
        conf_df = prediction_level['confidence_data']
        
        # Prediction-level statistics by confidence
        confident_predictions = conf_df[conf_df['confidence'] == 'confident']
        review_predictions = conf_df[conf_df['confidence'] == 'requires_human_review']
        
        confident_pred_accuracy = confident_predictions['is_correct'].mean() if len(confident_predictions) > 0 else 0
        review_pred_accuracy = review_predictions['is_correct'].mean() if len(review_predictions) > 0 else 0
        
        return {
            # Prediction-level metrics (individual ICD code predictions)
            'prediction_level': {
                'total_predictions': len(conf_df),
                'confident_predictions': len(confident_predictions),
                'review_predictions': len(review_predictions),
                'confident_accuracy': confident_pred_accuracy,
                'review_accuracy': review_pred_accuracy,
                'overall_accuracy': conf_df['is_correct'].mean(),
                'calibration_gap': confident_pred_accuracy - review_pred_accuracy
            },
            
            # Encounter-level metrics (full encounter evaluation)
            'encounter_level': {
                'total_encounters': encounter_level['total_encounters'],
                'confident_encounters': encounter_level['confident']['encounter_count'],
                'review_encounters': encounter_level['requires_human_review']['encounter_count'],
                'confident_metrics': {
                    'precision': encounter_level['confident']['avg_precision'],
                    'recall': encounter_level['confident']['avg_recall'],
                    'f1': encounter_level['confident']['avg_f1']
                },
                'review_metrics': {
                    'precision': encounter_level['requires_human_review']['avg_precision'],
                    'recall': encounter_level['requires_human_review']['avg_recall'],
                    'f1': encounter_level['requires_human_review']['avg_f1']
                }
            },
            
            # Raw data for plotting
            'raw_data': {
                'confidence_data': conf_df,
                'encounter_data': encounter_level
            }
        } 