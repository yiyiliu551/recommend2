"""
Recommendation System Evaluation Metrics
Author: Yang Liu
Description: Comprehensive evaluation metrics for recommendation systems
Used at: Weibo, Qunar, SHAREit
"""

import numpy as np
from typing import List, Dict, Tuple, Union
from collections import defaultdict
import pandas as pd


class RecommendationMetrics:
    """
    Production-ready evaluation metrics for recommendation systems
    Including: Precision, Recall, F1, AUC, NDCG, MAP, Coverage, Diversity
    """
    
    @staticmethod
    def precision_at_k(recommended: List, relevant: List, k: int) -> float:
        """
        Calculate Precision@K
        
        Args:
            recommended: List of recommended items
            relevant: List of relevant items
            k: Number of top items to consider
        """
        if k <= 0:
            return 0.0
        
        recommended_at_k = recommended[:k]
        relevant_set = set(relevant)
        
        hits = sum(1 for item in recommended_at_k if item in relevant_set)
        
        return hits / min(k, len(recommended_at_k)) if recommended_at_k else 0.0
    
    @staticmethod
    def recall_at_k(recommended: List, relevant: List, k: int) -> float:
        """
        Calculate Recall@K
        """
        if not relevant or k <= 0:
            return 0.0
        
        recommended_at_k = set(recommended[:k])
        relevant_set = set(relevant)
        
        hits = len(recommended_at_k & relevant_set)
        
        return hits / len(relevant_set)
    
    @staticmethod
    def f1_at_k(recommended: List, relevant: List, k: int) -> float:
        """
        Calculate F1 Score@K
        """
        precision = RecommendationMetrics.precision_at_k(recommended, relevant, k)
        recall = RecommendationMetrics.recall_at_k(recommended, relevant, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def ndcg_at_k(recommended: List, relevant: List, k: int, 
                  relevance_scores: Dict = None) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K)
        
        Args:
            recommended: List of recommended items
            relevant: List of relevant items  
            k: Number of top items
            relevance_scores: Dict mapping items to relevance scores
        """
        if k <= 0:
            return 0.0
        
        recommended_at_k = recommended[:k]
        
        # Default binary relevance if scores not provided
        if relevance_scores is None:
            relevance_scores = {item: 1.0 for item in relevant}
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(recommended_at_k):
            rel = relevance_scores.get(item, 0)
            dcg += rel / np.log2(i + 2)  # i+2 because index starts at 0
        
        # Calculate ideal DCG
        ideal_scores = sorted([relevance_scores.get(item, 0) for item in relevant], 
                            reverse=True)[:k]
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_scores))
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def map_at_k(recommendations: Dict[str, List], 
                 relevants: Dict[str, List], k: int) -> float:
        """
        Calculate Mean Average Precision (MAP@K)
        
        Args:
            recommendations: Dict of user_id -> recommended items
            relevants: Dict of user_id -> relevant items
            k: Number of top items
        """
        ap_scores = []
        
        for user_id, rec_items in recommendations.items():
            if user_id not in relevants:
                continue
            
            relevant_items = relevants[user_id]
            if not relevant_items:
                continue
            
            # Calculate AP for this user
            hits = 0
            sum_precisions = 0.0
            
            for i, item in enumerate(rec_items[:k]):
                if item in relevant_items:
                    hits += 1
                    sum_precisions += hits / (i + 1)
            
            ap = sum_precisions / min(len(relevant_items), k)
            ap_scores.append(ap)
        
        return np.mean(ap_scores) if ap_scores else 0.0
    
    @staticmethod
    def auc_score(y_true: List[int], y_scores: List[float]) -> float:
        """
        Calculate Area Under ROC Curve (AUC)
        
        Args:
            y_true: True binary labels (0 or 1)
            y_scores: Predicted scores
        """
        # Sort by scores
        sorted_indices = np.argsort(y_scores)[::-1]
        y_true_sorted = [y_true[i] for i in sorted_indices]
        
        # Calculate AUC using trapezoidal rule
        n_pos = sum(y_true)
        n_neg = len(y_true) - n_pos
        
        if n_pos == 0 or n_neg == 0:
            return 0.5
        
        tp = 0
        fp = 0
        auc = 0.0
        
        for label in y_true_sorted:
            if label == 1:
                tp += 1
            else:
                fp += 1
                auc += tp
        
        return auc / (n_pos * n_neg)
    
    @staticmethod
    def coverage(recommendations: Dict[str, List], all_items: set) -> float:
        """
        Calculate catalog coverage
        
        Args:
            recommendations: Dict of user_id -> recommended items
            all_items: Set of all available items
        """
        recommended_items = set()
        for items in recommendations.values():
            recommended_items.update(items)
        
        return len(recommended_items) / len(all_items) if all_items else 0.0
    
    @staticmethod
    def diversity(recommendations: List, item_features: Dict) -> float:
        """
        Calculate diversity of recommendations using item features
        
        Args:
            recommendations: List of recommended items
            item_features: Dict mapping items to feature vectors
        """
        if len(recommendations) < 2:
            return 0.0
        
        distances = []
        for i in range(len(recommendations)):
            for j in range(i + 1, len(recommendations)):
                item_i = recommendations[i]
                item_j = recommendations[j]
                
                if item_i in item_features and item_j in item_features:
                    # Calculate distance between items
                    feat_i = item_features[item_i]
                    feat_j = item_features[item_j]
                    
                    # Simple Jaccard distance for categorical features
                    if isinstance(feat_i, set) and isinstance(feat_j, set):
                        intersection = len(feat_i & feat_j)
                        union = len(feat_i | feat_j)
                        distance = 1 - (intersection / union) if union > 0 else 1
                    else:
                        # Euclidean distance for numerical features
                        distance = np.linalg.norm(np.array(feat_i) - np.array(feat_j))
                    
                    distances.append(distance)
        
        return np.mean(distances) if distances else 0.0
    
    @staticmethod
    def hit_rate_at_k(recommendations: Dict[str, List], 
                      test_interactions: Dict[str, List], k: int) -> float:
        """
        Calculate Hit Rate@K (percentage of users with at least one hit)
        """
        hits = 0
        
        for user_id, rec_items in recommendations.items():
            if user_id in test_interactions:
                test_items = set(test_interactions[user_id])
                rec_items_k = set(rec_items[:k])
                
                if rec_items_k & test_items:
                    hits += 1
        
        return hits / len(recommendations) if recommendations else 0.0
    
    @staticmethod
    def mrr(recommendations: Dict[str, List], 
            relevants: Dict[str, List]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR)
        """
        rr_scores = []
        
        for user_id, rec_items in recommendations.items():
            if user_id not in relevants:
                continue
            
            relevant_items = set(relevants[user_id])
            
            for rank, item in enumerate(rec_items, 1):
                if item in relevant_items:
                    rr_scores.append(1.0 / rank)
                    break
            else:
                rr_scores.append(0.0)
        
        return np.mean(rr_scores) if rr_scores else 0.0


class OnlineMetrics:
    """
    Online metrics for A/B testing and real-time monitoring
    """
    
    @staticmethod
    def ctr(clicks: int, impressions: int) -> float:
        """Click-Through Rate"""
        return clicks / impressions if impressions > 0 else 0.0
    
    @staticmethod
    def conversion_rate(conversions: int, clicks: int) -> float:
        """Conversion Rate"""
        return conversions / clicks if clicks > 0 else 0.0
    
    @staticmethod
    def average_session_duration(durations: List[float]) -> float:
        """Average Session Duration"""
        return np.mean(durations) if durations else 0.0
    
    @staticmethod
    def user_retention(returning_users: int, total_users: int) -> float:
        """User Retention Rate"""
        return returning_users / total_users if total_users > 0 else 0.0
    
    @staticmethod
    def revenue_per_user(total_revenue: float, total_users: int) -> float:
        """Average Revenue Per User (ARPU)"""
        return total_revenue / total_users if total_users > 0 else 0.0


class MetricsEvaluator:
    """
    Complete evaluation pipeline for recommendation systems
    """
    
    def __init__(self):
        self.metrics_history = []
        
    def evaluate_model(self, 
                       predictions: Dict[str, List],
                       ground_truth: Dict[str, List],
                       k_values: List[int] = [5, 10, 20]) -> Dict:
        """
        Comprehensive model evaluation
        
        Args:
            predictions: Dict of user_id -> predicted items
            ground_truth: Dict of user_id -> actual items
            k_values: List of K values to evaluate
        
        Returns:
            Dict of metric names -> values
        """
        results = {}
        
        for k in k_values:
            # Calculate metrics for each K
            precisions = []
            recalls = []
            f1s = []
            ndcgs = []
            
            for user_id, pred_items in predictions.items():
                if user_id in ground_truth:
                    true_items = ground_truth[user_id]
                    
                    precisions.append(
                        RecommendationMetrics.precision_at_k(pred_items, true_items, k)
                    )
                    recalls.append(
                        RecommendationMetrics.recall_at_k(pred_items, true_items, k)
                    )
                    f1s.append(
                        RecommendationMetrics.f1_at_k(pred_items, true_items, k)
                    )
                    ndcgs.append(
                        RecommendationMetrics.ndcg_at_k(pred_items, true_items, k)
                    )
            
            results[f'precision@{k}'] = np.mean(precisions)
            results[f'recall@{k}'] = np.mean(recalls)
            results[f'f1@{k}'] = np.mean(f1s)
            results[f'ndcg@{k}'] = np.mean(ndcgs)
        
        # Calculate overall metrics
        results['map'] = RecommendationMetrics.map_at_k(predictions, ground_truth, max(k_values))
        results['mrr'] = RecommendationMetrics.mrr(predictions, ground_truth)
        results['hit_rate@10'] = RecommendationMetrics.hit_rate_at_k(predictions, ground_truth, 10)
        
        # Store in history
        self.metrics_history.append(results)
        
        return results
    
    def print_evaluation_report(self, metrics: Dict):
        """
        Print formatted evaluation report
        """
        print("\n" + "="*60)
        print("RECOMMENDATION SYSTEM EVALUATION REPORT")
        print("="*60)
        
        # Group metrics
        ranking_metrics = {}
        retrieval_metrics = {}
        
        for metric, value in metrics.items():
            if any(m in metric for m in ['precision', 'recall', 'f1']):
                retrieval_metrics[metric] = value
            else:
                ranking_metrics[metric] = value
        
        print("\nRetrieval Metrics:")
        print("-" * 30)
        for metric, value in retrieval_metrics.items():
            print(f"  {metric:20s}: {value:.4f}")
        
        print("\nRanking Metrics:")
        print("-" * 30)
        for metric, value in ranking_metrics.items():
            print(f"  {metric:20s}: {value:.4f}")
        
        print("="*60)


def demo_evaluation():
    """
    Demonstrate evaluation metrics with sample data
    """
    print("Recommendation System Metrics Demo")
    print("="*50)
    
    # Sample predictions and ground truth
    predictions = {
        'user1': ['item1', 'item2', 'item3', 'item4', 'item5'],
        'user2': ['item3', 'item4', 'item5', 'item6', 'item7'],
        'user3': ['item1', 'item5', 'item8', 'item9', 'item10'],
    }
    
    ground_truth = {
        'user1': ['item1', 'item3', 'item5'],
        'user2': ['item4', 'item6'],
        'user3': ['item5', 'item8', 'item11'],
    }
    
    # Create evaluator
    evaluator = MetricsEvaluator()
    
    # Evaluate
    metrics = evaluator.evaluate_model(predictions, ground_truth, k_values=[3, 5, 10])
    
    # Print report
    evaluator.print_evaluation_report(metrics)
    
    # Demonstrate online metrics
    print("\nOnline Metrics Example:")
    print("-" * 30)
    print(f"CTR: {OnlineMetrics.ctr(1250, 10000):.4f}")
    print(f"Conversion Rate: {OnlineMetrics.conversion_rate(125, 1250):.4f}")
    print(f"User Retention: {OnlineMetrics.user_retention(7500, 10000):.4f}")


if __name__ == "__main__":
    demo_evaluation()