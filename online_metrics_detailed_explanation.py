"""
Detailed Explanation of Online Recommendation Evaluation Metrics
Comprehensive guide to calculation logic and business significance

Author: Yang Liu
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass

# ===== CORE METRIC CATEGORIES =====

class OnlineMetricsExplainer:
    """Detailed explanations and calculations for online recommendation metrics"""
    
    def __init__(self):
        self.metric_categories = {
            'engagement': 'User interaction and engagement patterns',
            'business': 'Revenue and commercial value metrics', 
            'quality': 'Recommendation algorithm effectiveness',
            'technical': 'System performance and reliability'
        }
    
    # ===== ENGAGEMENT METRICS (MOST CRITICAL) =====
    
    def calculate_click_through_rate(self, impressions: List, clicks: List) -> Dict[str, Any]:
        """
        Click-Through Rate (CTR) - Most important online metric
        
        Formula: CTR = Total Clicks / Total Impressions
        
        Business Significance:
        - Primary indicator of recommendation relevance
        - Directly correlates with user satisfaction
        - Industry benchmarks: 2-8% for e-commerce recommendations
        
        Calculation Logic:
        """
        total_impressions = len(impressions)
        total_clicks = len(clicks)
        
        if total_impressions == 0:
            return {'ctr': 0.0, 'confidence': 'no_data'}
        
        ctr = total_clicks / total_impressions
        
        # Calculate confidence intervals (Wilson Score)
        confidence_interval = self._calculate_wilson_confidence_interval(
            successes=total_clicks, 
            trials=total_impressions,
            confidence_level=0.95
        )
        
        # Categorize performance
        performance_category = self._categorize_ctr_performance(ctr)
        
        return {
            'ctr': ctr,
            'total_impressions': total_impressions,
            'total_clicks': total_clicks,
            'confidence_interval': confidence_interval,
            'performance_category': performance_category,
            'benchmark_comparison': self._compare_to_benchmark(ctr, benchmark=0.05),
            'calculation_formula': 'CTR = clicks / impressions',
            'business_impact': f'CTR improvement of 1% = ~{total_impressions * 0.01:.0f} additional clicks'
        }
    
    def calculate_conversion_funnel_metrics(self, user_actions: List[Dict]) -> Dict[str, Any]:
        """
        Complete conversion funnel analysis from impression to purchase
        
        Funnel Stages:
        Impression → Click → View → Add to Cart → Purchase
        
        Key Metrics:
        - CTR: Click / Impression
        - View Rate: View / Click  
        - Cart Rate: Add to Cart / View
        - Purchase Rate: Purchase / View
        - End-to-End Conversion: Purchase / Impression
        """
        
        # Count actions by type
        action_counts = defaultdict(int)
        for action in user_actions:
            action_counts[action['type']] += 1
        
        impressions = action_counts.get('impression', 0)
        clicks = action_counts.get('click', 0) 
        views = action_counts.get('view', 0)
        add_carts = action_counts.get('add_cart', 0)
        purchases = action_counts.get('purchase', 0)
        
        # Calculate funnel metrics with safe division
        def safe_divide(numerator, denominator):
            return numerator / denominator if denominator > 0 else 0.0
        
        funnel_metrics = {
            'ctr': safe_divide(clicks, impressions),
            'click_to_view_rate': safe_divide(views, clicks),
            'view_to_cart_rate': safe_divide(add_carts, views),
            'view_to_purchase_rate': safe_divide(purchases, views),
            'cart_to_purchase_rate': safe_divide(purchases, add_carts),
            'end_to_end_conversion': safe_divide(purchases, impressions)
        }
        
        # Calculate funnel drop-off rates
        dropoff_analysis = {
            'impression_to_click_dropoff': 1 - funnel_metrics['ctr'],
            'click_to_view_dropoff': 1 - funnel_metrics['click_to_view_rate'],
            'view_to_cart_dropoff': 1 - funnel_metrics['view_to_cart_rate'],
            'cart_to_purchase_dropoff': 1 - funnel_metrics['cart_to_purchase_rate']
        }
        
        # Identify biggest bottleneck
        biggest_bottleneck = max(dropoff_analysis.items(), key=lambda x: x[1])
        
        return {
            'funnel_metrics': funnel_metrics,
            'action_counts': dict(action_counts),
            'dropoff_analysis': dropoff_analysis,
            'biggest_bottleneck': {
                'stage': biggest_bottleneck[0],
                'dropoff_rate': biggest_bottleneck[1],
                'optimization_priority': 'HIGH' if biggest_bottleneck[1] > 0.7 else 'MEDIUM'
            },
            'funnel_efficiency_score': funnel_metrics['end_to_end_conversion'] * 100,
            'calculation_explanation': {
                'ctr': 'clicks / impressions',
                'view_rate': 'views / clicks',
                'cart_rate': 'add_carts / views',
                'purchase_rate': 'purchases / views',
                'end_to_end': 'purchases / impressions'
            }
        }
    
    def calculate_dwell_time_metrics(self, interactions: List[Dict]) -> Dict[str, Any]:
        """
        Dwell Time Analysis - Time users spend engaging with recommended content
        
        Key Insights:
        - Longer dwell time = Higher engagement
        - Dwell time distribution reveals content quality
        - Optimal dwell time varies by content type
        """
        
        dwell_times = [i.get('dwell_time', 0) for i in interactions if i.get('dwell_time', 0) > 0]
        
        if not dwell_times:
            return {'error': 'no_dwell_time_data'}
        
        # Basic statistics
        mean_dwell = np.mean(dwell_times)
        median_dwell = np.median(dwell_times)
        std_dwell = np.std(dwell_times)
        
        # Percentile analysis
        percentiles = {
            'p25': np.percentile(dwell_times, 25),
            'p50': np.percentile(dwell_times, 50), 
            'p75': np.percentile(dwell_times, 75),
            'p90': np.percentile(dwell_times, 90),
            'p95': np.percentile(dwell_times, 95)
        }
        
        # Categorize dwell times
        engagement_categories = {
            'bounce': len([t for t in dwell_times if t < 5]),        # < 5 seconds
            'brief': len([t for t in dwell_times if 5 <= t < 30]),   # 5-30 seconds
            'engaged': len([t for t in dwell_times if 30 <= t < 120]), # 30s-2min
            'deep': len([t for t in dwell_times if t >= 120])        # > 2 minutes
        }
        
        # Calculate engagement score
        total_interactions = len(dwell_times)
        engagement_score = (
            engagement_categories['brief'] * 0.3 +
            engagement_categories['engaged'] * 0.7 + 
            engagement_categories['deep'] * 1.0
        ) / total_interactions
        
        return {
            'mean_dwell_seconds': mean_dwell,
            'median_dwell_seconds': median_dwell,
            'std_dwell_seconds': std_dwell,
            'percentiles': percentiles,
            'engagement_categories': engagement_categories,
            'engagement_score': engagement_score,
            'total_interactions': total_interactions,
            'quality_assessment': self._assess_dwell_time_quality(mean_dwell, engagement_score),
            'optimization_recommendations': self._get_dwell_time_recommendations(engagement_categories, total_interactions)
        }
    
    # ===== BUSINESS METRICS (ROI FOCUSED) =====
    
    def calculate_revenue_metrics(self, transactions: List[Dict], impressions: int) -> Dict[str, Any]:
        """
        Revenue and Business Value Metrics
        
        Key Metrics:
        - Revenue Per Impression (RPI)
        - Average Order Value (AOV)
        - Revenue Per User (RPU)
        - Customer Lifetime Value impact
        """
        
        if not transactions:
            return {'error': 'no_transaction_data'}
        
        # Basic revenue calculations
        total_revenue = sum(t.get('amount', 0) for t in transactions)
        num_transactions = len(transactions)
        unique_users = len(set(t.get('user_id') for t in transactions if t.get('user_id')))
        
        # Core business metrics
        revenue_per_impression = total_revenue / max(impressions, 1)
        average_order_value = total_revenue / max(num_transactions, 1)
        revenue_per_user = total_revenue / max(unique_users, 1)
        
        # Transaction distribution analysis
        transaction_amounts = [t.get('amount', 0) for t in transactions]
        revenue_distribution = {
            'min_order': min(transaction_amounts),
            'max_order': max(transaction_amounts),
            'median_order': np.median(transaction_amounts),
            'std_order': np.std(transaction_amounts)
        }
        
        # High-value customer analysis (top 20% by transaction value)
        high_value_threshold = np.percentile(transaction_amounts, 80)
        high_value_transactions = [t for t in transactions if t.get('amount', 0) >= high_value_threshold]
        high_value_revenue = sum(t.get('amount', 0) for t in high_value_transactions)
        
        # ROI calculation (assuming recommendation system cost)
        system_cost_per_impression = 0.001  # $0.001 per impression (example)
        total_system_cost = impressions * system_cost_per_impression
        roi = (total_revenue - total_system_cost) / max(total_system_cost, 1) * 100
        
        return {
            'total_revenue': total_revenue,
            'revenue_per_impression': revenue_per_impression,
            'average_order_value': average_order_value,
            'revenue_per_user': revenue_per_user,
            'num_transactions': num_transactions,
            'unique_customers': unique_users,
            'revenue_distribution': revenue_distribution,
            'high_value_analysis': {
                'threshold': high_value_threshold,
                'count': len(high_value_transactions),
                'revenue_percentage': high_value_revenue / total_revenue * 100,
                'customer_percentage': len(high_value_transactions) / num_transactions * 100
            },
            'roi_analysis': {
                'system_cost': total_system_cost,
                'net_revenue': total_revenue - total_system_cost,
                'roi_percentage': roi,
                'payback_ratio': total_revenue / max(total_system_cost, 1)
            },
            'business_impact_summary': f'Generated ${total_revenue:.2f} from {impressions} impressions (${revenue_per_impression:.4f} per impression)'
        }
    
    def calculate_customer_lifetime_value_impact(self, user_interactions: Dict[str, List]) -> Dict[str, Any]:
        """
        Customer Lifetime Value (CLV) Impact Analysis
        
        Measures how recommendations affect long-term customer value
        """
        
        clv_metrics = {}
        
        for user_id, interactions in user_interactions.items():
            # Calculate user metrics
            total_purchases = len([i for i in interactions if i.get('action') == 'purchase'])
            total_spent = sum(i.get('amount', 0) for i in interactions if i.get('action') == 'purchase')
            days_active = self._calculate_user_active_days(interactions)
            
            # Estimate CLV components
            purchase_frequency = total_purchases / max(days_active / 30, 1)  # purchases per month
            average_order_value = total_spent / max(total_purchases, 1)
            estimated_lifespan_months = min(days_active / 30 * 2, 24)  # conservative estimate
            
            # CLV calculation: AOV × Purchase Frequency × Lifespan
            estimated_clv = average_order_value * purchase_frequency * estimated_lifespan_months
            
            clv_metrics[user_id] = {
                'total_purchases': total_purchases,
                'total_spent': total_spent,
                'days_active': days_active,
                'purchase_frequency_monthly': purchase_frequency,
                'average_order_value': average_order_value,
                'estimated_clv': estimated_clv
            }
        
        # Aggregate CLV metrics
        all_clvs = [metrics['estimated_clv'] for metrics in clv_metrics.values()]
        average_clv = np.mean(all_clvs) if all_clvs else 0
        median_clv = np.median(all_clvs) if all_clvs else 0
        
        return {
            'individual_clv_metrics': clv_metrics,
            'aggregate_metrics': {
                'average_clv': average_clv,
                'median_clv': median_clv,
                'total_estimated_clv': sum(all_clvs),
                'clv_distribution': {
                    'p25': np.percentile(all_clvs, 25) if all_clvs else 0,
                    'p75': np.percentile(all_clvs, 75) if all_clvs else 0,
                    'p90': np.percentile(all_clvs, 90) if all_clvs else 0
                }
            },
            'clv_impact_assessment': self._assess_clv_impact(average_clv),
            'recommendation_attribution': f'Recommendations contributed to ${sum(all_clvs):.2f} in estimated CLV'
        }
    
    # ===== QUALITY METRICS (ALGORITHM EFFECTIVENESS) =====
    
    def calculate_recommendation_diversity_metrics(self, recommendations: List[Dict]) -> Dict[str, Any]:
        """
        Recommendation Diversity Analysis
        
        Types of Diversity:
        1. Intra-list diversity: Variety within single recommendation list
        2. Inter-list diversity: Variety across different user sessions
        3. Category diversity: Distribution across product categories
        4. Price diversity: Distribution across price ranges
        """
        
        if not recommendations:
            return {'error': 'no_recommendations'}
        
        # Extract attributes for diversity analysis
        categories = [r.get('category', 'unknown') for r in recommendations]
        brands = [r.get('brand', 'unknown') for r in recommendations]
        prices = [r.get('price', 0) for r in recommendations if r.get('price', 0) > 0]
        
        # Category diversity (Simpson's Diversity Index)
        category_counts = defaultdict(int)
        for cat in categories:
            category_counts[cat] += 1
        
        total_items = len(categories)
        simpson_diversity = 1 - sum((count/total_items)**2 for count in category_counts.values())
        
        # Shannon diversity (entropy-based)
        shannon_diversity = -sum((count/total_items) * np.log(count/total_items) 
                                for count in category_counts.values() if count > 0)
        
        # Brand diversity
        unique_brands = len(set(brands))
        brand_diversity_ratio = unique_brands / len(brands)
        
        # Price diversity (coefficient of variation)
        price_diversity = np.std(prices) / np.mean(prices) if prices and np.mean(prices) > 0 else 0
        
        # Gini coefficient for price distribution
        gini_coefficient = self._calculate_gini_coefficient(prices) if prices else 0
        
        return {
            'category_diversity': {
                'simpson_index': simpson_diversity,
                'shannon_entropy': shannon_diversity,
                'unique_categories': len(category_counts),
                'category_distribution': dict(category_counts)
            },
            'brand_diversity': {
                'unique_brands': unique_brands,
                'brand_diversity_ratio': brand_diversity_ratio,
                'total_brands_available': unique_brands
            },
            'price_diversity': {
                'coefficient_of_variation': price_diversity,
                'gini_coefficient': gini_coefficient,
                'price_range': max(prices) - min(prices) if prices else 0,
                'price_std': np.std(prices) if prices else 0
            },
            'overall_diversity_score': (simpson_diversity + brand_diversity_ratio + min(price_diversity, 1.0)) / 3,
            'diversity_assessment': self._assess_diversity_quality(simpson_diversity, brand_diversity_ratio),
            'optimization_suggestions': self._get_diversity_optimization_suggestions(category_counts, brand_diversity_ratio)
        }
    
    def calculate_recommendation_novelty_metrics(self, user_history: List[str], recommendations: List[str]) -> Dict[str, Any]:
        """
        Recommendation Novelty Analysis
        
        Novelty measures how new/unseen recommended items are for users
        Higher novelty = better discovery potential
        Lower novelty = safer, more predictable recommendations
        """
        
        if not user_history or not recommendations:
            return {'error': 'insufficient_data'}
        
        # Convert to sets for efficient comparison
        history_set = set(user_history)
        recommendation_set = set(recommendations)
        
        # Basic novelty metrics
        novel_items = recommendation_set - history_set
        familiar_items = recommendation_set & history_set
        
        novelty_ratio = len(novel_items) / len(recommendations)
        familiarity_ratio = len(familiar_items) / len(recommendations)
        
        # Temporal novelty (items not seen in recent history)
        recent_history = set(user_history[-20:])  # last 20 items
        recently_novel = recommendation_set - recent_history
        recent_novelty_ratio = len(recently_novel) / len(recommendations)
        
        # Category-level novelty
        if hasattr(self, '_get_item_categories'):
            user_categories = set(self._get_item_categories(item) for item in user_history)
            rec_categories = set(self._get_item_categories(item) for item in recommendations)
            novel_categories = rec_categories - user_categories
            category_novelty_ratio = len(novel_categories) / len(rec_categories) if rec_categories else 0
        else:
            category_novelty_ratio = 0.5  # Mock value
        
        # Popularity-based novelty (less popular items = more novel)
        popularity_scores = [self._get_item_popularity(item) for item in recommendations]
        avg_popularity = np.mean(popularity_scores) if popularity_scores else 0.5
        popularity_novelty = 1 - avg_popularity  # inverse of popularity
        
        return {
            'item_level_novelty': {
                'novel_items_count': len(novel_items),
                'familiar_items_count': len(familiar_items),
                'novelty_ratio': novelty_ratio,
                'familiarity_ratio': familiarity_ratio
            },
            'temporal_novelty': {
                'recent_novel_count': len(recently_novel),
                'recent_novelty_ratio': recent_novelty_ratio
            },
            'category_novelty': {
                'novel_categories_count': len(novel_categories) if 'novel_categories' in locals() else 0,
                'category_novelty_ratio': category_novelty_ratio
            },
            'popularity_novelty': {
                'avg_popularity_score': avg_popularity,
                'popularity_novelty_score': popularity_novelty
            },
            'overall_novelty_score': (novelty_ratio + recent_novelty_ratio + category_novelty_ratio + popularity_novelty) / 4,
            'novelty_assessment': self._assess_novelty_balance(novelty_ratio),
            'business_impact': {
                'discovery_potential': 'HIGH' if novelty_ratio > 0.7 else 'MEDIUM' if novelty_ratio > 0.4 else 'LOW',
                'conversion_risk': 'HIGH' if novelty_ratio > 0.8 else 'LOW',
                'recommendation': self._get_novelty_recommendation(novelty_ratio)
            }
        }
    
    # ===== TECHNICAL METRICS (SYSTEM PERFORMANCE) =====
    
    def calculate_system_performance_metrics(self, request_logs: List[Dict]) -> Dict[str, Any]:
        """
        System Performance and Reliability Metrics
        
        Key Areas:
        - Latency and response times
        - Throughput and capacity
        - Error rates and reliability
        - Resource utilization
        """
        
        if not request_logs:
            return {'error': 'no_request_data'}
        
        # Extract timing information
        response_times = [log.get('response_time_ms', 0) for log in request_logs if log.get('response_time_ms')]
        successful_requests = [log for log in request_logs if log.get('status', 'error') == 'success']
        failed_requests = [log for log in request_logs if log.get('status', 'error') == 'error']
        
        # Latency analysis
        if response_times:
            latency_metrics = {
                'mean_latency_ms': np.mean(response_times),
                'median_latency_ms': np.median(response_times),
                'p95_latency_ms': np.percentile(response_times, 95),
                'p99_latency_ms': np.percentile(response_times, 99),
                'max_latency_ms': max(response_times),
                'std_latency_ms': np.std(response_times)
            }
        else:
            latency_metrics = {}
        
        # Throughput analysis
        if request_logs:
            timestamps = [log.get('timestamp', 0) for log in request_logs if log.get('timestamp')]
            if timestamps:
                time_span_seconds = max(timestamps) - min(timestamps)
                requests_per_second = len(request_logs) / max(time_span_seconds, 1)
            else:
                requests_per_second = 0
        else:
            requests_per_second = 0
        
        # Reliability metrics
        total_requests = len(request_logs)
        success_rate = len(successful_requests) / max(total_requests, 1)
        error_rate = len(failed_requests) / max(total_requests, 1)
        
        # Error categorization
        error_types = defaultdict(int)
        for failed_req in failed_requests:
            error_type = failed_req.get('error_type', 'unknown')
            error_types[error_type] += 1
        
        # Performance assessment
        performance_grade = self._assess_system_performance(latency_metrics.get('p95_latency_ms', 1000), 
                                                          success_rate, 
                                                          requests_per_second)
        
        return {
            'latency_metrics': latency_metrics,
            'throughput_metrics': {
                'requests_per_second': requests_per_second,
                'total_requests': total_requests,
                'peak_qps': requests_per_second * 1.5  # Estimated peak
            },
            'reliability_metrics': {
                'success_rate': success_rate,
                'error_rate': error_rate,
                'successful_requests': len(successful_requests),
                'failed_requests': len(failed_requests),
                'error_types': dict(error_types)
            },
            'performance_grade': performance_grade,
            'sla_compliance': {
                'latency_sla_99p': latency_metrics.get('p99_latency_ms', 1000) < 500,  # < 500ms SLA
                'availability_sla': success_rate > 0.999,  # 99.9% availability
                'throughput_sla': requests_per_second > 100  # > 100 RPS capability
            },
            'optimization_priorities': self._get_performance_optimization_priorities(latency_metrics, success_rate)
        }
    
    # ===== A/B TESTING METRICS =====
    
    def calculate_ab_test_statistical_significance(self, control_data: Dict, treatment_data: Dict) -> Dict[str, Any]:
        """
        Statistical Significance Testing for A/B Tests
        
        Methods:
        - Z-test for proportions (CTR, conversion rates)
        - T-test for continuous metrics (revenue, dwell time)
        - Chi-square test for categorical distributions
        - Confidence intervals and effect sizes
        """
        
        results = {}
        
        # Test CTR difference
        if 'clicks' in control_data and 'impressions' in control_data:
            ctr_test = self._z_test_proportions(
                control_successes=control_data['clicks'],
                control_trials=control_data['impressions'],
                treatment_successes=treatment_data.get('clicks', 0),
                treatment_trials=treatment_data.get('impressions', 1)
            )
            results['ctr_significance_test'] = ctr_test
        
        # Test revenue difference (t-test)
        if 'revenue_per_user' in control_data:
            revenue_test = self._t_test_means(
                control_values=control_data.get('revenue_distribution', []),
                treatment_values=treatment_data.get('revenue_distribution', [])
            )
            results['revenue_significance_test'] = revenue_test
        
        # Overall experiment assessment
        significant_improvements = sum(1 for test in results.values() 
                                     if test.get('p_value', 1) < 0.05 and test.get('effect_size', 0) > 0)
        
        experiment_conclusion = {
            'statistically_significant_metrics': significant_improvements,
            'total_metrics_tested': len(results),
            'overall_significance': significant_improvements > 0,
            'recommendation': 'LAUNCH' if significant_improvements >= 2 else 'ITERATE' if significant_improvements == 1 else 'ABANDON'
        }
        
        results['experiment_conclusion'] = experiment_conclusion
        
        return results
    
    # ===== UTILITY METHODS =====
    
    def _calculate_wilson_confidence_interval(self, successes: int, trials: int, confidence_level: float = 0.95) -> Dict[str, float]:
        """Wilson Score confidence interval for binomial proportions"""
        if trials == 0:
            return {'lower': 0, 'upper': 0}
        
        from scipy import stats
        z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        p = successes / trials
        
        denominator = 1 + z**2 / trials
        centre = (p + z**2 / (2 * trials)) / denominator
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials) / denominator
        
        return {
            'lower': max(0, centre - margin),
            'upper': min(1, centre + margin),
            'point_estimate': p
        }
    
    def _categorize_ctr_performance(self, ctr: float) -> str:
        """Categorize CTR performance based on industry benchmarks"""
        if ctr >= 0.08:
            return 'EXCELLENT'
        elif ctr >= 0.05:
            return 'GOOD'
        elif ctr >= 0.02:
            return 'AVERAGE'
        else:
            return 'POOR'
    
    def _compare_to_benchmark(self, value: float, benchmark: float) -> Dict[str, Any]:
        """Compare metric value to benchmark"""
        difference = value - benchmark
        percentage_diff = (difference / benchmark) * 100 if benchmark != 0 else 0
        
        return {
            'benchmark_value': benchmark,
            'current_value': value,
            'absolute_difference': difference,
            'percentage_difference': percentage_diff,
            'performance_vs_benchmark': 'ABOVE' if difference > 0 else 'BELOW' if difference < 0 else 'AT'
        }
    
    def _assess_dwell_time_quality(self, mean_dwell: float, engagement_score: float) -> str:
        """Assess overall dwell time quality"""
        if mean_dwell > 60 and engagement_score > 0.7:
            return 'EXCELLENT_ENGAGEMENT'
        elif mean_dwell > 30 and engagement_score > 0.5:
            return 'GOOD_ENGAGEMENT'
        elif mean_dwell > 10:
            return 'MODERATE_ENGAGEMENT'
        else:
            return 'LOW_ENGAGEMENT'
    
    def _get_dwell_time_recommendations(self, categories: Dict, total: int) -> List[str]:
        """Generate recommendations based on dwell time patterns"""
        recommendations = []
        
        bounce_rate = categories['bounce'] / total
        if bounce_rate > 0.5:
            recommendations.append('HIGH_BOUNCE_RATE: Improve content relevance and initial engagement')
        
        deep_engagement = categories['deep'] / total
        if deep_engagement < 0.1:
            recommendations.append('LOW_DEEP_ENGAGEMENT: Add more compelling content to increase time spent')
        
        return recommendations
    
    def _calculate_user_active_days(self, interactions: List[Dict]) -> int:
        """Calculate number of active days for a user"""
        if not interactions:
            return 0
        
        timestamps = [i.get('timestamp', 0) for i in interactions if i.get('timestamp')]
        if not timestamps:
            return 1
        
        min_time = min(timestamps)
        max_time = max(timestamps)
        return max(1, int((max_time - min_time) / 86400))  # Convert to days
    
    def _assess_clv_impact(self, average_clv: float) -> str:
        """Assess CLV impact level"""
        if average_clv > 1000:
            return 'HIGH_VALUE_CUSTOMERS'
        elif average_clv > 500:
            return 'MEDIUM_VALUE_CUSTOMERS'
        elif average_clv > 100:
            return 'STANDARD_VALUE_CUSTOMERS'
        else:
            return 'LOW_VALUE_CUSTOMERS'
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for inequality measurement"""
        if not values or len(values) < 2:
            return 0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        
        return (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * np.sum(sorted_values)) - (n + 1) / n
    
    def _assess_diversity_quality(self, simpson_diversity: float, brand_diversity: float) -> str:
        """Assess overall diversity quality"""
        diversity_score = (simpson_diversity + brand_diversity) / 2
        
        if diversity_score > 0.8:
            return 'HIGHLY_DIVERSE'
        elif diversity_score > 0.6:
            return 'MODERATELY_DIVERSE'
        elif diversity_score > 0.4:
            return 'SOMEWHAT_DIVERSE'
        else:
            return 'LOW_DIVERSITY'
    
    def _get_diversity_optimization_suggestions(self, category_counts: Dict, brand_diversity: float) -> List[str]:
        """Generate diversity optimization suggestions"""
        suggestions = []
        
        # Check category concentration
        total_items = sum(category_counts.values())
        max_category_share = max(category_counts.values()) / total_items
        
        if max_category_share > 0.5:
            suggestions.append('CATEGORY_CONCENTRATION: Reduce dominance of single category')
        
        if brand_diversity < 0.5:
            suggestions.append('BRAND_CONCENTRATION: Increase brand variety in recommendations')
        
        if len(category_counts) < 3:
            suggestions.append('CATEGORY_EXPANSION: Include items from more categories')
        
        return suggestions
    
    def _get_item_popularity(self, item_id: str) -> float:
        """Get item popularity score (mock implementation)"""
        # In real implementation, this would query item popularity from database
        return hash(item_id) % 100 / 100.0
    
    def _assess_novelty_balance(self, novelty_ratio: float) -> str:
        """Assess novelty balance"""
        if novelty_ratio > 0.8:
            return 'HIGH_NOVELTY_HIGH_RISK'
        elif novelty_ratio > 0.6:
            return 'BALANCED_EXPLORATION'
        elif novelty_ratio > 0.3:
            return 'CONSERVATIVE_RECOMMENDATIONS'
        else:
            return 'LOW_NOVELTY_SAFE_BETS'
    
    def _get_novelty_recommendation(self, novelty_ratio: float) -> str:
        """Get novelty optimization recommendation"""
        if novelty_ratio > 0.8:
            return 'Consider reducing novelty to improve conversion rates'
        elif novelty_ratio < 0.3:
            return 'Increase novelty to improve user discovery and long-term engagement'
        else:
            return 'Novelty level appears well-balanced'
    
    def _assess_system_performance(self, p95_latency: float, success_rate: float, qps: float) -> Dict[str, str]:
        """Assess overall system performance"""
        latency_grade = 'A' if p95_latency < 100 else 'B' if p95_latency < 300 else 'C' if p95_latency < 500 else 'D'
        reliability_grade = 'A' if success_rate > 0.999 else 'B' if success_rate > 0.995 else 'C' if success_rate > 0.99 else 'D'
        throughput_grade = 'A' if qps > 1000 else 'B' if qps > 500 else 'C' if qps > 100 else 'D'
        
        return {
            'latency_grade': latency_grade,
            'reliability_grade': reliability_grade,
            'throughput_grade': throughput_grade,
            'overall_grade': min(latency_grade, reliability_grade, throughput_grade)
        }
    
    def _get_performance_optimization_priorities(self, latency_metrics: Dict, success_rate: float) -> List[str]:
        """Get performance optimization priorities"""
        priorities = []
        
        p95_latency = latency_metrics.get('p95_latency_ms', 0)
        if p95_latency > 500:
            priorities.append('CRITICAL: Reduce P95 latency below 500ms')
        elif p95_latency > 300:
            priorities.append('HIGH: Optimize P95 latency to under 300ms')
        
        if success_rate < 0.99:
            priorities.append('CRITICAL: Improve system reliability above 99%')
        elif success_rate < 0.999:
            priorities.append('MEDIUM: Improve reliability to 99.9%')
        
        return priorities
    
    def _z_test_proportions(self, control_successes: int, control_trials: int, 
                           treatment_successes: int, treatment_trials: int) -> Dict[str, Any]:
        """Z-test for comparing two proportions"""
        if control_trials == 0 or treatment_trials == 0:
            return {'error': 'insufficient_data'}
        
        p1 = control_successes / control_trials
        p2 = treatment_successes / treatment_trials
        
        # Pooled proportion
        p_pool = (control_successes + treatment_successes) / (control_trials + treatment_trials)
        
        # Standard error
        se = np.sqrt(p_pool * (1 - p_pool) * (1/control_trials + 1/treatment_trials))
        
        # Z-statistic
        z_stat = (p2 - p1) / se if se > 0 else 0
        
        # P-value (two-tailed)
        from scipy import stats
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # Effect size (Cohen's h)
        effect_size = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))
        
        return {
            'control_rate': p1,
            'treatment_rate': p2,
            'difference': p2 - p1,
            'relative_improvement': (p2 - p1) / p1 * 100 if p1 > 0 else 0,
            'z_statistic': z_stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'is_significant': p_value < 0.05,
            'confidence_interval_95': {
                'lower': (p2 - p1) - 1.96 * se,
                'upper': (p2 - p1) + 1.96 * se
            }
        }
    
    def _t_test_means(self, control_values: List[float], treatment_values: List[float]) -> Dict[str, Any]:
        """T-test for comparing means of two groups"""
        if not control_values or not treatment_values:
            return {'error': 'insufficient_data'}
        
        from scipy import stats
        
        control_mean = np.mean(control_values)
        treatment_mean = np.mean(treatment_values)
        
        # Welch's t-test (unequal variances)
        t_stat, p_value = stats.ttest_ind(treatment_values, control_values, equal_var=False)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(control_values) - 1) * np.var(control_values) + 
                             (len(treatment_values) - 1) * np.var(treatment_values)) / 
                            (len(control_values) + len(treatment_values) - 2))
        cohens_d = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0
        
        return {
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'difference': treatment_mean - control_mean,
            'relative_improvement': (treatment_mean - control_mean) / control_mean * 100 if control_mean != 0 else 0,
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'is_significant': p_value < 0.05,
            'effect_size_interpretation': 'LARGE' if abs(cohens_d) > 0.8 else 'MEDIUM' if abs(cohens_d) > 0.5 else 'SMALL'
        }


# ===== DEMO AND USAGE EXAMPLES =====

def demonstrate_metrics_calculations():
    """Comprehensive demonstration of all metric calculations"""
    
    print("=" * 80)
    print("Online Recommendation Metrics - Detailed Calculations Demo")
    print("=" * 80)
    
    explainer = OnlineMetricsExplainer()
    
    # Sample data
    sample_impressions = [{'user': f'user_{i}', 'item': f'item_{i%100}'} for i in range(10000)]
    sample_clicks = [{'user': f'user_{i}', 'item': f'item_{i%100}'} for i in range(500)]
    
    sample_actions = [
        {'type': 'impression', 'user': 'user_1', 'item': 'item_1'},
        {'type': 'click', 'user': 'user_1', 'item': 'item_1'},
        {'type': 'view', 'user': 'user_1', 'item': 'item_1'},
        {'type': 'add_cart', 'user': 'user_1', 'item': 'item_1'},
        {'type': 'purchase', 'user': 'user_1', 'item': 'item_1', 'amount': 99.99}
    ]
    
    # 1. CTR Analysis
    print("\n1. CLICK-THROUGH RATE ANALYSIS")
    print("-" * 40)
    ctr_results = explainer.calculate_click_through_rate(sample_impressions, sample_clicks)
    print(f"CTR: {ctr_results['ctr']:.4f}")
    print(f"Performance Category: {ctr_results['performance_category']}")
    print(f"Confidence Interval: [{ctr_results['confidence_interval']['lower']:.4f}, {ctr_results['confidence_interval']['upper']:.4f}]")
    print(f"Business Impact: {ctr_results['business_impact']}")
    
    # 2. Conversion Funnel
    print("\n2. CONVERSION FUNNEL ANALYSIS")
    print("-" * 40)
    funnel_results = explainer.calculate_conversion_funnel_metrics(sample_actions * 1000)  # Scale up
    print(f"End-to-End Conversion: {funnel_results['funnel_metrics']['end_to_end_conversion']:.4f}")
    print(f"Biggest Bottleneck: {funnel_results['biggest_bottleneck']['stage']}")
    print(f"Funnel Efficiency Score: {funnel_results['funnel_efficiency_score']:.2f}%")
    
    # 3. Dwell Time Analysis
    print("\n3. DWELL TIME ANALYSIS")
    print("-" * 40)
    sample_interactions = [{'dwell_time': np.random.exponential(30)} for _ in range(1000)]
    dwell_results = explainer.calculate_dwell_time_metrics(sample_interactions)
    print(f"Mean Dwell Time: {dwell_results['mean_dwell_seconds']:.1f} seconds")
    print(f"Engagement Score: {dwell_results['engagement_score']:.3f}")
    print(f"Quality Assessment: {dwell_results['quality_assessment']}")
    
    # 4. Revenue Metrics
    print("\n4. REVENUE METRICS ANALYSIS")
    print("-" * 40)
    sample_transactions = [
        {'amount': 50 + np.random.exponential(50), 'user_id': f'user_{i}'} 
        for i in range(200)
    ]
    revenue_results = explainer.calculate_revenue_metrics(sample_transactions, 10000)
    print(f"Revenue Per Impression: ${revenue_results['revenue_per_impression']:.4f}")
    print(f"Average Order Value: ${revenue_results['average_order_value']:.2f}")
    print(f"ROI: {revenue_results['roi_analysis']['roi_percentage']:.1f}%")
    
    # 5. Diversity Analysis
    print("\n5. RECOMMENDATION DIVERSITY")
    print("-" * 40)
    sample_recommendations = [
        {'category': f'category_{i%5}', 'brand': f'brand_{i%10}', 'price': 20 + i*5}
        for i in range(20)
    ]
    diversity_results = explainer.calculate_recommendation_diversity_metrics(sample_recommendations)
    print(f"Simpson Diversity Index: {diversity_results['category_diversity']['simpson_index']:.3f}")
    print(f"Overall Diversity Score: {diversity_results['overall_diversity_score']:.3f}")
    print(f"Assessment: {diversity_results['diversity_assessment']}")
    
    print("\n" + "=" * 80)
    print("All metric calculations completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_metrics_calculations()