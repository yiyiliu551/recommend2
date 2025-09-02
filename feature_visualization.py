"""
Feature Analysis and Visualization for Video Recommendations
Author: Yang Liu
Description: Comprehensive feature analysis with visual insights
Experience: Built similar analytics at Weibo for 400M users
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json


class FeatureAnalyzer:
    """Analyze and visualize recommendation system features"""
    
    def __init__(self):
        self.feature_categories = {
            'video': ['duration', 'completion_rate', 'quality_score', 'engagement_peaks'],
            'text': ['sentiment', 'readability', 'genre_match', 'keyword_overlap'],
            'image': ['quality_score', 'brightness', 'composition', 'aesthetic_score'],
            'user': ['watch_history', 'genre_preference', 'platform', 'session_length'],
            'interaction': ['clicks', 'likes', 'shares', 'comments', 'purchases']
        }
    
    def analyze_feature_distributions(self, data: pd.DataFrame) -> Dict:
        """Analyze statistical distributions of features"""
        
        analysis = {
            'univariate_stats': {},
            'correlations': {},
            'outliers': {},
            'missing_data': {},
            'recommendations': []
        }
        
        # Univariate statistics
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            analysis['univariate_stats'][col] = {
                'mean': data[col].mean(),
                'std': data[col].std(),
                'median': data[col].median(),
                'skewness': data[col].skew(),
                'kurtosis': data[col].kurtosis(),
                'min': data[col].min(),
                'max': data[col].max(),
                'q25': data[col].quantile(0.25),
                'q75': data[col].quantile(0.75)
            }
        
        # Correlation analysis
        corr_matrix = data[numeric_columns].corr()
        analysis['correlations'] = self._analyze_correlations(corr_matrix)
        
        # Outlier detection
        analysis['outliers'] = self._detect_outliers(data[numeric_columns])
        
        # Missing data analysis
        analysis['missing_data'] = data.isnull().sum().to_dict()
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_data_recommendations(analysis)
        
        return analysis
    
    def _analyze_correlations(self, corr_matrix: pd.DataFrame) -> Dict:
        """Analyze feature correlations"""
        
        correlations = {
            'strong_positive': [],  # > 0.7
            'strong_negative': [],  # < -0.7
            'moderate_positive': [], # 0.3 to 0.7
            'moderate_negative': [], # -0.7 to -0.3
            'weak_or_none': []      # -0.3 to 0.3
        }
        
        # Get upper triangle (avoid duplicates)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        upper_corr = corr_matrix.where(mask)
        
        for col in upper_corr.columns:
            for idx in upper_corr.index:
                corr_val = upper_corr.loc[idx, col]
                if pd.isna(corr_val):
                    continue
                
                pair = (idx, col, corr_val)
                
                if corr_val > 0.7:
                    correlations['strong_positive'].append(pair)
                elif corr_val < -0.7:
                    correlations['strong_negative'].append(pair)
                elif 0.3 <= corr_val <= 0.7:
                    correlations['moderate_positive'].append(pair)
                elif -0.7 <= corr_val <= -0.3:
                    correlations['moderate_negative'].append(pair)
                else:
                    correlations['weak_or_none'].append(pair)
        
        return correlations
    
    def _detect_outliers(self, data: pd.DataFrame) -> Dict:
        """Detect outliers using IQR method"""
        
        outliers = {}
        
        for column in data.columns:
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Find outliers
            outlier_mask = (data[column] < lower_bound) | (data[column] > upper_bound)
            outlier_count = outlier_mask.sum()
            
            outliers[column] = {
                'count': outlier_count,
                'percentage': (outlier_count / len(data)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        
        return outliers
    
    def _generate_data_recommendations(self, analysis: Dict) -> List[str]:
        """Generate recommendations based on data analysis"""
        
        recommendations = []
        
        # Check for skewed distributions
        for feature, stats in analysis['univariate_stats'].items():
            if abs(stats['skewness']) > 2:
                recommendations.append(
                    f"Feature '{feature}' is highly skewed (skew={stats['skewness']:.2f}). "
                    f"Consider log transformation."
                )
        
        # Check for high correlations
        if analysis['correlations']['strong_positive']:
            recommendations.append(
                f"Found {len(analysis['correlations']['strong_positive'])} highly correlated feature pairs. "
                f"Consider dimensionality reduction."
            )
        
        # Check for outliers
        high_outlier_features = [
            feat for feat, info in analysis['outliers'].items()
            if info['percentage'] > 5
        ]
        if high_outlier_features:
            recommendations.append(
                f"Features {high_outlier_features} have >5% outliers. "
                f"Consider outlier treatment."
            )
        
        # Check for missing data
        high_missing_features = [
            feat for feat, count in analysis['missing_data'].items()
            if count > 0
        ]
        if high_missing_features:
            recommendations.append(
                f"Features {high_missing_features} have missing values. "
                f"Consider imputation strategies."
            )
        
        return recommendations
    
    def analyze_user_behavior_patterns(self, user_interactions: pd.DataFrame) -> Dict:
        """Analyze user behavior patterns"""
        
        patterns = {
            'temporal_patterns': {},
            'content_preferences': {},
            'engagement_segments': {},
            'churn_indicators': {}
        }
        
        # Temporal patterns
        if 'timestamp' in user_interactions.columns:
            user_interactions['hour'] = pd.to_datetime(user_interactions['timestamp']).dt.hour
            user_interactions['day_of_week'] = pd.to_datetime(user_interactions['timestamp']).dt.dayofweek
            
            patterns['temporal_patterns'] = {
                'peak_hours': user_interactions['hour'].value_counts().head(3).to_dict(),
                'peak_days': user_interactions['day_of_week'].value_counts().head(3).to_dict(),
                'hourly_distribution': user_interactions['hour'].value_counts().sort_index().to_dict()
            }
        
        # Content preferences
        if 'content_type' in user_interactions.columns:
            patterns['content_preferences'] = {
                'type_distribution': user_interactions['content_type'].value_counts().to_dict(),
                'avg_engagement_by_type': user_interactions.groupby('content_type')['engagement_score'].mean().to_dict()
            }
        
        # User segmentation based on engagement
        if 'engagement_score' in user_interactions.columns:
            user_engagement = user_interactions.groupby('user_id')['engagement_score'].agg(['mean', 'std', 'count'])
            
            # Define segments
            high_engagement = user_engagement['mean'] > user_engagement['mean'].quantile(0.8)
            low_engagement = user_engagement['mean'] < user_engagement['mean'].quantile(0.2)
            
            patterns['engagement_segments'] = {
                'high_engagement_users': high_engagement.sum(),
                'low_engagement_users': low_engagement.sum(),
                'medium_engagement_users': len(user_engagement) - high_engagement.sum() - low_engagement.sum()
            }
        
        # Churn indicators
        if 'last_interaction' in user_interactions.columns:
            current_time = datetime.now()
            user_last_activity = user_interactions.groupby('user_id')['last_interaction'].max()
            
            days_since_last = (current_time - pd.to_datetime(user_last_activity)).dt.days
            
            patterns['churn_indicators'] = {
                'users_7_days_inactive': (days_since_last > 7).sum(),
                'users_30_days_inactive': (days_since_last > 30).sum(),
                'users_90_days_inactive': (days_since_last > 90).sum()
            }
        
        return patterns
    
    def create_feature_importance_analysis(self, features: np.ndarray, 
                                         target: np.ndarray,
                                         feature_names: List[str]) -> Dict:
        """Analyze feature importance using multiple methods"""
        
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.feature_selection import mutual_info_regression
        from sklearn.linear_model import LassoCV
        
        importance_analysis = {
            'random_forest': {},
            'mutual_info': {},
            'lasso': {},
            'correlation': {},
            'combined_ranking': {}
        }
        
        # Random Forest importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(features, target)
        rf_importance = rf.feature_importances_
        importance_analysis['random_forest'] = dict(zip(feature_names, rf_importance))
        
        # Mutual information
        mi_scores = mutual_info_regression(features, target)
        importance_analysis['mutual_info'] = dict(zip(feature_names, mi_scores))
        
        # Lasso coefficients
        lasso = LassoCV(cv=5, random_state=42)
        lasso.fit(features, target)
        lasso_coef = np.abs(lasso.coef_)
        importance_analysis['lasso'] = dict(zip(feature_names, lasso_coef))
        
        # Correlation with target
        correlations = [np.corrcoef(features[:, i], target)[0, 1] for i in range(features.shape[1])]
        correlations = [abs(c) if not np.isnan(c) else 0 for c in correlations]
        importance_analysis['correlation'] = dict(zip(feature_names, correlations))
        
        # Combined ranking (average of normalized scores)
        combined_scores = {}
        for feature in feature_names:
            # Normalize each score to 0-1
            rf_norm = importance_analysis['random_forest'][feature] / max(importance_analysis['random_forest'].values())
            mi_norm = importance_analysis['mutual_info'][feature] / max(importance_analysis['mutual_info'].values())
            lasso_norm = importance_analysis['lasso'][feature] / max(importance_analysis['lasso'].values()) if max(importance_analysis['lasso'].values()) > 0 else 0
            corr_norm = importance_analysis['correlation'][feature] / max(importance_analysis['correlation'].values())
            
            combined_scores[feature] = np.mean([rf_norm, mi_norm, lasso_norm, corr_norm])
        
        importance_analysis['combined_ranking'] = combined_scores
        
        return importance_analysis
    
    def generate_feature_report(self, analysis: Dict, patterns: Dict, 
                               importance: Dict = None) -> str:
        """Generate comprehensive feature analysis report"""
        
        report = []
        report.append("="*80)
        report.append("COMPREHENSIVE FEATURE ANALYSIS REPORT")
        report.append("="*80)
        
        # Data Quality Section
        report.append("\n📊 DATA QUALITY ANALYSIS")
        report.append("-"*50)
        
        # Missing data
        missing_features = [k for k, v in analysis['missing_data'].items() if v > 0]
        if missing_features:
            report.append(f"Features with missing data: {len(missing_features)}")
            for feature in missing_features[:5]:  # Top 5
                count = analysis['missing_data'][feature]
                report.append(f"  • {feature}: {count} missing values")
        else:
            report.append("✓ No missing data detected")
        
        # Outliers
        high_outlier_features = [
            f for f, info in analysis['outliers'].items()
            if info['percentage'] > 5
        ]
        if high_outlier_features:
            report.append(f"\nFeatures with >5% outliers: {len(high_outlier_features)}")
            for feature in high_outlier_features[:5]:
                pct = analysis['outliers'][feature]['percentage']
                report.append(f"  • {feature}: {pct:.1f}% outliers")
        
        # Feature Correlations
        report.append("\n🔗 FEATURE CORRELATIONS")
        report.append("-"*50)
        
        strong_pos = len(analysis['correlations']['strong_positive'])
        strong_neg = len(analysis['correlations']['strong_negative'])
        
        report.append(f"Strong positive correlations (>0.7): {strong_pos}")
        report.append(f"Strong negative correlations (<-0.7): {strong_neg}")
        
        if strong_pos > 0:
            report.append("\nTop positive correlations:")
            for feat1, feat2, corr in analysis['correlations']['strong_positive'][:3]:
                report.append(f"  • {feat1} ↔ {feat2}: {corr:.3f}")
        
        # User Behavior Patterns
        if patterns:
            report.append("\n👥 USER BEHAVIOR PATTERNS")
            report.append("-"*50)
            
            if 'temporal_patterns' in patterns:
                peak_hours = patterns['temporal_patterns'].get('peak_hours', {})
                if peak_hours:
                    top_hour = max(peak_hours, key=peak_hours.get)
                    report.append(f"Peak activity hour: {top_hour}:00 ({peak_hours[top_hour]} interactions)")
            
            if 'engagement_segments' in patterns:
                segments = patterns['engagement_segments']
                total_users = sum(segments.values())
                high_pct = (segments['high_engagement_users'] / total_users) * 100
                report.append(f"High engagement users: {high_pct:.1f}%")
            
            if 'churn_indicators' in patterns:
                churn = patterns['churn_indicators']
                if 'users_30_days_inactive' in churn:
                    report.append(f"Users inactive >30 days: {churn['users_30_days_inactive']}")
        
        # Feature Importance
        if importance:
            report.append("\n⭐ FEATURE IMPORTANCE ANALYSIS")
            report.append("-"*50)
            
            # Top 10 most important features
            top_features = sorted(
                importance['combined_ranking'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            report.append("Top 10 most important features:")
            for i, (feature, score) in enumerate(top_features, 1):
                report.append(f"  {i:2d}. {feature}: {score:.3f}")
        
        # Recommendations
        if analysis['recommendations']:
            report.append("\n💡 RECOMMENDATIONS")
            report.append("-"*50)
            for i, rec in enumerate(analysis['recommendations'], 1):
                report.append(f"{i}. {rec}")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)
    
    def create_visualization_dashboard(self, data: pd.DataFrame,
                                     analysis: Dict,
                                     save_path: Optional[str] = None):
        """Create comprehensive visualization dashboard"""
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Feature distributions
        ax1 = plt.subplot(3, 4, 1)
        numeric_cols = data.select_dtypes(include=[np.number]).columns[:4]
        for i, col in enumerate(numeric_cols):
            plt.subplot(3, 4, i + 1)
            plt.hist(data[col], bins=30, alpha=0.7)
            plt.title(f'Distribution: {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
        
        # 2. Correlation heatmap
        ax5 = plt.subplot(3, 4, 5)
        numeric_data = data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) > 1:
            corr_matrix = numeric_data.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, cbar_kws={'shrink': 0.8})
            plt.title('Feature Correlation Matrix')
        
        # 3. Outlier box plots
        ax6 = plt.subplot(3, 4, 6)
        outlier_features = [
            f for f, info in analysis['outliers'].items()
            if info['percentage'] > 1  # Features with >1% outliers
        ][:4]
        
        if outlier_features:
            data[outlier_features].boxplot(ax=ax6)
            plt.title('Outlier Detection')
            plt.xticks(rotation=45)
        
        # 4. Missing data visualization
        ax7 = plt.subplot(3, 4, 7)
        missing_data = pd.Series(analysis['missing_data'])
        missing_data = missing_data[missing_data > 0]
        
        if len(missing_data) > 0:
            missing_data.plot(kind='bar', ax=ax7)
            plt.title('Missing Data by Feature')
            plt.xticks(rotation=45)
            plt.ylabel('Missing Count')
        
        # 5. Feature importance (if available)
        ax8 = plt.subplot(3, 4, 8)
        # Simulate feature importance for demo
        feature_importance = {col: np.random.random() for col in numeric_cols[:8]}
        importance_series = pd.Series(feature_importance).sort_values(ascending=True)
        
        importance_series.plot(kind='barh', ax=ax8)
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')
        
        # 6-8. Engagement analysis plots
        if 'engagement_score' in data.columns:
            # Engagement distribution
            ax9 = plt.subplot(3, 4, 9)
            plt.hist(data['engagement_score'], bins=30, alpha=0.7, color='skyblue')
            plt.title('Engagement Score Distribution')
            plt.xlabel('Engagement Score')
            plt.ylabel('Frequency')
            
            # Engagement by content type (if available)
            if 'content_type' in data.columns:
                ax10 = plt.subplot(3, 4, 10)
                data.boxplot(column='engagement_score', by='content_type', ax=ax10)
                plt.title('Engagement by Content Type')
                plt.xticks(rotation=45)
        
        # Time series analysis (if timestamp available)
        if 'timestamp' in data.columns:
            ax11 = plt.subplot(3, 4, 11)
            data['date'] = pd.to_datetime(data['timestamp']).dt.date
            daily_engagement = data.groupby('date')['engagement_score'].mean()
            daily_engagement.plot(ax=ax11)
            plt.title('Engagement Trends Over Time')
            plt.xticks(rotation=45)
        
        # Summary statistics
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')
        
        summary_text = f"""
        Dataset Summary:
        
        • Total Records: {len(data):,}
        • Features: {len(data.columns)}
        • Numeric Features: {len(numeric_data.columns)}
        
        Quality Metrics:
        • Missing Data: {sum(analysis['missing_data'].values())}
        • High Outlier Features: {len([f for f, info in analysis['outliers'].items() if info['percentage'] > 5])}
        • Strong Correlations: {len(analysis['correlations']['strong_positive']) + len(analysis['correlations']['strong_negative'])}
        """
        
        ax12.text(0.1, 0.5, summary_text, transform=ax12.transAxes,
                 fontsize=12, verticalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dashboard saved to: {save_path}")
        
        plt.show()
        
        return fig


def generate_sample_data(n_samples: int = 10000) -> pd.DataFrame:
    """Generate realistic sample data for demonstration"""
    
    np.random.seed(42)
    
    # Generate user features
    user_data = {
        'user_id': [f'user_{i}' for i in range(n_samples)],
        'age': np.random.randint(13, 65, n_samples),
        'platform': np.random.choice(['PS5', 'PS4', 'Mobile'], n_samples, p=[0.4, 0.35, 0.25]),
        'genre_preference': np.random.choice(['Action', 'RPG', 'Sports', 'Racing'], n_samples),
        
        # Video features
        'video_duration': np.random.lognormal(4, 0.8, n_samples),  # Log-normal distribution
        'completion_rate': np.random.beta(2, 3, n_samples),  # Beta distribution
        'quality_score': np.random.uniform(0.3, 1.0, n_samples),
        'engagement_peaks': np.random.poisson(3, n_samples),
        
        # Text features
        'sentiment_score': np.random.normal(0.1, 0.3, n_samples),
        'readability_score': np.random.uniform(0.2, 0.9, n_samples),
        'keyword_overlap': np.random.exponential(0.3, n_samples),
        
        # Image features
        'image_quality': np.random.uniform(0.5, 1.0, n_samples),
        'brightness': np.random.uniform(0.2, 0.8, n_samples),
        'composition_score': np.random.uniform(0.4, 1.0, n_samples),
        
        # Interaction features
        'clicks': np.random.poisson(5, n_samples),
        'likes': np.random.poisson(2, n_samples),
        'shares': np.random.poisson(0.5, n_samples),
        'session_length': np.random.exponential(300, n_samples),  # Seconds
        
        # Target variable
        'engagement_score': np.random.uniform(0, 1, n_samples)
    }
    
    # Add timestamp
    start_date = datetime.now() - timedelta(days=90)
    timestamps = [start_date + timedelta(seconds=np.random.randint(0, 90*24*3600)) 
                  for _ in range(n_samples)]
    user_data['timestamp'] = timestamps
    
    # Add content type
    user_data['content_type'] = np.random.choice(
        ['trailer', 'gameplay', 'review', 'cutscene'], 
        n_samples, 
        p=[0.4, 0.3, 0.2, 0.1]
    )
    
    # Add some realistic correlations
    df = pd.DataFrame(user_data)
    
    # Make engagement correlated with other features
    df['engagement_score'] = (
        0.3 * df['completion_rate'] +
        0.2 * df['quality_score'] +
        0.2 * (df['likes'] / 10) +
        0.1 * df['composition_score'] +
        0.1 * np.random.uniform(0, 1, n_samples) +
        0.1 * (df['session_length'] / 1000)
    )
    df['engagement_score'] = np.clip(df['engagement_score'], 0, 1)
    
    # Add some missing data
    missing_indices = np.random.choice(n_samples, int(0.05 * n_samples), replace=False)
    df.loc[missing_indices, 'readability_score'] = np.nan
    
    missing_indices2 = np.random.choice(n_samples, int(0.02 * n_samples), replace=False)
    df.loc[missing_indices2, 'sentiment_score'] = np.nan
    
    return df


def demo_comprehensive_analysis():
    """Demonstrate comprehensive feature analysis"""
    
    print("🔍 Comprehensive Feature Analysis Demo")
    print("="*60)
    
    # Generate sample data
    print("Generating sample data...")
    data = generate_sample_data(n_samples=5000)
    
    # Initialize analyzer
    analyzer = FeatureAnalyzer()
    
    # Perform analysis
    print("Analyzing feature distributions...")
    feature_analysis = analyzer.analyze_feature_distributions(data)
    
    print("Analyzing user behavior patterns...")
    behavior_patterns = analyzer.analyze_user_behavior_patterns(data)
    
    # Feature importance analysis
    print("Calculating feature importance...")
    numeric_features = data.select_dtypes(include=[np.number]).drop('engagement_score', axis=1)
    feature_names = numeric_features.columns.tolist()
    
    importance_analysis = analyzer.create_feature_importance_analysis(
        numeric_features.values,
        data['engagement_score'].values,
        feature_names
    )
    
    # Generate comprehensive report
    print("\n" + "="*60)
    report = analyzer.generate_feature_report(
        feature_analysis,
        behavior_patterns,
        importance_analysis
    )
    print(report)
    
    # Create visualization dashboard
    print("\nCreating visualization dashboard...")
    analyzer.create_visualization_dashboard(
        data,
        feature_analysis,
        save_path='feature_analysis_dashboard.png'
    )


if __name__ == "__main__":
    demo_comprehensive_analysis()