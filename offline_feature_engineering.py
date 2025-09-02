"""
Offline Feature Engineering Pipeline using Hadoop/Spark
Author: Yang Liu
Description: Large-scale batch feature computation for recommendation system
Experience: Built similar pipeline at Weibo handling 400M+ users daily
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer, VectorAssembler, MinMaxScaler
from pyspark.ml.stat import Correlation
from typing import Dict, List, Optional
import json
from datetime import datetime, timedelta


class OfflineFeatureEngineer:
    """
    Comprehensive offline feature engineering pipeline
    Processes TB-scale data using Spark distributed computing
    """
    
    def __init__(self, app_name: str = "PlayStation_Feature_Engineering"):
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.adaptive.skewJoin.enabled", "true") \
            .config("spark.sql.shuffle.partitions", "400") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        
        # Feature computation windows
        self.time_windows = {
            '1h': '1 hour',
            '6h': '6 hours', 
            '1d': '1 day',
            '3d': '3 days',
            '7d': '7 days',
            '14d': '14 days',
            '30d': '30 days'
        }
        
        # Action weights for engagement scoring
        self.action_weights = {
            'view': 1.0,
            'click': 2.0,
            'add_to_wishlist': 3.0,
            'like': 3.5,
            'share': 4.0,
            'purchase': 10.0,
            'review': 5.0
        }
    
    def compute_user_behavioral_features(self, interactions_df):
        """
        Compute comprehensive user behavioral features
        Input: user_id, item_id, action, timestamp, session_id, platform
        """
        print("Computing user behavioral features...")
        
        # Add derived timestamp features
        interactions_enriched = interactions_df.withColumn(
            "hour", hour(from_unixtime(col("timestamp") / 1000))
        ).withColumn(
            "day_of_week", dayofweek(from_unixtime(col("timestamp") / 1000))
        ).withColumn(
            "date", to_date(from_unixtime(col("timestamp") / 1000))
        ).withColumn(
            "action_weight", 
            when(col("action") == "purchase", self.action_weights['purchase'])
            .when(col("action") == "share", self.action_weights['share'])
            .when(col("action") == "like", self.action_weights['like'])
            .when(col("action") == "click", self.action_weights['click'])
            .when(col("action") == "view", self.action_weights['view'])
            .otherwise(1.0)
        )
        
        # Create time-based features for multiple windows
        user_features_list = []
        
        for window_name, window_duration in self.time_windows.items():
            print(f"Processing {window_name} window features...")
            
            # Filter data for current time window
            window_df = interactions_enriched.filter(
                col("timestamp") >= unix_timestamp(
                    current_timestamp() - expr(f"INTERVAL {window_duration}")
                ) * 1000
            )
            
            # Aggregate features per user for this window
            window_features = window_df.groupBy("user_id").agg(
                # Basic counting features
                count("*").alias(f"total_interactions_{window_name}"),
                countDistinct("item_id").alias(f"unique_items_{window_name}"),
                countDistinct("session_id").alias(f"unique_sessions_{window_name}"),
                countDistinct("date").alias(f"active_days_{window_name}"),
                
                # Engagement features
                sum("action_weight").alias(f"total_engagement_{window_name}"),
                avg("action_weight").alias(f"avg_engagement_{window_name}"),
                stddev("action_weight").alias(f"std_engagement_{window_name}"),
                
                # Action-specific counts
                sum(when(col("action") == "purchase", 1).otherwise(0)).alias(f"purchases_{window_name}"),
                sum(when(col("action") == "view", 1).otherwise(0)).alias(f"views_{window_name}"),
                sum(when(col("action") == "click", 1).otherwise(0)).alias(f"clicks_{window_name}"),
                sum(when(col("action") == "like", 1).otherwise(0)).alias(f"likes_{window_name}"),
                sum(when(col("action") == "share", 1).otherwise(0)).alias(f"shares_{window_name}"),
                
                # Temporal patterns
                collect_set("hour").alias(f"active_hours_{window_name}"),
                collect_set("day_of_week").alias(f"active_days_of_week_{window_name}"),
                
                # Platform diversity
                countDistinct("platform").alias(f"platform_diversity_{window_name}"),
                mode("platform").alias(f"preferred_platform_{window_name}")
            ).withColumn(
                # Derived metrics
                f"conversion_rate_{window_name}",
                col(f"purchases_{window_name}") / (col(f"views_{window_name}") + 1)
            ).withColumn(
                f"engagement_per_session_{window_name}",
                col(f"total_engagement_{window_name}") / (col(f"unique_sessions_{window_name}") + 1)
            ).withColumn(
                f"items_per_session_{window_name}",
                col(f"unique_items_{window_name}") / (col(f"unique_sessions_{window_name}") + 1)
            ).withColumn(
                f"activity_intensity_{window_name}",
                col(f"total_interactions_{window_name}") / (col(f"active_days_{window_name}") + 1)
            )
            
            user_features_list.append(window_features)
        
        # Join all time window features
        user_behavioral_features = user_features_list[0]
        for features_df in user_features_list[1:]:
            user_behavioral_features = user_behavioral_features.join(features_df, "user_id", "outer")
        
        # Add cross-window trend features
        user_behavioral_features = user_behavioral_features.withColumn(
            "engagement_trend_7d_vs_30d",
            col("avg_engagement_7d") / (col("avg_engagement_30d") + 0.001)
        ).withColumn(
            "activity_trend_1d_vs_7d", 
            col("total_interactions_1d") / (col("total_interactions_7d") / 7 + 0.001)
        )
        
        return user_behavioral_features
    
    def compute_item_content_features(self, interactions_df, item_metadata_df):
        """
        Compute item-level content and popularity features
        """
        print("Computing item content features...")
        
        # Item popularity metrics across time windows
        item_features_list = []
        
        for window_name, window_duration in self.time_windows.items():
            window_df = interactions_df.filter(
                col("timestamp") >= unix_timestamp(
                    current_timestamp() - expr(f"INTERVAL {window_duration}")
                ) * 1000
            ).withColumn(
                "action_weight",
                when(col("action") == "purchase", self.action_weights['purchase'])
                .when(col("action") == "view", self.action_weights['view'])
                .otherwise(1.0)
            )
            
            window_item_features = window_df.groupBy("item_id").agg(
                count("*").alias(f"total_interactions_{window_name}"),
                countDistinct("user_id").alias(f"unique_users_{window_name}"),
                sum("action_weight").alias(f"total_engagement_{window_name}"),
                avg("action_weight").alias(f"avg_engagement_{window_name}"),
                
                # Purchase conversion metrics
                sum(when(col("action") == "purchase", 1).otherwise(0)).alias(f"purchases_{window_name}"),
                sum(when(col("action") == "view", 1).otherwise(0)).alias(f"views_{window_name}"),
                
                # User diversity metrics
                stddev("action_weight").alias(f"engagement_std_{window_name}"),
                countDistinct("platform").alias(f"platform_reach_{window_name}")
            ).withColumn(
                f"purchase_rate_{window_name}",
                col(f"purchases_{window_name}") / (col(f"views_{window_name}") + 1)
            ).withColumn(
                f"popularity_score_{window_name}",
                log1p(col(f"total_interactions_{window_name}")) * 
                log1p(col(f"unique_users_{window_name}"))
            )
            
            item_features_list.append(window_item_features)
        
        # Join time window features
        item_features = item_features_list[0]
        for features_df in item_features_list[1:]:
            item_features = item_features.join(features_df, "item_id", "outer")
        
        # Add content-based features from metadata
        if item_metadata_df:
            item_features = item_features.join(item_metadata_df, "item_id", "left")
            
            # Process categorical features
            item_features = item_features.withColumn(
                "genre_count", size(split(col("genres"), ","))
            ).withColumn(
                "tag_count", size(split(col("tags"), ","))  
            ).withColumn(
                "price_category",
                when(col("price") < 20, "budget")
                .when(col("price") < 40, "mid_range") 
                .when(col("price") < 60, "premium")
                .otherwise("deluxe")
            )
        
        # Add trend features
        item_features = item_features.withColumn(
            "popularity_trend_7d_vs_30d",
            col("popularity_score_7d") / (col("popularity_score_30d") + 0.001)
        ).withColumn(
            "engagement_trend_1d_vs_7d",
            col("avg_engagement_1d") / (col("avg_engagement_7d") + 0.001)
        )
        
        return item_features
    
    def compute_user_item_interaction_features(self, interactions_df):
        """
        Compute user-item pair specific features
        """
        print("Computing user-item interaction features...")
        
        # Create user-item pairs with aggregated features
        user_item_features = interactions_df.groupBy("user_id", "item_id").agg(
            count("*").alias("interaction_count"),
            countDistinct("session_id").alias("session_count"),
            sum(when(col("action") == "purchase", 1).otherwise(0)).alias("purchase_count"),
            sum(when(col("action") == "view", 1).otherwise(0)).alias("view_count"),
            sum(when(col("action") == "like", 1).otherwise(0)).alias("like_count"),
            
            # Temporal features
            min("timestamp").alias("first_interaction_time"),
            max("timestamp").alias("last_interaction_time"),
            collect_list("action").alias("action_sequence"),
            
            # Session-based features
            avg("session_length").alias("avg_session_length"),
            countDistinct("platform").alias("platform_diversity")
        ).withColumn(
            "interaction_span_days",
            datediff(
                from_unixtime(col("last_interaction_time") / 1000),
                from_unixtime(col("first_interaction_time") / 1000)
            )
        ).withColumn(
            "purchase_conversion_rate",
            col("purchase_count") / (col("view_count") + 1)
        ).withColumn(
            "engagement_intensity", 
            col("interaction_count") / (col("interaction_span_days") + 1)
        )
        
        # Add sequence-based features
        user_item_features = user_item_features.withColumn(
            "action_sequence_length", size(col("action_sequence"))
        ).withColumn(
            "has_purchase_sequence", 
            array_contains(col("action_sequence"), "purchase")
        ).withColumn(
            "last_action",
            element_at(col("action_sequence"), -1)
        )
        
        return user_item_features
    
    def compute_contextual_features(self, interactions_df):
        """
        Compute contextual features (time, platform, session)
        """
        print("Computing contextual features...")
        
        # Add rich temporal context
        contextual_df = interactions_df.withColumn(
            "hour_of_day", hour(from_unixtime(col("timestamp") / 1000))
        ).withColumn(
            "day_of_week", dayofweek(from_unixtime(col("timestamp") / 1000))
        ).withColumn(
            "is_weekend", 
            when(col("day_of_week").isin([1, 7]), 1).otherwise(0)  # Sunday=1, Saturday=7
        ).withColumn(
            "time_of_day_category",
            when(col("hour_of_day").between(6, 11), "morning")
            .when(col("hour_of_day").between(12, 17), "afternoon")
            .when(col("hour_of_day").between(18, 22), "evening")
            .otherwise("night")
        )
        
        # Session-level contextual features
        session_window = Window.partitionBy("session_id").orderBy("timestamp")
        
        contextual_features = contextual_df.withColumn(
            "session_position", row_number().over(session_window)
        ).withColumn(
            "time_since_session_start",
            col("timestamp") - first(col("timestamp")).over(
                Window.partitionBy("session_id").orderBy("timestamp")
                .rowsBetween(Window.unboundedPreceding, Window.currentRow)
            )
        ).withColumn(
            "time_since_last_action",
            col("timestamp") - lag(col("timestamp"), 1).over(
                Window.partitionBy("user_id").orderBy("timestamp")
            )
        )
        
        # Platform and device context
        contextual_features = contextual_features.withColumn(
            "is_mobile", when(col("platform") == "Mobile", 1).otherwise(0)
        ).withColumn(
            "is_console", when(col("platform").isin(["PS5", "PS4"]), 1).otherwise(0)
        )
        
        return contextual_features
    
    def compute_cross_features(self, user_features, item_features, 
                              user_item_features, contextual_features):
        """
        Compute cross-domain interaction features
        """
        print("Computing cross-domain features...")
        
        # Join all feature sets
        cross_features = contextual_features \
            .join(user_features, "user_id", "left") \
            .join(item_features, "item_id", "left") \
            .join(user_item_features, ["user_id", "item_id"], "left")
        
        # Compute interaction features
        cross_features = cross_features.withColumn(
            # User-item affinity score
            "user_item_affinity",
            coalesce(col("interaction_count"), lit(0)) * 
            log1p(coalesce(col("unique_users_7d"), lit(1)))
        ).withColumn(
            # Popularity vs user preference match
            "popularity_preference_match",
            when(
                (col("total_interactions_7d") > col("avg_engagement_7d")) &
                (col("popularity_score_7d") > percentile_approx(col("popularity_score_7d"), 0.8)),
                1.0
            ).otherwise(0.0)
        ).withColumn(
            # Platform consistency score
            "platform_consistency",
            when(col("platform") == col("preferred_platform_7d"), 1.0).otherwise(0.5)
        ).withColumn(
            # Time pattern alignment
            "temporal_alignment",
            when(
                array_contains(coalesce(col("active_hours_7d"), array()), col("hour_of_day")),
                1.0
            ).otherwise(0.3)
        )
        
        # Feature combinations for deep learning
        cross_features = cross_features.withColumn(
            "user_activity_x_item_popularity",
            coalesce(col("total_interactions_7d"), lit(0)) * 
            coalesce(col("popularity_score_7d"), lit(0))
        ).withColumn(
            "conversion_rate_x_engagement",
            coalesce(col("conversion_rate_7d"), lit(0)) *
            coalesce(col("avg_engagement_7d"), lit(0))
        )
        
        return cross_features
    
    def run_feature_engineering_pipeline(self, 
                                        interactions_path: str,
                                        item_metadata_path: str,
                                        output_path: str):
        """
        Run complete offline feature engineering pipeline
        """
        print("Starting offline feature engineering pipeline...")
        start_time = datetime.now()
        
        # Load raw data
        print("Loading interaction data...")
        interactions_df = self.spark.read.parquet(interactions_path)
        
        print("Loading item metadata...")
        item_metadata_df = self.spark.read.parquet(item_metadata_path) if item_metadata_path else None
        
        # Compute feature sets
        print("\n=== Computing Feature Sets ===")
        
        # 1. User behavioral features
        user_features = self.compute_user_behavioral_features(interactions_df)
        user_features.write.mode("overwrite").parquet(f"{output_path}/user_features")
        
        # 2. Item content features  
        item_features = self.compute_item_content_features(interactions_df, item_metadata_df)
        item_features.write.mode("overwrite").parquet(f"{output_path}/item_features")
        
        # 3. User-item interaction features
        user_item_features = self.compute_user_item_interaction_features(interactions_df)
        user_item_features.write.mode("overwrite").parquet(f"{output_path}/user_item_features")
        
        # 4. Contextual features
        contextual_features = self.compute_contextual_features(interactions_df)
        contextual_features.write.mode("overwrite").parquet(f"{output_path}/contextual_features")
        
        # 5. Cross-domain features
        cross_features = self.compute_cross_features(
            user_features, item_features, user_item_features, contextual_features
        )
        cross_features.write.mode("overwrite").partitionBy("date").parquet(f"{output_path}/cross_features")
        
        # Generate feature statistics
        self.generate_feature_stats(
            user_features, item_features, user_item_features, 
            f"{output_path}/feature_stats"
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n=== Pipeline Completed ===")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Output saved to: {output_path}")
        
        return cross_features
    
    def generate_feature_stats(self, user_features, item_features, 
                              user_item_features, stats_output_path):
        """
        Generate comprehensive feature statistics for monitoring
        """
        print("Generating feature statistics...")
        
        stats = {}
        
        # User feature stats
        user_stats = user_features.select([
            count("*").alias("total_users"),
            avg("total_interactions_7d").alias("avg_user_interactions_7d"),
            stddev("total_interactions_7d").alias("std_user_interactions_7d"),
            min("total_interactions_7d").alias("min_user_interactions_7d"), 
            max("total_interactions_7d").alias("max_user_interactions_7d")
        ]).collect()[0]
        
        stats['user_features'] = user_stats.asDict()
        
        # Item feature stats
        item_stats = item_features.select([
            count("*").alias("total_items"),
            avg("popularity_score_7d").alias("avg_item_popularity_7d"),
            stddev("popularity_score_7d").alias("std_item_popularity_7d")
        ]).collect()[0]
        
        stats['item_features'] = item_stats.asDict()
        
        # Save stats as JSON
        stats_df = self.spark.sparkContext.parallelize([json.dumps(stats)]).toDF(["stats"])
        stats_df.write.mode("overwrite").text(stats_output_path)
        
        print("Feature statistics generated")
        
    def create_training_dataset(self, features_df, negative_sampling_ratio: float = 4.0):
        """
        Create training dataset with positive/negative sampling
        """
        print("Creating training dataset...")
        
        # Positive samples (interactions that occurred)
        positive_samples = features_df.filter(col("interaction_count") > 0).withColumn("label", lit(1.0))
        
        # Negative sampling (users who didn't interact with items)
        # Simple random sampling approach
        all_users = features_df.select("user_id").distinct()
        all_items = features_df.select("item_id").distinct()
        
        # Cross join for all possible user-item pairs (be careful with large datasets!)
        all_pairs = all_users.crossJoin(all_items)
        
        # Subtract positive samples to get negative candidates
        negative_candidates = all_pairs.join(
            positive_samples.select("user_id", "item_id"),
            ["user_id", "item_id"],
            "left_anti"
        )
        
        # Sample negative examples
        negative_sample_count = int(positive_samples.count() * negative_sampling_ratio)
        negative_samples = negative_candidates.sample(
            False, 
            min(1.0, negative_sample_count / negative_candidates.count())
        ).limit(negative_sample_count).withColumn("label", lit(0.0))
        
        # Combine positive and negative samples
        training_data = positive_samples.union(negative_samples)
        
        print(f"Training dataset created:")
        print(f"  Positive samples: {positive_samples.count()}")
        print(f"  Negative samples: {negative_samples.count()}")
        
        return training_data


def create_sample_offline_data(spark, output_path: str):
    """
    Create sample data for offline feature engineering demo
    """
    print("Creating sample offline data...")
    
    import random
    from datetime import datetime, timedelta
    
    # Generate sample interactions (larger scale)
    n_users = 100000
    n_items = 10000  
    n_interactions = 5000000
    
    interactions_data = []
    start_date = datetime.now() - timedelta(days=60)
    
    for i in range(n_interactions):
        user_id = f"user_{random.randint(1, n_users)}"
        item_id = f"game_{random.randint(1, n_items)}"
        action = random.choices(
            ['view', 'click', 'like', 'share', 'purchase'],
            weights=[50, 30, 10, 5, 5]
        )[0]
        
        timestamp = int((start_date + timedelta(
            seconds=random.randint(0, 60*24*3600)
        )).timestamp() * 1000)
        
        session_id = f"session_{random.randint(1, n_interactions//20)}"
        platform = random.choices(['PS5', 'PS4', 'Mobile'], weights=[40, 35, 25])[0]
        session_length = random.randint(60, 3600)
        
        interactions_data.append({
            "user_id": user_id,
            "item_id": item_id,
            "action": action,
            "timestamp": timestamp,
            "session_id": session_id,
            "platform": platform,
            "session_length": session_length
        })
        
        if (i + 1) % 100000 == 0:
            print(f"Generated {i + 1} interactions...")
    
    # Create DataFrame and save
    interactions_df = spark.createDataFrame(interactions_data)
    interactions_df.write.mode("overwrite").parquet(f"{output_path}/raw_interactions")
    
    # Generate sample item metadata
    item_metadata = []
    genres = ['Action', 'RPG', 'Sports', 'Racing', 'Adventure', 'Puzzle', 'Strategy', 'Horror']
    
    for i in range(1, n_items + 1):
        item_genres = random.sample(genres, random.randint(1, 3))
        tags = [f"tag_{random.randint(1, 50)}" for _ in range(random.randint(2, 8))]
        
        item_metadata.append({
            "item_id": f"game_{i}",
            "title": f"Game {i}",
            "genres": ",".join(item_genres),
            "tags": ",".join(tags),
            "price": random.randint(10, 70),
            "release_date": (datetime.now() - timedelta(
                days=random.randint(30, 1095)
            )).strftime("%Y-%m-%d"),
            "rating": random.choice(['E', 'E10+', 'T', 'M'])
        })
    
    item_metadata_df = spark.createDataFrame(item_metadata)
    item_metadata_df.write.mode("overwrite").parquet(f"{output_path}/item_metadata")
    
    print(f"Sample data created at {output_path}")
    return interactions_df, item_metadata_df


def demo_offline_feature_engineering():
    """
    Demonstrate offline feature engineering pipeline
    """
    print("🏭 Offline Feature Engineering Pipeline Demo")
    print("="*60)
    
    # Initialize feature engineer
    engineer = OfflineFeatureEngineer()
    
    # Create sample data
    sample_data_path = "/tmp/sample_offline_data"
    interactions_df, item_metadata_df = create_sample_offline_data(
        engineer.spark, sample_data_path
    )
    
    # Run feature engineering pipeline
    output_path = "/tmp/offline_features"
    
    final_features = engineer.run_feature_engineering_pipeline(
        interactions_path=f"{sample_data_path}/raw_interactions",
        item_metadata_path=f"{sample_data_path}/item_metadata",
        output_path=output_path
    )
    
    # Show sample results
    print("\n📊 Sample Feature Results:")
    final_features.select(
        "user_id", "item_id", "total_interactions_7d", 
        "popularity_score_7d", "user_item_affinity",
        "platform_consistency", "temporal_alignment"
    ).show(10, truncate=False)
    
    # Create training dataset
    training_data = engineer.create_training_dataset(final_features)
    training_data.write.mode("overwrite").parquet(f"{output_path}/training_data")
    
    print(f"\n✅ Pipeline completed successfully!")
    print(f"Features saved to: {output_path}")
    print(f"Ready for real-time feature fusion with Flink!")
    
    # Stop Spark
    engineer.spark.stop()


if __name__ == "__main__":
    demo_offline_feature_engineering()