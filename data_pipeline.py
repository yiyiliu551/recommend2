"""
Distributed Data Processing Pipeline
Author: Yang Liu
Description: Production-ready data pipeline using PySpark for recommendation systems
Experience: Used at Weibo (400M DAU), Qunar, CNKI
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
from typing import Dict, List, Optional
import json
from datetime import datetime, timedelta


class RecommendationDataPipeline:
    """
    Scalable data pipeline for processing user interactions and generating features
    Handles billions of events daily
    """
    
    def __init__(self, app_name: str = "PlayStation_Recommendation_Pipeline"):
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.shuffle.partitions", "200") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        
    def process_user_interactions(self, input_path: str, output_path: str):
        """
        Process raw user interaction logs
        
        Input schema:
        - user_id: string
        - item_id: string  
        - action: string (view, click, purchase, like, share)
        - timestamp: long
        - session_id: string
        - platform: string (PS5, PS4, Mobile, Web)
        """
        print(f"Processing user interactions from {input_path}")
        
        # Read raw data
        df = self.spark.read.json(input_path)
        
        # Data cleaning and validation
        df_cleaned = df.filter(
            (col("user_id").isNotNull()) & 
            (col("item_id").isNotNull()) &
            (col("timestamp") > 0)
        )
        
        # Add derived columns
        df_processed = df_cleaned.withColumn(
            "date", from_unixtime(col("timestamp") / 1000).cast("date")
        ).withColumn(
            "hour", hour(from_unixtime(col("timestamp") / 1000))
        ).withColumn(
            "day_of_week", dayofweek(from_unixtime(col("timestamp") / 1000))
        ).withColumn(
            "action_weight", 
            when(col("action") == "purchase", 5.0)
            .when(col("action") == "like", 3.0)
            .when(col("action") == "click", 2.0)
            .when(col("action") == "view", 1.0)
            .otherwise(0.5)
        )
        
        # Deduplicate
        df_dedup = df_processed.dropDuplicates(["user_id", "item_id", "timestamp"])
        
        # Save processed data
        df_dedup.write \
            .mode("overwrite") \
            .partitionBy("date") \
            .parquet(output_path)
        
        print(f"Processed {df_dedup.count()} interactions")
        return df_dedup
    
    def generate_user_features(self, interactions_df):
        """
        Generate user-level features for ML models
        """
        print("Generating user features...")
        
        # User activity metrics
        user_activity = interactions_df.groupBy("user_id").agg(
            count("*").alias("total_interactions"),
            countDistinct("item_id").alias("unique_items"),
            countDistinct("session_id").alias("total_sessions"),
            avg("action_weight").alias("avg_action_weight"),
            stddev("action_weight").alias("stddev_action_weight"),
            
            # Action-specific counts
            sum(when(col("action") == "purchase", 1).otherwise(0)).alias("purchase_count"),
            sum(when(col("action") == "view", 1).otherwise(0)).alias("view_count"),
            sum(when(col("action") == "click", 1).otherwise(0)).alias("click_count"),
            
            # Time-based features
            min("timestamp").alias("first_interaction_time"),
            max("timestamp").alias("last_interaction_time"),
            countDistinct("date").alias("active_days"),
            
            # Platform preferences
            mode("platform").alias("preferred_platform")
        )
        
        # Calculate derived metrics
        user_features = user_activity.withColumn(
            "conversion_rate",
            col("purchase_count") / (col("view_count") + 1)  # Add 1 to avoid division by zero
        ).withColumn(
            "engagement_score",
            col("total_interactions") / (col("active_days") + 1)
        ).withColumn(
            "diversity_score",
            col("unique_items") / (col("total_interactions") + 1)
        ).withColumn(
            "user_lifetime_days",
            datediff(
                from_unixtime(col("last_interaction_time") / 1000),
                from_unixtime(col("first_interaction_time") / 1000)
            )
        )
        
        return user_features
    
    def generate_item_features(self, interactions_df):
        """
        Generate item-level features for ML models
        """
        print("Generating item features...")
        
        # Item popularity metrics
        item_stats = interactions_df.groupBy("item_id").agg(
            count("*").alias("total_interactions"),
            countDistinct("user_id").alias("unique_users"),
            avg("action_weight").alias("avg_engagement"),
            
            # Action breakdown
            sum(when(col("action") == "purchase", 1).otherwise(0)).alias("purchases"),
            sum(when(col("action") == "view", 1).otherwise(0)).alias("views"),
            sum(when(col("action") == "like", 1).otherwise(0)).alias("likes"),
            
            # Time patterns
            collect_list("hour").alias("interaction_hours"),
            collect_list("day_of_week").alias("interaction_days")
        )
        
        # Calculate popularity score with time decay
        window_spec = Window.partitionBy("item_id").orderBy(desc("timestamp"))
        
        recent_interactions = interactions_df \
            .withColumn("rank", row_number().over(window_spec)) \
            .filter(col("rank") <= 1000)  # Last 1000 interactions
        
        recent_stats = recent_interactions.groupBy("item_id").agg(
            avg("action_weight").alias("recent_engagement"),
            count("*").alias("recent_interaction_count")
        )
        
        # Join with overall stats
        item_features = item_stats.join(recent_stats, "item_id", "left") \
            .withColumn(
                "popularity_score",
                col("total_interactions") * 0.3 + 
                col("unique_users") * 0.3 +
                coalesce(col("recent_interaction_count"), lit(0)) * 0.4
            ).withColumn(
                "conversion_rate",
                col("purchases") / (col("views") + 1)
            )
        
        return item_features
    
    def create_user_item_matrix(self, interactions_df):
        """
        Create sparse user-item interaction matrix for collaborative filtering
        """
        print("Creating user-item matrix...")
        
        # Aggregate interactions per user-item pair
        user_item_df = interactions_df.groupBy("user_id", "item_id").agg(
            sum("action_weight").alias("interaction_score"),
            count("*").alias("interaction_count"),
            max("timestamp").alias("last_interaction")
        )
        
        # Apply time decay
        current_time = datetime.now().timestamp() * 1000
        decay_factor = 86400000  # 1 day in milliseconds
        
        user_item_matrix = user_item_df.withColumn(
            "time_decay_factor",
            exp(-(lit(current_time) - col("last_interaction")) / decay_factor)
        ).withColumn(
            "weighted_score",
            col("interaction_score") * col("time_decay_factor")
        )
        
        return user_item_matrix
    
    def generate_sequence_features(self, interactions_df):
        """
        Generate sequence-based features for deep learning models
        """
        print("Generating sequence features...")
        
        # Window for user sequences
        user_window = Window.partitionBy("user_id").orderBy("timestamp")
        
        # Create sequences
        sequences = interactions_df \
            .withColumn("prev_item", lag("item_id", 1).over(user_window)) \
            .withColumn("next_item", lead("item_id", 1).over(user_window)) \
            .withColumn("time_since_last", 
                       col("timestamp") - lag("timestamp", 1).over(user_window)) \
            .withColumn("session_position", row_number().over(
                Window.partitionBy("user_id", "session_id").orderBy("timestamp")
            ))
        
        # Aggregate sequence patterns
        sequence_features = sequences.groupBy("user_id").agg(
            collect_list("item_id").alias("item_sequence"),
            collect_list("action").alias("action_sequence"),
            avg("time_since_last").alias("avg_time_between_actions"),
            stddev("time_since_last").alias("stddev_time_between_actions")
        ).withColumn(
            "sequence_length", size(col("item_sequence"))
        ).withColumn(
            "last_5_items", slice(col("item_sequence"), -5, 5)
        )
        
        return sequence_features
    
    def calculate_item_similarity(self, user_item_matrix):
        """
        Calculate item-item similarity using cosine similarity
        """
        print("Calculating item similarities...")
        
        # Self-join to create item pairs
        matrix1 = user_item_matrix.select(
            col("user_id"),
            col("item_id").alias("item1"),
            col("weighted_score").alias("score1")
        )
        
        matrix2 = user_item_matrix.select(
            col("user_id"),
            col("item_id").alias("item2"),
            col("weighted_score").alias("score2")
        )
        
        # Join on user to get co-occurrences
        item_pairs = matrix1.join(matrix2, "user_id") \
            .filter(col("item1") < col("item2"))  # Avoid duplicates
        
        # Calculate cosine similarity
        similarity = item_pairs.groupBy("item1", "item2").agg(
            sum(col("score1") * col("score2")).alias("dot_product"),
            sum(col("score1") * col("score1")).alias("norm1_sq"),
            sum(col("score2") * col("score2")).alias("norm2_sq"),
            count("user_id").alias("co_occurrence_count")
        ).withColumn(
            "cosine_similarity",
            col("dot_product") / (sqrt(col("norm1_sq")) * sqrt(col("norm2_sq")))
        ).filter(
            col("co_occurrence_count") >= 5  # Minimum co-occurrences
        )
        
        return similarity
    
    def generate_training_data(self, interactions_df, user_features, item_features):
        """
        Generate training data for ML models
        """
        print("Generating training data...")
        
        # Join all features
        training_data = interactions_df \
            .join(user_features, "user_id") \
            .join(item_features, "item_id")
        
        # Create label (1 for positive interaction, 0 for negative)
        training_data = training_data.withColumn(
            "label",
            when(col("action").isin(["purchase", "like"]), 1.0).otherwise(0.0)
        )
        
        # Select features for model
        feature_cols = [
            "user_lifetime_days", "engagement_score", "diversity_score",
            "item_popularity_score", "item_conversion_rate",
            "hour", "day_of_week", "action_weight"
        ]
        
        # Assemble features
        from pyspark.ml.feature import VectorAssembler
        
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features"
        )
        
        final_data = assembler.transform(training_data) \
            .select("user_id", "item_id", "features", "label", "timestamp")
        
        # Split into train/test
        train_data = final_data.filter(col("timestamp") < lit(1609459200000))  # Before 2021
        test_data = final_data.filter(col("timestamp") >= lit(1609459200000))
        
        return train_data, test_data
    
    def run_pipeline(self, input_path: str, output_base_path: str):
        """
        Run complete data processing pipeline
        """
        print("Starting data processing pipeline...")
        start_time = datetime.now()
        
        # Process interactions
        interactions = self.process_user_interactions(
            input_path, 
            f"{output_base_path}/processed_interactions"
        )
        
        # Generate features
        user_features = self.generate_user_features(interactions)
        user_features.write.mode("overwrite").parquet(f"{output_base_path}/user_features")
        
        item_features = self.generate_item_features(interactions)
        item_features.write.mode("overwrite").parquet(f"{output_base_path}/item_features")
        
        # Create matrices
        user_item_matrix = self.create_user_item_matrix(interactions)
        user_item_matrix.write.mode("overwrite").parquet(f"{output_base_path}/user_item_matrix")
        
        # Calculate similarities
        item_similarity = self.calculate_item_similarity(user_item_matrix)
        item_similarity.write.mode("overwrite").parquet(f"{output_base_path}/item_similarity")
        
        # Generate training data
        train, test = self.generate_training_data(interactions, user_features, item_features)
        train.write.mode("overwrite").parquet(f"{output_base_path}/train_data")
        test.write.mode("overwrite").parquet(f"{output_base_path}/test_data")
        
        end_time = datetime.now()
        print(f"Pipeline completed in {(end_time - start_time).seconds} seconds")
        
        # Print statistics
        print("\nPipeline Statistics:")
        print(f"Total interactions: {interactions.count()}")
        print(f"Unique users: {user_features.count()}")
        print(f"Unique items: {item_features.count()}")
        print(f"Training samples: {train.count()}")
        print(f"Test samples: {test.count()}")
    
    def stop(self):
        """Stop Spark session"""
        self.spark.stop()


def create_sample_data(spark, num_users=10000, num_items=5000, num_interactions=1000000):
    """
    Create sample interaction data for testing
    """
    import random
    from datetime import datetime, timedelta
    
    print(f"Creating sample data: {num_users} users, {num_items} items, {num_interactions} interactions")
    
    actions = ["view", "click", "purchase", "like", "share"]
    platforms = ["PS5", "PS4", "Mobile", "Web"]
    
    data = []
    start_date = datetime.now() - timedelta(days=365)
    
    for _ in range(num_interactions):
        user_id = f"user_{random.randint(1, num_users)}"
        item_id = f"game_{random.randint(1, num_items)}"
        action = random.choice(actions)
        timestamp = int((start_date + timedelta(
            seconds=random.randint(0, 365*24*3600)
        )).timestamp() * 1000)
        session_id = f"session_{random.randint(1, num_interactions//10)}"
        platform = random.choice(platforms)
        
        data.append({
            "user_id": user_id,
            "item_id": item_id,
            "action": action,
            "timestamp": timestamp,
            "session_id": session_id,
            "platform": platform
        })
    
    df = spark.createDataFrame(data)
    return df


if __name__ == "__main__":
    # Initialize pipeline
    pipeline = RecommendationDataPipeline()
    
    # Create sample data
    sample_data = create_sample_data(pipeline.spark, 
                                    num_users=1000, 
                                    num_items=500, 
                                    num_interactions=100000)
    
    # Save sample data
    sample_data.write.mode("overwrite").json("/tmp/sample_interactions")
    
    # Run pipeline
    pipeline.run_pipeline("/tmp/sample_interactions", "/tmp/pipeline_output")
    
    # Stop Spark
    pipeline.stop()