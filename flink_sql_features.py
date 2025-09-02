#!/usr/bin/env python3
"""
Flink SQL Implementation for Real-time Feature Engineering
Production-ready Flink SQL for e-commerce recommendation
Author: Yang Liu
"""

from pyflink.table import StreamTableEnvironment, EnvironmentSettings
from pyflink.table.expressions import col, lit, call
from pyflink.table.window import Tumble, Slide, Session
from pyflink.table.udf import udf, udtf, udaf
from pyflink.common import Row
import pandas as pd
from typing import Iterator


def create_flink_sql_job():
    """Create Flink SQL streaming job for feature computation"""
    
    # Create streaming environment
    env_settings = EnvironmentSettings.new_instance() \
        .in_streaming_mode() \
        .use_blink_planner() \
        .build()
    
    table_env = StreamTableEnvironment.create(environment_settings=env_settings)
    
    # Configure job parameters
    table_env.get_config().get_configuration().set_string(
        "pipeline.time-characteristic", "EventTime"
    )
    table_env.get_config().get_configuration().set_string(
        "execution.checkpointing.interval", "60000"
    )
    
    # Create Kafka source table
    table_env.execute_sql("""
        CREATE TABLE user_events (
            user_id STRING,
            item_id STRING,
            category_id INT,
            brand_id INT,
            action STRING,
            price DECIMAL(10, 2),
            session_id STRING,
            event_time TIMESTAMP(3),
            WATERMARK FOR event_time AS event_time - INTERVAL '5' SECOND
        ) WITH (
            'connector' = 'kafka',
            'topic' = 'user-events',
            'properties.bootstrap.servers' = 'localhost:9092',
            'properties.group.id' = 'flink-feature-group',
            'format' = 'json',
            'scan.startup.mode' = 'latest-offset'
        )
    """)
    
    # Create dimension table for item info (from HBase/Redis)
    table_env.execute_sql("""
        CREATE TABLE item_info (
            item_id STRING,
            category_id INT,
            brand_id INT,
            price DECIMAL(10, 2),
            release_date DATE,
            PRIMARY KEY (item_id) NOT ENFORCED
        ) WITH (
            'connector' = 'hbase-2.2',
            'table-name' = 'item_info',
            'zookeeper.quorum' = 'localhost:2181'
        )
    """)
    
    # 1. Real-time user behavior statistics (5-minute window)
    table_env.execute_sql("""
        CREATE VIEW user_stats_5min AS
        SELECT 
            user_id,
            window_start,
            window_end,
            COUNT(*) as total_actions,
            COUNT(DISTINCT item_id) as unique_items,
            COUNT(DISTINCT session_id) as sessions,
            SUM(CASE WHEN action = 'view' THEN 1 ELSE 0 END) as views,
            SUM(CASE WHEN action = 'click' THEN 1 ELSE 0 END) as clicks,
            SUM(CASE WHEN action = 'purchase' THEN 1 ELSE 0 END) as purchases,
            SUM(CASE WHEN action = 'add_cart' THEN 1 ELSE 0 END) as add_carts,
            AVG(price) as avg_price,
            MAX(event_time) as last_action_time
        FROM TABLE(
            TUMBLE(TABLE user_events, DESCRIPTOR(event_time), INTERVAL '5' MINUTES)
        )
        GROUP BY user_id, window_start, window_end
    """)
    
    # 2. Sliding window user engagement (30-minute window, 5-minute slide)
    table_env.execute_sql("""
        CREATE VIEW user_engagement_30min AS
        SELECT 
            user_id,
            window_start,
            window_end,
            COUNT(*) as action_count,
            COUNT(DISTINCT item_id) as item_diversity,
            SUM(CASE 
                WHEN action = 'view' THEN 1
                WHEN action = 'click' THEN 2
                WHEN action = 'add_cart' THEN 3
                WHEN action = 'purchase' THEN 10
                ELSE 0
            END) as engagement_score,
            COUNT(CASE WHEN action = 'purchase' THEN 1 END) * 1.0 / 
                NULLIF(COUNT(*), 0) as purchase_rate,
            LISTAGG(DISTINCT category_id) as browsed_categories
        FROM TABLE(
            HOP(TABLE user_events, DESCRIPTOR(event_time), 
                INTERVAL '5' MINUTES, INTERVAL '30' MINUTES)
        )
        GROUP BY user_id, window_start, window_end
    """)
    
    # 3. Item popularity in real-time
    table_env.execute_sql("""
        CREATE VIEW item_popularity_realtime AS
        SELECT 
            item_id,
            window_start,
            window_end,
            COUNT(DISTINCT user_id) as unique_users,
            COUNT(*) as total_interactions,
            SUM(CASE WHEN action = 'view' THEN 1 ELSE 0 END) as views,
            SUM(CASE WHEN action = 'click' THEN 1 ELSE 0 END) as clicks,
            SUM(CASE WHEN action = 'purchase' THEN 1 ELSE 0 END) as purchases,
            CAST(SUM(CASE WHEN action = 'click' THEN 1 ELSE 0 END) AS DOUBLE) / 
                NULLIF(SUM(CASE WHEN action = 'view' THEN 1 ELSE 0 END), 0) as ctr,
            CAST(SUM(CASE WHEN action = 'purchase' THEN 1 ELSE 0 END) AS DOUBLE) / 
                NULLIF(SUM(CASE WHEN action = 'view' THEN 1 ELSE 0 END), 0) as cvr
        FROM TABLE(
            TUMBLE(TABLE user_events, DESCRIPTOR(event_time), INTERVAL '10' MINUTES)
        )
        GROUP BY item_id, window_start, window_end
    """)
    
    # 4. Session-based features
    table_env.execute_sql("""
        CREATE VIEW session_features AS
        SELECT 
            session_id,
            user_id,
            window_start,
            window_end,
            COUNT(*) as session_length,
            COUNT(DISTINCT item_id) as items_viewed,
            TIMESTAMPDIFF(SECOND, MIN(event_time), MAX(event_time)) as duration_seconds,
            STRING_AGG(action, ',') as action_sequence,
            BOOL_OR(action = 'purchase') as has_purchase
        FROM TABLE(
            SESSION(TABLE user_events PARTITION BY session_id, 
                    DESCRIPTOR(event_time), INTERVAL '30' MINUTES)
        )
        GROUP BY session_id, user_id, window_start, window_end
    """)
    
    # 5. User-Item cross features with JOIN
    table_env.execute_sql("""
        CREATE VIEW user_item_cross_features AS
        SELECT 
            e.user_id,
            e.item_id,
            e.event_time,
            u.engagement_score as user_engagement,
            i.ctr as item_ctr,
            i.cvr as item_cvr,
            u.item_diversity as user_diversity,
            i.unique_users as item_popularity,
            u.engagement_score * i.ctr as cross_score,
            ABS(u.avg_price - e.price) as price_diff
        FROM user_events e
        LEFT JOIN LATERAL TABLE(
            SELECT * FROM user_engagement_30min
            WHERE user_id = e.user_id
            AND e.event_time BETWEEN window_start AND window_end
        ) u ON TRUE
        LEFT JOIN LATERAL TABLE(
            SELECT * FROM item_popularity_realtime
            WHERE item_id = e.item_id
            AND e.event_time BETWEEN window_start AND window_end
        ) i ON TRUE
    """)
    
    # 6. Category-level statistics
    table_env.execute_sql("""
        CREATE VIEW category_trends AS
        SELECT 
            category_id,
            window_start,
            window_end,
            COUNT(DISTINCT user_id) as category_users,
            COUNT(DISTINCT item_id) as category_items,
            COUNT(*) as category_interactions,
            AVG(price) as avg_category_price,
            STDDEV(price) as price_variance,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price) as median_price
        FROM TABLE(
            TUMBLE(TABLE user_events, DESCRIPTOR(event_time), INTERVAL '1' HOUR)
        )
        GROUP BY category_id, window_start, window_end
    """)
    
    # 7. Real-time A/B test metrics
    table_env.execute_sql("""
        CREATE VIEW ab_test_metrics AS
        SELECT
            user_id,
            CASE 
                WHEN MOD(HASH_CODE(user_id), 2) = 0 THEN 'control'
                ELSE 'experiment'
            END as test_group,
            window_start,
            window_end,
            COUNT(CASE WHEN action = 'click' THEN 1 END) * 1.0 / 
                NULLIF(COUNT(CASE WHEN action = 'view' THEN 1 END), 0) as ctr,
            COUNT(CASE WHEN action = 'purchase' THEN 1 END) * 1.0 / 
                NULLIF(COUNT(CASE WHEN action = 'click' THEN 1 END), 0) as cvr,
            AVG(price) as avg_transaction_value
        FROM TABLE(
            TUMBLE(TABLE user_events, DESCRIPTOR(event_time), INTERVAL '1' HOUR)
        )
        GROUP BY user_id, window_start, window_end
    """)
    
    # Output to Kafka sink
    table_env.execute_sql("""
        CREATE TABLE feature_output (
            feature_type STRING,
            feature_key STRING,
            feature_value STRING,
            update_time TIMESTAMP(3)
        ) WITH (
            'connector' = 'kafka',
            'topic' = 'realtime-features',
            'properties.bootstrap.servers' = 'localhost:9092',
            'format' = 'json',
            'sink.partitioner' = 'round-robin'
        )
    """)
    
    # Write user features to output
    table_env.execute_sql("""
        INSERT INTO feature_output
        SELECT 
            'user_engagement' as feature_type,
            user_id as feature_key,
            JSON_OBJECT(
                'engagement_score', engagement_score,
                'item_diversity', item_diversity,
                'purchase_rate', purchase_rate,
                'window_start', CAST(window_start AS STRING),
                'window_end', CAST(window_end AS STRING)
            ) as feature_value,
            window_end as update_time
        FROM user_engagement_30min
    """)
    
    # Write item features to output
    table_env.execute_sql("""
        INSERT INTO feature_output
        SELECT 
            'item_popularity' as feature_type,
            item_id as feature_key,
            JSON_OBJECT(
                'unique_users', unique_users,
                'ctr', ctr,
                'cvr', cvr,
                'total_interactions', total_interactions
            ) as feature_value,
            window_end as update_time
        FROM item_popularity_realtime
    """)
    
    # Create Redis sink for serving
    table_env.execute_sql("""
        CREATE TABLE redis_features (
            key_type STRING,
            key_id STRING,
            features MAP<STRING, STRING>,
            ttl INT
        ) WITH (
            'connector' = 'redis',
            'host' = 'localhost',
            'port' = '6379',
            'command' = 'HSET',
            'key-pattern' = '{key_type}:{key_id}',
            'value-pattern' = '{features}',
            'ttl' = '{ttl}'
        )
    """)
    
    # Write to Redis for online serving
    table_env.execute_sql("""
        INSERT INTO redis_features
        SELECT 
            'user' as key_type,
            user_id as key_id,
            MAP[
                'engagement_score', CAST(engagement_score AS STRING),
                'item_diversity', CAST(item_diversity AS STRING),
                'purchase_rate', CAST(purchase_rate AS STRING),
                'action_count', CAST(action_count AS STRING)
            ] as features,
            1800 as ttl  -- 30 minutes TTL
        FROM user_engagement_30min
    """)
    
    return table_env


# Custom aggregate function for percentile
@udaf(result_type='DOUBLE')
class PercentileAgg:
    """Calculate percentile in streaming"""
    
    def __init__(self):
        self.values = []
        self.percentile = 0.5
    
    def accumulate(self, value, percentile=0.5):
        if value is not None:
            self.values.append(float(value))
            self.percentile = percentile
    
    def get_value(self):
        if not self.values:
            return None
        sorted_values = sorted(self.values)
        index = int(len(sorted_values) * self.percentile)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def merge(self, other):
        self.values.extend(other.values)


def main():
    """Main entry point for Flink SQL job"""
    print("Starting Flink SQL Real-time Feature Engineering...")
    
    # Create and configure job
    table_env = create_flink_sql_job()
    
    # Register custom functions
    table_env.create_temporary_function("percentile_agg", PercentileAgg())
    
    # The job will run continuously
    # In production, this would be submitted to Flink cluster
    print("Flink SQL job configured and running...")
    print("Features are being computed and written to Kafka and Redis...")


if __name__ == "__main__":
    main()