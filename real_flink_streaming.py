#!/usr/bin/env python3
"""
Real Apache Flink Streaming Implementation for E-commerce Recommendation
Using PyFlink for real-time feature computation
Author: Yang Liu
"""

from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors import FlinkKafkaConsumer, FlinkKafkaProducer
from pyflink.datastream.window import TumblingEventTimeWindows, SlidingEventTimeWindows
from pyflink.datastream.functions import MapFunction, KeyedProcessFunction, AggregateFunction
from pyflink.datastream.state import ValueStateDescriptor, ListStateDescriptor, MapStateDescriptor
from pyflink.common.serialization import SimpleStringSchema, JsonRowDeserializationSchema
from pyflink.common.typeinfo import Types
from pyflink.common.watermark_strategy import WatermarkStrategy
from pyflink.common import Duration, Time
from pyflink.table import StreamTableEnvironment
import json
from datetime import datetime
from typing import Iterable


class UserEventDeserializer(JsonRowDeserializationSchema):
    """Deserialize Kafka messages to user events"""
    
    def __init__(self):
        super().__init__(
            json_schema="""{
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "item_id": {"type": "string"},
                    "action": {"type": "string"},
                    "timestamp": {"type": "number"},
                    "session_id": {"type": "string"},
                    "category_id": {"type": "integer"},
                    "brand_id": {"type": "integer"},
                    "price": {"type": "number"}
                }
            }""",
            timestamp_format="SQL"
        )


class RealTimeFeatureExtractor(MapFunction):
    """Extract real-time features from user events"""
    
    def map(self, event):
        # Parse event
        user_id = event['user_id']
        item_id = event['item_id']
        action = event['action']
        timestamp = event['timestamp']
        
        # Extract features
        features = {
            'user_id': user_id,
            'item_id': item_id,
            'action': action,
            'timestamp': timestamp,
            'hour': datetime.fromtimestamp(timestamp).hour,
            'day_of_week': datetime.fromtimestamp(timestamp).weekday(),
            'is_weekend': datetime.fromtimestamp(timestamp).weekday() >= 5,
            'action_weight': self._get_action_weight(action)
        }
        
        return json.dumps(features)
    
    def _get_action_weight(self, action):
        weights = {
            'view': 1.0,
            'click': 2.0,
            'add_cart': 3.0,
            'favorite': 4.0,
            'purchase': 10.0
        }
        return weights.get(action, 1.0)


class UserBehaviorAggregator(AggregateFunction):
    """Aggregate user behavior in time windows"""
    
    def create_accumulator(self):
        return {
            'total_actions': 0,
            'unique_items': set(),
            'action_counts': {},
            'total_weight': 0.0,
            'last_timestamp': 0
        }
    
    def add(self, event, accumulator):
        accumulator['total_actions'] += 1
        accumulator['unique_items'].add(event['item_id'])
        
        action = event['action']
        accumulator['action_counts'][action] = accumulator['action_counts'].get(action, 0) + 1
        accumulator['total_weight'] += event['action_weight']
        accumulator['last_timestamp'] = max(accumulator['last_timestamp'], event['timestamp'])
        
        return accumulator
    
    def get_result(self, accumulator):
        return {
            'total_actions': accumulator['total_actions'],
            'unique_items_count': len(accumulator['unique_items']),
            'action_counts': accumulator['action_counts'],
            'engagement_score': accumulator['total_weight'],
            'last_timestamp': accumulator['last_timestamp']
        }
    
    def merge(self, acc1, acc2):
        acc1['total_actions'] += acc2['total_actions']
        acc1['unique_items'].update(acc2['unique_items'])
        
        for action, count in acc2['action_counts'].items():
            acc1['action_counts'][action] = acc1['action_counts'].get(action, 0) + count
        
        acc1['total_weight'] += acc2['total_weight']
        acc1['last_timestamp'] = max(acc1['last_timestamp'], acc2['last_timestamp'])
        
        return acc1


class ItemPopularityProcessor(KeyedProcessFunction):
    """Process item popularity with state management"""
    
    def __init__(self):
        self.state = None
    
    def open(self, runtime_context):
        # Initialize state
        state_descriptor = ValueStateDescriptor(
            "item_stats",
            Types.MAP(Types.STRING(), Types.FLOAT())
        )
        self.state = runtime_context.get_state(state_descriptor)
    
    def process_element(self, value, ctx):
        # Get current state
        current_stats = self.state.value() or {
            'views': 0,
            'clicks': 0,
            'purchases': 0,
            'last_update': 0
        }
        
        # Update stats
        action = value['action']
        if action == 'view':
            current_stats['views'] += 1
        elif action == 'click':
            current_stats['clicks'] += 1
        elif action == 'purchase':
            current_stats['purchases'] += 1
        
        current_stats['last_update'] = value['timestamp']
        
        # Calculate CTR and CVR
        if current_stats['views'] > 0:
            current_stats['ctr'] = current_stats['clicks'] / current_stats['views']
            current_stats['cvr'] = current_stats['purchases'] / current_stats['views']
        
        # Update state
        self.state.update(current_stats)
        
        # Emit result
        result = {
            'item_id': value['item_id'],
            'stats': current_stats,
            'timestamp': value['timestamp']
        }
        
        yield json.dumps(result)
    
    def on_timer(self, timestamp, ctx):
        # Periodic state cleanup or output
        pass


class CrossFeatureProcessor(KeyedProcessFunction):
    """Calculate user-item cross features with dual state"""
    
    def __init__(self):
        self.user_state = None
        self.item_state = None
    
    def open(self, runtime_context):
        # User state
        self.user_state = runtime_context.get_map_state(
            MapStateDescriptor(
                "user_features",
                Types.STRING(),
                Types.FLOAT()
            )
        )
        
        # Item state  
        self.item_state = runtime_context.get_map_state(
            MapStateDescriptor(
                "item_features",
                Types.STRING(),
                Types.FLOAT()
            )
        )
    
    def process_element(self, value, ctx):
        user_id = value['user_id']
        item_id = value['item_id']
        
        # Get user features from state
        user_activity = self.user_state.get(f"{user_id}_activity") or 0
        user_avg_price = self.user_state.get(f"{user_id}_avg_price") or 0
        
        # Get item features from state
        item_popularity = self.item_state.get(f"{item_id}_popularity") or 0
        item_price = value.get('price', 0)
        
        # Calculate cross features
        cross_features = {
            'user_item_key': f"{user_id}_{item_id}",
            'activity_popularity_match': user_activity * item_popularity,
            'price_sensitivity': abs(user_avg_price - item_price) / max(user_avg_price, 1),
            'personalized_score': self._calculate_personalized_score(
                user_activity, item_popularity, user_avg_price, item_price
            ),
            'timestamp': value['timestamp']
        }
        
        yield json.dumps(cross_features)
    
    def _calculate_personalized_score(self, user_activity, item_popularity, 
                                     user_avg_price, item_price):
        # Custom scoring logic
        activity_score = min(user_activity / 100, 1.0)
        popularity_score = min(item_popularity / 1000, 1.0)
        price_match = 1.0 - min(abs(user_avg_price - item_price) / max(user_avg_price, 1), 1.0)
        
        return activity_score * 0.3 + popularity_score * 0.3 + price_match * 0.4


def create_flink_job():
    """Create and configure Flink streaming job"""
    
    # Set up the execution environment
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(4)
    env.enable_checkpointing(60000)  # Checkpoint every 60 seconds
    
    # Kafka source configuration
    kafka_props = {
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'recommendation-feature-group',
        'auto.offset.reset': 'latest'
    }
    
    # Create Kafka consumer
    kafka_consumer = FlinkKafkaConsumer(
        topics='user-events',
        deserialization_schema=UserEventDeserializer(),
        properties=kafka_props
    )
    
    # Set watermark strategy for event time processing
    kafka_consumer.set_start_from_latest()
    watermark_strategy = WatermarkStrategy.for_bounded_out_of_orderness(Duration.of_seconds(5))
    
    # Create data stream
    event_stream = env.add_source(kafka_consumer).assign_timestamps_and_watermarks(watermark_strategy)
    
    # 1. Extract real-time features
    feature_stream = event_stream.map(
        RealTimeFeatureExtractor(),
        output_type=Types.STRING()
    )
    
    # 2. User behavior aggregation with sliding windows
    user_windows = event_stream \
        .key_by(lambda x: x['user_id']) \
        .window(SlidingEventTimeWindows.of(
            Time.minutes(30),  # Window size
            Time.minutes(5)    # Slide interval
        )) \
        .aggregate(
            UserBehaviorAggregator(),
            output_type=Types.MAP(Types.STRING(), Types.PICKLED_BYTE_ARRAY())
        )
    
    # 3. Item popularity processing with state
    item_stats = event_stream \
        .key_by(lambda x: x['item_id']) \
        .process(
            ItemPopularityProcessor(),
            output_type=Types.STRING()
        )
    
    # 4. Cross features calculation
    cross_features = event_stream \
        .key_by(lambda x: f"{x['user_id']}_{x['item_id']}") \
        .process(
            CrossFeatureProcessor(),
            output_type=Types.STRING()
        )
    
    # Kafka sink for feature output
    feature_producer = FlinkKafkaProducer(
        topic='realtime-features',
        serialization_schema=SimpleStringSchema(),
        producer_config={
            'bootstrap.servers': 'localhost:9092',
            'transaction.timeout.ms': '900000'
        },
        semantic=FlinkKafkaProducer.Semantic.EXACTLY_ONCE
    )
    
    # Output streams to Kafka
    feature_stream.add_sink(feature_producer)
    user_windows.map(lambda x: json.dumps(x)).add_sink(feature_producer)
    item_stats.add_sink(feature_producer)
    cross_features.add_sink(feature_producer)
    
    # Redis sink for serving
    feature_stream.add_sink(RedisFeatureSink())
    
    return env


class RedisFeatureSink:
    """Custom sink to write features to Redis"""
    
    def __init__(self, redis_host='localhost', redis_port=6379):
        import redis
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
    
    def invoke(self, value, context):
        """Write feature to Redis"""
        try:
            data = json.loads(value)
            
            # User features
            if 'user_id' in data:
                key = f"user_features:{data['user_id']}"
                self.redis_client.hset(key, mapping=data)
                self.redis_client.expire(key, 3600)  # 1 hour TTL
            
            # Item features
            if 'item_id' in data and 'stats' in data:
                key = f"item_features:{data['item_id']}"
                self.redis_client.hset(key, mapping=data['stats'])
                self.redis_client.expire(key, 3600)
            
            # Cross features
            if 'user_item_key' in data:
                key = f"cross_features:{data['user_item_key']}"
                self.redis_client.hset(key, mapping=data)
                self.redis_client.expire(key, 1800)  # 30 min TTL
                
        except Exception as e:
            print(f"Error writing to Redis: {e}")


def main():
    """Main entry point"""
    print("Starting Real-time Feature Computation with Apache Flink...")
    
    # Create and execute job
    env = create_flink_job()
    
    # Execute the job
    env.execute("E-commerce Recommendation Real-time Features")


if __name__ == "__main__":
    main()