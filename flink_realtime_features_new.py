"""
Flink Real-time Feature Update System - E-commerce Recommendation Real-time Processing Layer
Author: Yang Liu  
Description: Real-time feature computation and model updates based on Flink stream processing, 10-20 minute update frequency
Tech Stack: PyFlink + Kafka + Redis + Sliding Window + Real-time Aggregation
Experience: Real-time data processing architecture practices from Weibo 400M DAU, Qunar, SHAREit
"""

import json
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import queue
import hashlib

# Mock Flink and Kafka
class MockKafkaConsumer:
    """Mock Kafka consumer"""
    def __init__(self, topic: str, group_id: str):
        self.topic = topic
        self.group_id = group_id
        self.messages = []
        self._generate_mock_data()
    
    def _generate_mock_data(self):
        """Generate mock real-time data stream"""
        actions = ['view', 'click', 'cart', 'purchase', 'like', 'share']
        for i in range(1000):
            message = {
                'user_id': f'user_{i % 100:03d}',
                'item_id': f'item_{i % 500:03d}',
                'action': np.random.choice(actions),
                'category_id': np.random.randint(1, 21),
                'brand_id': np.random.randint(1, 101),
                'price': np.random.uniform(10, 1000),
                'timestamp': datetime.now().isoformat(),
                'session_id': f'session_{i % 200:03d}',
                'device_type': np.random.choice(['pc', 'mobile', 'app']),
                'page_id': f'page_{np.random.randint(1, 10)}'
            }
            self.messages.append(json.dumps(message))
    
    def poll(self, timeout_ms: int = 1000) -> List[str]:
        """Poll messages"""
        if self.messages:
            # Return 1-5 messages each time
            count = min(np.random.randint(1, 6), len(self.messages))
            batch = self.messages[:count]
            self.messages = self.messages[count:]
            return batch
        return []


class MockRedisClient:
    """Mock Redis client"""
    def __init__(self):
        self.store = {}
        self.hash_store = defaultdict(dict)
        self.sorted_sets = defaultdict(list)
    
    def set(self, key: str, value: str, ex: int = None):
        """Set key-value pair"""
        self.store[key] = value
    
    def get(self, key: str) -> Optional[str]:
        """Get value"""
        return self.store.get(key)
    
    def hset(self, name: str, key: str, value: str):
        """Hash table set"""
        self.hash_store[name][key] = value
    
    def hget(self, name: str, key: str) -> Optional[str]:
        """Hash table get"""
        return self.hash_store[name].get(key)
    
    def hgetall(self, name: str) -> Dict[str, str]:
        """Get all hash table fields"""
        return dict(self.hash_store[name])
    
    def zadd(self, name: str, mapping: Dict[str, float]):
        """Add to sorted set"""
        for member, score in mapping.items():
            self.sorted_sets[name].append((member, score))
            # Keep sorted
            self.sorted_sets[name].sort(key=lambda x: x[1], reverse=True)
    
    def zrange(self, name: str, start: int, end: int) -> List[str]:
        """Get sorted set range"""
        return [member for member, _ in self.sorted_sets[name][start:end+1]]


@dataclass
class RealTimeFeature:
    """Real-time feature"""
    feature_key: str
    feature_value: float
    update_time: datetime
    window_type: str = "sliding"  # sliding, tumbling
    window_size_minutes: int = 10


class SlidingWindowAggregator:
    """Sliding window aggregator"""
    
    def __init__(self, window_size_minutes: int = 10, slide_size_minutes: int = 2):
        """
        Initialize sliding window
        Args:
            window_size_minutes: Window size (minutes)
            slide_size_minutes: Slide step (minutes)
        """
        self.window_size = timedelta(minutes=window_size_minutes)
        self.slide_size = timedelta(minutes=slide_size_minutes)
        
        # Store window data {window_start: [events]}
        self.windows = defaultdict(list)
        
        # Aggregation result cache
        self.aggregated_features = {}
        
        self.last_cleanup = datetime.now()
        
    def add_event(self, event: Dict[str, Any]):
        """Add event to sliding window"""
        event_time = datetime.fromisoformat(event['timestamp'])
        
        # Calculate which windows the event belongs to
        current_time = datetime.now()
        
        # Calculate window start time backwards
        window_start = current_time - self.window_size
        
        # Only process events within time window
        if event_time >= window_start:
            # Calculate specific window slot
            slot_start = self._get_window_slot(event_time)
            self.windows[slot_start].append(event)
        
        # Periodically clean expired windows
        if (current_time - self.last_cleanup).seconds > 300:  # Clean every 5 minutes
            self._cleanup_expired_windows()
    
    def _get_window_slot(self, event_time: datetime) -> datetime:
        """Get window slot for event"""
        # Align window by slide step (simplified implementation)
        minutes_since_epoch = int(event_time.timestamp() / 60)
        slide_minutes = int(self.slide_size.total_seconds() / 60)
        slot_minutes = (minutes_since_epoch // slide_minutes) * slide_minutes
        return datetime.fromtimestamp(slot_minutes * 60)
    
    def _cleanup_expired_windows(self):
        """Clean expired windows"""
        current_time = datetime.now()
        expired_threshold = current_time - self.window_size - self.slide_size
        
        expired_windows = [
            window_start for window_start in self.windows.keys()
            if window_start < expired_threshold
        ]
        
        for window_start in expired_windows:
            del self.windows[window_start]
        
        self.last_cleanup = current_time
    
    def aggregate_features(self) -> Dict[str, Dict[str, float]]:
        """Aggregate features within window"""
        current_time = datetime.now()
        
        # User dimension aggregation
        user_features = defaultdict(lambda: {
            'click_count': 0, 'view_count': 0, 'cart_count': 0, 'purchase_count': 0,
            'view_duration': 0, 'session_count': 0, 'category_diversity': 0
        })
        
        # Item dimension aggregation  
        item_features = defaultdict(lambda: {
            'exposure_count': 0, 'click_count': 0, 'cart_count': 0, 'purchase_count': 0,
            'ctr': 0.0, 'cvr': 0.0, 'popularity_score': 0.0
        })
        
        # Iterate through all active windows
        active_threshold = current_time - self.window_size
        
        for window_start, events in self.windows.items():
            if window_start >= active_threshold:
                # User behavior aggregation
                user_sessions = defaultdict(set)
                user_categories = defaultdict(set)
                
                for event in events:
                    user_id = event['user_id']
                    item_id = event['item_id']
                    action = event['action']
                    category_id = event.get('category_id', 0)
                    session_id = event.get('session_id', '')
                    
                    # User feature aggregation
                    if action in user_features[user_id]:
                        user_features[user_id][action + '_count'] += 1
                    
                    user_sessions[user_id].add(session_id)
                    user_categories[user_id].add(category_id)
                    
                    # Item feature aggregation
                    item_features[item_id]['exposure_count'] += 1
                    if action == 'click':
                        item_features[item_id]['click_count'] += 1
                    elif action == 'cart':
                        item_features[item_id]['cart_count'] += 1
                    elif action == 'purchase':
                        item_features[item_id]['purchase_count'] += 1
                
                # Calculate derived features
                for user_id in user_features:
                    user_features[user_id]['session_count'] = len(user_sessions[user_id])
                    user_features[user_id]['category_diversity'] = len(user_categories[user_id])
        
        # Calculate conversion rate and other derived metrics
        for item_id in item_features:
            exposure = item_features[item_id]['exposure_count']
            clicks = item_features[item_id]['click_count']
            purchases = item_features[item_id]['purchase_count']
            
            item_features[item_id]['ctr'] = clicks / max(exposure, 1)
            item_features[item_id]['cvr'] = purchases / max(clicks, 1)
            item_features[item_id]['popularity_score'] = np.log1p(clicks + purchases)
        
        return {
            'user_features': dict(user_features),
            'item_features': dict(item_features)
        }


class FlinkRealTimeProcessor:
    """Flink real-time feature processor"""
    
    def __init__(self):
        """Initialize Flink real-time processor"""
        
        # Kafka consumer
        self.kafka_consumer = MockKafkaConsumer("user_behavior", "flink_processor")
        
        # Redis storage
        self.redis_client = MockRedisClient()
        
        # Sliding window aggregators
        self.window_aggregators = {
            'user_behavior_10m': SlidingWindowAggregator(window_size_minutes=10, slide_size_minutes=2),
            'user_behavior_20m': SlidingWindowAggregator(window_size_minutes=20, slide_size_minutes=5),
            'item_popularity_10m': SlidingWindowAggregator(window_size_minutes=10, slide_size_minutes=2)
        }
        
        # Real-time feature cache
        self.feature_cache = {}
        
        # Processor state
        self.is_running = False
        self.processed_count = 0
        self.last_update_time = datetime.now()
        
        # Update configuration
        self.update_config = {
            'batch_size': 100,                    # Batch processing size
            'update_interval_seconds': 30,        # Update interval (seconds)
            'feature_ttl_seconds': 1800,         # Feature TTL (seconds)
            'enable_model_update': True,          # Enable model updates
            'model_update_threshold': 1000        # Model update threshold
        }
        
        print("🚀 Flink real-time feature processor initialized")
        print(f"   Sliding windows: {list(self.window_aggregators.keys())}")
        print(f"   Update interval: {self.update_config['update_interval_seconds']} seconds")
    
    def start_processing(self):
        """Start real-time processing"""
        self.is_running = True
        print("🔄 Starting Flink real-time feature processing...")
        
        while self.is_running:
            try:
                # 1. Consume Kafka messages
                messages = self.kafka_consumer.poll(timeout_ms=1000)
                
                if messages:
                    batch_events = []
                    for msg in messages:
                        try:
                            event = json.loads(msg)
                            batch_events.append(event)
                        except json.JSONDecodeError:
                            continue
                    
                    # 2. Process event batch
                    if batch_events:
                        self._process_event_batch(batch_events)
                
                # 3. Periodically update features
                current_time = datetime.now()
                if (current_time - self.last_update_time).seconds >= self.update_config['update_interval_seconds']:
                    self._update_features()
                    self.last_update_time = current_time
                
                # 4. Check model updates
                if (self.processed_count > 0 and 
                    self.processed_count % self.update_config['model_update_threshold'] == 0 and
                    self.update_config['enable_model_update']):
                    self._trigger_model_update()
                
                time.sleep(0.1)  # Brief sleep to avoid high CPU usage
                
            except KeyboardInterrupt:
                print("\n⏹️  Stop signal received, shutting down processor...")
                break
            except Exception as e:
                print(f"❌ Processing error: {e}")
                time.sleep(1)
        
        self.is_running = False
        print("✅ Flink real-time processor stopped")
    
    def _process_event_batch(self, events: List[Dict[str, Any]]):
        """Process event batch"""
        
        for event in events:
            # Add to sliding window
            for aggregator in self.window_aggregators.values():
                aggregator.add_event(event)
            
            # Real-time feature computation
            self._compute_realtime_features(event)
        
        self.processed_count += len(events)
        
        if self.processed_count % 100 == 0:
            print(f"   📊 Processed {self.processed_count} events")
    
    def _compute_realtime_features(self, event: Dict[str, Any]):
        """Compute real-time features"""
        
        user_id = event['user_id']
        item_id = event['item_id']
        action = event['action']
        timestamp = datetime.fromisoformat(event['timestamp'])
        
        # User real-time features
        user_key = f"user_rt:{user_id}"
        
        # Get current user features
        current_user_features = self.redis_client.hgetall(user_key)
        
        # Update user behavior counts
        action_key = f"{action}_count_10m"
        current_count = int(current_user_features.get(action_key, 0))
        self.redis_client.hset(user_key, action_key, str(current_count + 1))
        
        # Update last active time
        self.redis_client.hset(user_key, "last_active_time", timestamp.isoformat())
        
        # Item real-time features
        item_key = f"item_rt:{item_id}"
        
        # Get current item features
        current_item_features = self.redis_client.hgetall(item_key)
        
        # Update item interaction counts
        exposure_count = int(current_item_features.get("exposure_count_10m", 0))
        self.redis_client.hset(item_key, "exposure_count_10m", str(exposure_count + 1))
        
        if action == 'click':
            click_count = int(current_item_features.get("click_count_10m", 0))
            self.redis_client.hset(item_key, "click_count_10m", str(click_count + 1))
        
        # Update item popularity score
        popularity = float(current_item_features.get("popularity_score", 0.0))
        action_weights = {'view': 1, 'click': 2, 'cart': 3, 'purchase': 5, 'like': 2, 'share': 3}
        popularity += action_weights.get(action, 1) * 0.1
        self.redis_client.hset(item_key, "popularity_score", str(popularity))
    
    def _update_features(self):
        """Update aggregated features"""
        print(f"🔄 Executing feature update ({datetime.now().strftime('%H:%M:%S')})")
        
        # Get aggregated features from sliding windows
        for window_name, aggregator in self.window_aggregators.items():
            aggregated = aggregator.aggregate_features()
            
            # Update user features
            for user_id, features in aggregated.get('user_features', {}).items():
                user_key = f"user_agg:{user_id}"
                for feature_name, value in features.items():
                    self.redis_client.hset(user_key, f"{feature_name}_{window_name}", str(value))
            
            # Update item features
            for item_id, features in aggregated.get('item_features', {}).items():
                item_key = f"item_agg:{item_id}"
                for feature_name, value in features.items():
                    self.redis_client.hset(item_key, f"{feature_name}_{window_name}", str(value))
        
        # Update trending items ranking
        self._update_trending_items()
        
        # Update active users list
        self._update_active_users()
        
        print(f"   ✅ Feature update completed")
    
    def _update_trending_items(self):
        """Update trending items ranking"""
        
        # Collect popularity scores for all items
        trending_items = {}
        
        for item_key in self.redis_client.hash_store.keys():
            if item_key.startswith("item_rt:"):
                item_id = item_key.split(":")[1]
                features = self.redis_client.hgetall(item_key)
                
                popularity = float(features.get("popularity_score", 0.0))
                click_count = int(features.get("click_count_10m", 0))
                
                # Comprehensive popularity score (popularity score + click count weight)
                trending_score = popularity + np.log1p(click_count) * 0.5
                trending_items[item_id] = trending_score
        
        # Update trending leaderboard
        if trending_items:
            self.redis_client.zadd("trending_items_10m", trending_items)
    
    def _update_active_users(self):
        """Update active users list"""
        
        current_time = datetime.now()
        active_threshold = current_time - timedelta(minutes=10)
        
        active_users = {}
        
        for user_key in self.redis_client.hash_store.keys():
            if user_key.startswith("user_rt:"):
                user_id = user_key.split(":")[1]
                features = self.redis_client.hgetall(user_key)
                
                last_active = features.get("last_active_time")
                if last_active:
                    try:
                        last_active_time = datetime.fromisoformat(last_active)
                        if last_active_time >= active_threshold:
                            # Activity score = total behavior count
                            activity_score = 0
                            for key, value in features.items():
                                if key.endswith("_count_10m"):
                                    activity_score += int(value)
                            
                            active_users[user_id] = activity_score
                    except:
                        continue
        
        # Update active users leaderboard
        if active_users:
            self.redis_client.zadd("active_users_10m", active_users)
    
    def _trigger_model_update(self):
        """Trigger model update"""
        print("🔄 Triggering DeepFM model incremental update...")
        
        # Get latest features for model update
        recent_features = self._collect_recent_features()
        
        # Simulate model update process (should call model training interface in production)
        if len(recent_features) > 0:
            print(f"   📊 Collected {len(recent_features)} samples for incremental training")
            print(f"   🤖 Executing DeepFM model incremental update...")
            
            # Simulate model update time
            time.sleep(1)
            
            print(f"   ✅ Model update completed")
        
    def _collect_recent_features(self) -> List[Dict[str, Any]]:
        """Collect recent feature data for model update"""
        
        features = []
        
        # Collect latest user and item features from Redis
        for key_pattern in ["user_rt:", "item_rt:"]:
            for key in self.redis_client.hash_store.keys():
                if key.startswith(key_pattern):
                    feature_data = self.redis_client.hgetall(key)
                    if feature_data:
                        feature_data['key'] = key
                        features.append(feature_data)
        
        return features
    
    def get_realtime_features(self, user_id: str, item_id: str) -> Dict[str, float]:
        """Get real-time features for user and item"""
        
        # User real-time features
        user_features = {}
        user_rt_key = f"user_rt:{user_id}"
        user_agg_key = f"user_agg:{user_id}"
        
        user_rt = self.redis_client.hgetall(user_rt_key)
        user_agg = self.redis_client.hgetall(user_agg_key)
        
        for key, value in user_rt.items():
            try:
                user_features[f"user_{key}"] = float(value)
            except:
                user_features[f"user_{key}"] = 0.0
        
        for key, value in user_agg.items():
            try:
                user_features[f"user_{key}"] = float(value)
            except:
                user_features[f"user_{key}"] = 0.0
        
        # Item real-time features
        item_features = {}
        item_rt_key = f"item_rt:{item_id}"
        item_agg_key = f"item_agg:{item_id}"
        
        item_rt = self.redis_client.hgetall(item_rt_key)
        item_agg = self.redis_client.hgetall(item_agg_key)
        
        for key, value in item_rt.items():
            try:
                item_features[f"item_{key}"] = float(value)
            except:
                item_features[f"item_{key}"] = 0.0
        
        for key, value in item_agg.items():
            try:
                item_features[f"item_{key}"] = float(value)
            except:
                item_features[f"item_{key}"] = 0.0
        
        # Merge features
        features = {**user_features, **item_features}
        
        return features
    
    def stop_processing(self):
        """Stop processing"""
        self.is_running = False


def demo_flink_realtime_processing():
    """Flink real-time processing demo"""
    print("🎯 Flink Real-time Feature Processing System Demo")
    print("=" * 50)
    
    # Initialize processor
    processor = FlinkRealTimeProcessor()
    
    print("\n🔄 Starting real-time processing (30 second demo)")
    
    # Run processor in separate thread
    processing_thread = threading.Thread(target=processor.start_processing)
    processing_thread.daemon = True
    processing_thread.start()
    
    # Run 30 second demo
    demo_duration = 30
    start_time = time.time()
    
    while time.time() - start_time < demo_duration:
        time.sleep(5)
        
        # Show processing progress
        elapsed = int(time.time() - start_time)
        print(f"⏱️  Runtime: {elapsed}s, Processed: {processor.processed_count} events")
        
        # Show real-time feature samples
        if elapsed % 10 == 0:  # Display every 10 seconds
            print("\n📊 Real-time Feature Samples:")
            
            # Get trending items
            trending_items = processor.redis_client.zrange("trending_items_10m", 0, 4)
            if trending_items:
                print(f"   🔥 Top 5 Trending Items: {trending_items}")
            
            # Get active users
            active_users = processor.redis_client.zrange("active_users_10m", 0, 4)  
            if active_users:
                print(f"   👥 Top 5 Active Users: {active_users}")
            
            # Get real-time features for specific user and item
            sample_features = processor.get_realtime_features("user_001", "item_001")
            if sample_features:
                print(f"   🎯 Real-time features for user_001 + item_001:")
                for key, value in list(sample_features.items())[:5]:
                    print(f"     {key}: {value:.4f}")
    
    # Stop processor
    processor.stop_processing()
    
    print(f"\n📈 Final Statistics:")
    print(f"   Total processed events: {processor.processed_count}")
    print(f"   Average processing speed: {processor.processed_count/demo_duration:.1f} events/sec")
    print(f"   Number of window aggregators: {len(processor.window_aggregators)}")
    print(f"   Redis key count: {len(processor.redis_client.store) + len(processor.redis_client.hash_store)}")
    
    print(f"\n✅ Flink real-time feature processing demo completed!")
    print("Key Features:")
    print("• Real-time sliding window aggregation (10min/20min windows)")
    print("• Kafka streaming data consumption")  
    print("• Redis real-time feature storage")
    print("• 10-20 minute level feature updates")
    print("• DeepFM model incremental update triggering")
    print("• Real-time ranking of trending items and active users")


if __name__ == "__main__":
    demo_flink_realtime_processing()