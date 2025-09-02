"""
Real-time Feature Processing System
Author: Yang Liu
Description: Stream processing for real-time recommendation features using Kafka/Flink patterns
Experience: Built similar systems at Weibo (400M DAU), SHAREit
"""

import json
import time
import redis
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
from datetime import datetime, timedelta
from dataclasses import dataclass
import hashlib
import threading
import queue


@dataclass
class UserEvent:
    """User interaction event"""
    user_id: str
    item_id: str
    action: str
    timestamp: float
    session_id: str
    platform: str
    metadata: Dict[str, Any] = None


class FeatureStore:
    """
    In-memory feature store with Redis-like interface
    In production, this would connect to actual Redis/DynamoDB
    """
    
    def __init__(self, ttl_seconds: int = 3600):
        self.store = {}
        self.ttl = ttl_seconds
        self.timestamps = {}
        
    def set(self, key: str, value: Any, ttl: int = None):
        """Set key-value with optional TTL"""
        self.store[key] = value
        self.timestamps[key] = time.time()
        
    def get(self, key: str) -> Any:
        """Get value by key"""
        if key in self.store:
            # Check TTL
            if time.time() - self.timestamps[key] > self.ttl:
                del self.store[key]
                del self.timestamps[key]
                return None
            return self.store[key]
        return None
    
    def incr(self, key: str, amount: int = 1):
        """Increment counter"""
        current = self.get(key) or 0
        self.set(key, current + amount)
        return current + amount
    
    def zadd(self, key: str, member: str, score: float):
        """Add to sorted set"""
        if key not in self.store:
            self.store[key] = {}
        self.store[key][member] = score
        self.timestamps[key] = time.time()
    
    def zrange(self, key: str, start: int, end: int) -> List[Tuple[str, float]]:
        """Get range from sorted set"""
        if key not in self.store:
            return []
        
        sorted_items = sorted(self.store[key].items(), 
                            key=lambda x: x[1], reverse=True)
        return sorted_items[start:end+1]


class RealtimeFeatureProcessor:
    """
    Process streaming events and generate real-time features
    Simulates Flink/Spark Streaming processing
    """
    
    def __init__(self, feature_store: FeatureStore = None):
        self.feature_store = feature_store or FeatureStore()
        self.event_buffer = deque(maxlen=10000)
        self.processing_queue = queue.Queue()
        self.is_running = False
        
        # Feature windows
        self.window_sizes = {
            '1min': 60,
            '5min': 300,
            '30min': 1800,
            '1hour': 3600,
            '1day': 86400
        }
        
        # Action weights for scoring
        self.action_weights = {
            'view': 1.0,
            'click': 2.0,
            'like': 3.0,
            'share': 4.0,
            'purchase': 5.0
        }
        
    def process_event(self, event: UserEvent):
        """
        Process single event and update features
        """
        # Add to buffer
        self.event_buffer.append(event)
        
        # Update user features
        self._update_user_features(event)
        
        # Update item features  
        self._update_item_features(event)
        
        # Update user-item features
        self._update_user_item_features(event)
        
        # Update session features
        self._update_session_features(event)
        
        # Detect patterns
        self._detect_patterns(event)
        
    def _update_user_features(self, event: UserEvent):
        """Update real-time user features"""
        user_key = f"user:{event.user_id}"
        
        # Count actions by type
        self.feature_store.incr(f"{user_key}:action:{event.action}")
        self.feature_store.incr(f"{user_key}:total_actions")
        
        # Update time windows
        for window_name, window_size in self.window_sizes.items():
            window_key = f"{user_key}:window:{window_name}"
            self.feature_store.zadd(window_key, event.item_id, event.timestamp)
            
            # Clean old entries
            cutoff_time = time.time() - window_size
            current_items = self.feature_store.get(window_key) or {}
            cleaned = {k: v for k, v in current_items.items() if v > cutoff_time}
            self.feature_store.set(window_key, cleaned)
        
        # Update user embedding (simplified)
        embedding_key = f"{user_key}:embedding"
        current_embedding = self.feature_store.get(embedding_key) or np.zeros(64)
        
        # Simple update based on item (in production, use actual item embeddings)
        item_hash = int(hashlib.md5(event.item_id.encode()).hexdigest()[:8], 16)
        item_vector = np.random.RandomState(item_hash).randn(64)
        
        # Exponential moving average
        alpha = 0.1
        new_embedding = (1 - alpha) * current_embedding + alpha * item_vector
        self.feature_store.set(embedding_key, new_embedding.tolist())
        
    def _update_item_features(self, event: UserEvent):
        """Update real-time item features"""
        item_key = f"item:{event.item_id}"
        
        # Popularity scores
        score = self.action_weights.get(event.action, 1.0)
        self.feature_store.incr(f"{item_key}:popularity", amount=int(score))
        
        # Unique users
        users_key = f"{item_key}:users"
        users_set = self.feature_store.get(users_key) or set()
        users_set.add(event.user_id)
        self.feature_store.set(users_key, users_set)
        
        # Time-decayed popularity
        for window_name, window_size in self.window_sizes.items():
            window_key = f"{item_key}:popularity:{window_name}"
            current_time = time.time()
            
            # Get current window data
            window_data = self.feature_store.get(window_key) or []
            
            # Add new event
            window_data.append((current_time, score))
            
            # Remove old events
            cutoff = current_time - window_size
            window_data = [(t, s) for t, s in window_data if t > cutoff]
            
            # Calculate weighted sum with time decay
            if window_data:
                weighted_sum = sum(s * np.exp(-(current_time - t) / window_size) 
                                 for t, s in window_data)
                self.feature_store.set(f"{window_key}:score", weighted_sum)
            
            self.feature_store.set(window_key, window_data)
            
    def _update_user_item_features(self, event: UserEvent):
        """Update user-item interaction features"""
        key = f"user_item:{event.user_id}:{event.item_id}"
        
        # Interaction count
        self.feature_store.incr(f"{key}:count")
        
        # Last interaction time
        self.feature_store.set(f"{key}:last_time", event.timestamp)
        
        # Action history
        history_key = f"{key}:history"
        history = self.feature_store.get(history_key) or []
        history.append({
            'action': event.action,
            'timestamp': event.timestamp,
            'session': event.session_id
        })
        
        # Keep last 10 interactions
        if len(history) > 10:
            history = history[-10:]
        
        self.feature_store.set(history_key, history)
        
    def _update_session_features(self, event: UserEvent):
        """Update session-level features"""
        session_key = f"session:{event.session_id}"
        
        # Session events
        events_key = f"{session_key}:events"
        events = self.feature_store.get(events_key) or []
        events.append({
            'item_id': event.item_id,
            'action': event.action,
            'timestamp': event.timestamp
        })
        self.feature_store.set(events_key, events)
        
        # Session statistics
        self.feature_store.incr(f"{session_key}:event_count")
        self.feature_store.set(f"{session_key}:last_event", event.timestamp)
        
        # Calculate session duration
        if len(events) > 1:
            duration = events[-1]['timestamp'] - events[0]['timestamp']
            self.feature_store.set(f"{session_key}:duration", duration)
            
    def _detect_patterns(self, event: UserEvent):
        """Detect behavioral patterns in real-time"""
        user_key = f"user:{event.user_id}"
        
        # Check for burst activity
        recent_actions = self.feature_store.get(f"{user_key}:window:1min") or {}
        if len(recent_actions) > 20:
            self.feature_store.set(f"{user_key}:pattern:burst", True)
        
        # Check for purchase intent
        recent_events = [e for e in self.event_buffer 
                        if e.user_id == event.user_id][-10:]
        
        if len(recent_events) >= 3:
            action_sequence = [e.action for e in recent_events[-3:]]
            if action_sequence == ['view', 'click', 'like']:
                self.feature_store.set(f"{user_key}:pattern:high_intent", True)
                
    def get_user_features(self, user_id: str) -> Dict:
        """Get all features for a user"""
        user_key = f"user:{user_id}"
        features = {}
        
        # Action counts
        for action in self.action_weights.keys():
            count = self.feature_store.get(f"{user_key}:action:{action}") or 0
            features[f'action_count_{action}'] = count
        
        # Window features
        for window_name in self.window_sizes.keys():
            window_key = f"{user_key}:window:{window_name}"
            window_data = self.feature_store.get(window_key) or {}
            features[f'items_in_{window_name}'] = len(window_data)
        
        # Embedding
        embedding = self.feature_store.get(f"{user_key}:embedding")
        if embedding:
            features['embedding'] = embedding
        
        # Patterns
        features['is_burst'] = self.feature_store.get(f"{user_key}:pattern:burst") or False
        features['high_intent'] = self.feature_store.get(f"{user_key}:pattern:high_intent") or False
        
        return features
    
    def get_item_features(self, item_id: str) -> Dict:
        """Get all features for an item"""
        item_key = f"item:{item_id}"
        features = {}
        
        # Popularity
        features['popularity'] = self.feature_store.get(f"{item_key}:popularity") or 0
        
        # Unique users
        users = self.feature_store.get(f"{item_key}:users") or set()
        features['unique_users'] = len(users)
        
        # Window popularity scores
        for window_name in self.window_sizes.keys():
            score = self.feature_store.get(f"{item_key}:popularity:{window_name}:score") or 0
            features[f'popularity_{window_name}'] = score
        
        return features
    
    def get_recommendation_features(self, user_id: str, item_ids: List[str]) -> List[Dict]:
        """
        Get features for ranking candidate items for a user
        """
        user_features = self.get_user_features(user_id)
        results = []
        
        for item_id in item_ids:
            item_features = self.get_item_features(item_id)
            
            # User-item features
            ui_key = f"user_item:{user_id}:{item_id}"
            interaction_count = self.feature_store.get(f"{ui_key}:count") or 0
            last_interaction = self.feature_store.get(f"{ui_key}:last_time") or 0
            
            # Combine features
            combined = {
                'user_id': user_id,
                'item_id': item_id,
                'user_features': user_features,
                'item_features': item_features,
                'interaction_count': interaction_count,
                'last_interaction': last_interaction,
                'time_since_last': time.time() - last_interaction if last_interaction else float('inf')
            }
            
            # Calculate real-time score
            score = self._calculate_realtime_score(combined)
            combined['realtime_score'] = score
            
            results.append(combined)
        
        return sorted(results, key=lambda x: x['realtime_score'], reverse=True)
    
    def _calculate_realtime_score(self, features: Dict) -> float:
        """
        Calculate real-time ranking score
        """
        score = 0.0
        
        # Item popularity (normalized)
        item_pop = features['item_features'].get('popularity', 0)
        score += np.log1p(item_pop) * 0.3
        
        # User-item affinity
        interaction_count = features.get('interaction_count', 0)
        score += np.log1p(interaction_count) * 0.4
        
        # Time decay
        time_since = features.get('time_since_last', float('inf'))
        if time_since < float('inf'):
            decay = np.exp(-time_since / 86400)  # 1 day decay
            score += decay * 0.2
        
        # User intent signals
        if features['user_features'].get('high_intent', False):
            score *= 1.5
        
        return score


class StreamSimulator:
    """
    Simulate real-time event stream for testing
    """
    
    def __init__(self, processor: RealtimeFeatureProcessor):
        self.processor = processor
        self.is_running = False
        
    def generate_events(self, n_events: int = 1000, n_users: int = 100, n_items: int = 500):
        """Generate and process random events"""
        import random
        
        actions = ['view', 'click', 'like', 'share', 'purchase']
        platforms = ['PS5', 'PS4', 'Mobile', 'Web']
        
        print(f"Generating {n_events} events...")
        
        for i in range(n_events):
            event = UserEvent(
                user_id=f"user_{random.randint(1, n_users)}",
                item_id=f"game_{random.randint(1, n_items)}",
                action=random.choice(actions),
                timestamp=time.time(),
                session_id=f"session_{random.randint(1, n_events//10)}",
                platform=random.choice(platforms)
            )
            
            self.processor.process_event(event)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} events")
            
            # Simulate real-time delay
            time.sleep(0.001)
        
        print("Event generation completed")


def demo_realtime_processing():
    """
    Demonstrate real-time feature processing
    """
    print("Real-time Feature Processing Demo")
    print("="*50)
    
    # Initialize components
    feature_store = FeatureStore()
    processor = RealtimeFeatureProcessor(feature_store)
    simulator = StreamSimulator(processor)
    
    # Generate events
    simulator.generate_events(n_events=500, n_users=50, n_items=100)
    
    # Get features for a user
    test_user = "user_1"
    print(f"\nFeatures for {test_user}:")
    user_features = processor.get_user_features(test_user)
    for key, value in list(user_features.items())[:10]:
        if key != 'embedding':  # Skip embedding for display
            print(f"  {key}: {value}")
    
    # Get features for an item
    test_item = "game_1"
    print(f"\nFeatures for {test_item}:")
    item_features = processor.get_item_features(test_item)
    for key, value in item_features.items():
        print(f"  {key}: {value}")
    
    # Get recommendation features
    candidate_items = [f"game_{i}" for i in range(1, 11)]
    print(f"\nRecommendation scores for {test_user}:")
    recommendations = processor.get_recommendation_features(test_user, candidate_items)
    
    for rec in recommendations[:5]:
        print(f"  {rec['item_id']}: {rec['realtime_score']:.3f}")


if __name__ == "__main__":
    demo_realtime_processing()