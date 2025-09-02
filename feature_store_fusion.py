"""
Feature Store and Offline/Online Feature Fusion
Author: Yang Liu
Description: Production feature store with offline batch + real-time stream fusion
Experience: Built similar system at Weibo serving 400M+ users with <10ms latency
"""

import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import pickle
import hashlib
from abc import ABC, abstractmethod
import logging


@dataclass
class FeatureSpec:
    """Feature specification and metadata"""
    name: str
    feature_type: str  # 'numerical', 'categorical', 'embedding'
    data_source: str   # 'offline', 'realtime', 'hybrid'
    refresh_frequency: str  # '5min', '1hour', '1day'
    ttl_seconds: int
    default_value: Any = None
    description: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FeatureValue:
    """Feature value with metadata"""
    value: Any
    timestamp: float
    source: str  # 'offline_batch', 'realtime_stream', 'computed'
    freshness_score: float = 1.0  # 0-1, higher is fresher
    
    def is_expired(self, ttl_seconds: int) -> bool:
        return time.time() - self.timestamp > ttl_seconds
    
    def to_dict(self) -> Dict:
        return asdict(self)


class FeatureStorage(ABC):
    """Abstract interface for feature storage backends"""
    
    @abstractmethod
    def write_feature(self, key: str, value: FeatureValue):
        pass
    
    @abstractmethod
    def read_feature(self, key: str) -> Optional[FeatureValue]:
        pass
    
    @abstractmethod
    def read_features_batch(self, keys: List[str]) -> Dict[str, FeatureValue]:
        pass
    
    @abstractmethod
    def delete_feature(self, key: str):
        pass


class RedisFeatureStorage(FeatureStorage):
    """Redis-based feature storage for low-latency serving"""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        # In production: self.redis_client = redis.Redis(host=host, port=port, db=db)
        self.redis_client = None  # Mock for demo
        self.memory_cache = {}    # Fallback to memory
        
    def write_feature(self, key: str, value: FeatureValue):
        """Write feature to Redis with automatic TTL"""
        serialized_value = json.dumps(value.to_dict())
        
        if self.redis_client:
            # self.redis_client.setex(key, int(value.ttl_seconds), serialized_value)
            pass
        else:
            # Memory fallback
            self.memory_cache[key] = {
                'value': serialized_value,
                'expiry': time.time() + 3600  # 1 hour default
            }
    
    def read_feature(self, key: str) -> Optional[FeatureValue]:
        """Read single feature from Redis"""
        if self.redis_client:
            # value = self.redis_client.get(key)
            value = None
        else:
            # Memory fallback
            cached = self.memory_cache.get(key)
            if cached and cached['expiry'] > time.time():
                value = cached['value']
            else:
                value = None
        
        if value:
            try:
                data = json.loads(value)
                return FeatureValue(**data)
            except (json.JSONDecodeError, TypeError):
                return None
        return None
    
    def read_features_batch(self, keys: List[str]) -> Dict[str, FeatureValue]:
        """Batch read features for efficiency"""
        result = {}
        
        if self.redis_client:
            # values = self.redis_client.mget(keys)
            values = [None] * len(keys)
        else:
            # Memory fallback
            values = []
            for key in keys:
                cached = self.memory_cache.get(key)
                if cached and cached['expiry'] > time.time():
                    values.append(cached['value'])
                else:
                    values.append(None)
        
        for key, value in zip(keys, values):
            if value:
                try:
                    data = json.loads(value)
                    result[key] = FeatureValue(**data)
                except (json.JSONDecodeError, TypeError):
                    continue
        
        return result
    
    def delete_feature(self, key: str):
        """Delete feature from Redis"""
        if self.redis_client:
            # self.redis_client.delete(key)
            pass
        else:
            self.memory_cache.pop(key, None)


class CassandraFeatureStorage(FeatureStorage):
    """Cassandra-based storage for offline features and historical data"""
    
    def __init__(self, hosts: List[str] = ['127.0.0.1']):
        # In production: from cassandra.cluster import Cluster
        # self.cluster = Cluster(hosts)
        # self.session = self.cluster.connect()
        self.session = None
        self.memory_store = {}  # Fallback
        
    def write_feature(self, key: str, value: FeatureValue):
        """Write feature to Cassandra"""
        if self.session:
            # query = "INSERT INTO features (key, value, timestamp, source) VALUES (?, ?, ?, ?)"
            # self.session.execute(query, (key, json.dumps(value.to_dict()), value.timestamp, value.source))
            pass
        else:
            self.memory_store[key] = value
    
    def read_feature(self, key: str) -> Optional[FeatureValue]:
        """Read feature from Cassandra"""
        if self.session:
            # query = "SELECT value FROM features WHERE key = ? LIMIT 1"
            # rows = self.session.execute(query, (key,))
            # if rows:
            #     data = json.loads(rows[0].value)
            #     return FeatureValue(**data)
            return None
        else:
            return self.memory_store.get(key)
    
    def read_features_batch(self, keys: List[str]) -> Dict[str, FeatureValue]:
        """Batch read from Cassandra"""
        result = {}
        for key in keys:
            feature = self.read_feature(key)
            if feature:
                result[key] = feature
        return result
    
    def delete_feature(self, key: str):
        """Delete feature from Cassandra"""
        if self.session:
            # query = "DELETE FROM features WHERE key = ?"
            # self.session.execute(query, (key,))
            pass
        else:
            self.memory_store.pop(key, None)


class FeatureStore:
    """
    Unified feature store managing offline and real-time features
    """
    
    def __init__(self):
        # Storage backends
        self.redis_storage = RedisFeatureStorage()      # Hot features, low latency
        self.cassandra_storage = CassandraFeatureStorage()  # Cold features, historical
        
        # Feature registry
        self.feature_specs = {}
        self.feature_groups = defaultdict(list)
        
        # Caching and optimization
        self.local_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Monitoring
        self.metrics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'feature_requests': 0,
            'stale_features_served': 0
        }
        
    def register_feature(self, spec: FeatureSpec):
        """Register a feature with metadata"""
        self.feature_specs[spec.name] = spec
        
        # Group features by data source for optimization
        self.feature_groups[spec.data_source].append(spec.name)
        
        logging.info(f"Registered feature: {spec.name} ({spec.data_source})")
    
    def write_offline_features(self, entity_id: str, features: Dict[str, Any]):
        """Write batch-computed offline features"""
        timestamp = time.time()
        
        for feature_name, value in features.items():
            if feature_name not in self.feature_specs:
                logging.warning(f"Unknown feature: {feature_name}")
                continue
                
            spec = self.feature_specs[feature_name]
            feature_key = f"{entity_id}:{feature_name}"
            
            feature_value = FeatureValue(
                value=value,
                timestamp=timestamp,
                source='offline_batch',
                freshness_score=1.0
            )
            
            # Write to appropriate storage based on access pattern
            if spec.refresh_frequency in ['5min', '1hour']:
                self.redis_storage.write_feature(feature_key, feature_value)
            else:
                self.cassandra_storage.write_feature(feature_key, feature_value)
                
    def write_realtime_features(self, entity_id: str, features: Dict[str, Any]):
        """Write stream-computed real-time features"""
        timestamp = time.time()
        
        for feature_name, value in features.items():
            feature_key = f"{entity_id}:{feature_name}"
            
            feature_value = FeatureValue(
                value=value,
                timestamp=timestamp,
                source='realtime_stream',
                freshness_score=1.0
            )
            
            # Real-time features always go to Redis for low latency
            self.redis_storage.write_feature(feature_key, feature_value)
    
    def get_features(self, entity_id: str, feature_names: List[str], 
                    max_staleness_seconds: int = 3600) -> Dict[str, Any]:
        """
        Get features with intelligent offline/online fusion
        """
        self.metrics['feature_requests'] += 1
        result = {}
        
        # Check local cache first
        cache_key = f"{entity_id}:{':'.join(sorted(feature_names))}"
        if cache_key in self.local_cache:
            cached_data = self.local_cache[cache_key]
            if time.time() - cached_data['timestamp'] < self.cache_ttl:
                self.metrics['cache_hits'] += 1
                return cached_data['features']
        
        self.metrics['cache_misses'] += 1
        
        # Group features by storage backend for efficient retrieval
        redis_keys = []
        cassandra_keys = []
        
        for feature_name in feature_names:
            feature_key = f"{entity_id}:{feature_name}"
            
            if feature_name in self.feature_specs:
                spec = self.feature_specs[feature_name]
                if spec.data_source in ['realtime', 'hybrid']:
                    redis_keys.append(feature_key)
                else:
                    cassandra_keys.append(feature_key)
            else:
                # Default to Redis for unknown features
                redis_keys.append(feature_key)
        
        # Batch read from Redis
        redis_features = self.redis_storage.read_features_batch(redis_keys)
        
        # Batch read from Cassandra
        cassandra_features = self.cassandra_storage.read_features_batch(cassandra_keys)
        
        # Merge and apply fusion logic
        all_features = {**redis_features, **cassandra_features}
        
        for feature_name in feature_names:
            feature_key = f"{entity_id}:{feature_name}"
            
            if feature_key in all_features:
                feature_value = all_features[feature_key]
                
                # Check staleness
                if feature_value.is_expired(max_staleness_seconds):
                    self.metrics['stale_features_served'] += 1
                    
                    # Try to get default value
                    if feature_name in self.feature_specs:
                        result[feature_name] = self.feature_specs[feature_name].default_value
                    else:
                        result[feature_name] = None
                else:
                    result[feature_name] = feature_value.value
            else:
                # Feature not found, use default
                if feature_name in self.feature_specs:
                    result[feature_name] = self.feature_specs[feature_name].default_value
                else:
                    result[feature_name] = None
        
        # Cache the result
        self.local_cache[cache_key] = {
            'features': result,
            'timestamp': time.time()
        }
        
        return result
    
    def get_user_features(self, user_id: str, feature_set: str = 'serving') -> Dict[str, Any]:
        """Get predefined feature set for a user"""
        feature_sets = {
            'serving': [
                'total_interactions_7d', 'unique_items_7d', 'conversion_rate_7d',
                'avg_engagement_7d', 'platform_diversity_7d', 'total_lifetime_actions',
                'dominant_action', 'purchase_intent_high', 'multi_platform_user'
            ],
            'training': [
                'total_interactions_1d', 'total_interactions_7d', 'total_interactions_30d',
                'unique_items_1d', 'unique_items_7d', 'conversion_rate_7d',
                'engagement_score_300s', 'activity_intensity_7d'
            ]
        }
        
        feature_names = feature_sets.get(feature_set, feature_sets['serving'])
        return self.get_features(user_id, feature_names)
    
    def get_item_features(self, item_id: str, feature_set: str = 'serving') -> Dict[str, Any]:
        """Get predefined feature set for an item"""
        feature_sets = {
            'serving': [
                'popularity_score_7d', 'unique_users_7d', 'trending_score',
                'purchase_rate_7d', 'engagement_velocity_3600s', 'total_interactions'
            ],
            'training': [
                'interaction_count_1d', 'interaction_count_7d', 'unique_user_count_7d',
                'velocity_3600s', 'trending_score', 'purchase_velocity_3600s'
            ]
        }
        
        feature_names = feature_sets.get(feature_set, feature_sets['serving'])
        return self.get_features(item_id, feature_names)
    
    def compute_hybrid_features(self, entity_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute features that require both offline and real-time data
        """
        # Get offline features (stable, historical)
        offline_features = self.get_features(entity_id, [
            'total_interactions_30d', 'avg_engagement_30d', 'conversion_rate_30d'
        ])
        
        # Get real-time features (dynamic, recent)
        realtime_features = self.get_features(entity_id, [
            'engagement_score_300s', 'interaction_count_300s', 'activity_burst'
        ])
        
        # Compute hybrid features
        hybrid = {}
        
        # Activity trend: recent vs historical
        recent_activity = realtime_features.get('interaction_count_300s', 0)
        historical_avg = offline_features.get('total_interactions_30d', 0) / (30 * 24 * 12)  # per 5-min
        
        if historical_avg > 0:
            hybrid['activity_trend_ratio'] = recent_activity / historical_avg
        else:
            hybrid['activity_trend_ratio'] = 1.0 if recent_activity > 0 else 0.0
        
        # Engagement acceleration
        recent_engagement = realtime_features.get('engagement_score_300s', 0)
        historical_engagement = offline_features.get('avg_engagement_30d', 0)
        
        if historical_engagement > 0:
            hybrid['engagement_acceleration'] = recent_engagement / historical_engagement
        else:
            hybrid['engagement_acceleration'] = 1.0
        
        # Context-aware features
        if 'time_of_day' in context:
            hour = context['time_of_day']
            # Simple time-based adjustment
            if 18 <= hour <= 22:  # Peak gaming hours
                hybrid['time_context_multiplier'] = 1.2
            elif 2 <= hour <= 6:   # Low activity hours
                hybrid['time_context_multiplier'] = 0.8
            else:
                hybrid['time_context_multiplier'] = 1.0
        
        return hybrid
    
    def get_feature_freshness_report(self) -> Dict[str, Any]:
        """Generate report on feature freshness and quality"""
        current_time = time.time()
        report = {
            'total_features': len(self.feature_specs),
            'by_source': defaultdict(int),
            'by_freshness': defaultdict(int),
            'stale_features': [],
            'metrics': self.metrics.copy()
        }
        
        # Sample some features to check freshness
        sample_entities = ['user_1', 'user_100', 'game_1', 'game_50']
        
        for entity_id in sample_entities:
            for feature_name, spec in list(self.feature_specs.items())[:10]:  # Sample first 10
                feature_key = f"{entity_id}:{feature_name}"
                
                # Try Redis first, then Cassandra
                feature = self.redis_storage.read_feature(feature_key)
                if not feature:
                    feature = self.cassandra_storage.read_feature(feature_key)
                
                if feature:
                    age_seconds = current_time - feature.timestamp
                    report['by_source'][feature.source] += 1
                    
                    if age_seconds < 300:  # 5 minutes
                        report['by_freshness']['very_fresh'] += 1
                    elif age_seconds < 3600:  # 1 hour
                        report['by_freshness']['fresh'] += 1
                    elif age_seconds < 86400:  # 1 day
                        report['by_freshness']['stale'] += 1
                    else:
                        report['by_freshness']['very_stale'] += 1
                        report['stale_features'].append(feature_key)
        
        return report


class OnlineOfflineFusion:
    """
    Advanced feature fusion for serving predictions
    """
    
    def __init__(self, feature_store: FeatureStore):
        self.feature_store = feature_store
        
        # Fusion strategies
        self.fusion_strategies = {
            'latest': self._fusion_latest,
            'weighted_average': self._fusion_weighted_average,
            'confidence_weighted': self._fusion_confidence_weighted
        }
    
    def get_recommendation_features(self, user_id: str, item_ids: List[str], 
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get comprehensive features for recommendation with intelligent fusion
        """
        # Get user features
        user_features = self.feature_store.get_user_features(user_id, 'serving')
        
        # Get item features for all candidates
        item_features = {}
        for item_id in item_ids:
            item_features[item_id] = self.feature_store.get_item_features(item_id, 'serving')
        
        # Compute hybrid features
        hybrid_features = self.feature_store.compute_hybrid_features(user_id, context)
        
        # Compute cross features for each user-item pair
        cross_features = {}
        for item_id in item_ids:
            cross_key = f"{user_id}_{item_id}"
            cross_features[cross_key] = self._compute_cross_features(
                user_features, item_features[item_id], context
            )
        
        return {
            'user_features': user_features,
            'item_features': item_features,
            'hybrid_features': hybrid_features,
            'cross_features': cross_features,
            'context': context,
            'fusion_metadata': {
                'timestamp': time.time(),
                'feature_sources': ['offline_batch', 'realtime_stream', 'computed']
            }
        }
    
    def _compute_cross_features(self, user_features: Dict[str, Any], 
                               item_features: Dict[str, Any], 
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Compute cross features between user, item, and context"""
        cross = {}
        
        # User activity vs item popularity
        user_activity = user_features.get('total_interactions_7d', 0)
        item_popularity = item_features.get('popularity_score_7d', 0)
        
        if user_activity > 0 and item_popularity > 0:
            # Activity-popularity alignment score
            activity_norm = min(user_activity / 100, 1.0)
            popularity_norm = min(item_popularity / 1000, 1.0)
            cross['activity_popularity_alignment'] = activity_norm * popularity_norm
        else:
            cross['activity_popularity_alignment'] = 0.0
        
        # Conversion propensity
        user_conversion = user_features.get('conversion_rate_7d', 0)
        item_conversion = item_features.get('purchase_rate_7d', 0)
        cross['conversion_propensity'] = (user_conversion + item_conversion) / 2
        
        # Trending interaction score
        user_engagement = user_features.get('avg_engagement_7d', 0)
        item_trending = item_features.get('trending_score', 0)
        cross['trending_engagement_score'] = user_engagement * item_trending / 100
        
        # Context adjustments
        if 'platform' in context:
            platform = context['platform']
            if platform in user_features.get('preferred_platforms', []):
                cross['platform_preference_match'] = 1.0
            else:
                cross['platform_preference_match'] = 0.5
        
        return cross
    
    def _fusion_latest(self, offline_value: Any, realtime_value: Any, 
                      offline_timestamp: float, realtime_timestamp: float) -> Any:
        """Use the most recent value"""
        return realtime_value if realtime_timestamp > offline_timestamp else offline_value
    
    def _fusion_weighted_average(self, offline_value: float, realtime_value: float, 
                               offline_timestamp: float, realtime_timestamp: float) -> float:
        """Weighted average based on recency"""
        current_time = time.time()
        
        offline_age = current_time - offline_timestamp
        realtime_age = current_time - realtime_timestamp
        
        # Exponential decay weights
        offline_weight = np.exp(-offline_age / 3600)    # 1 hour half-life
        realtime_weight = np.exp(-realtime_age / 300)   # 5 min half-life
        
        total_weight = offline_weight + realtime_weight
        if total_weight > 0:
            return (offline_value * offline_weight + realtime_value * realtime_weight) / total_weight
        else:
            return offline_value  # Fallback
    
    def _fusion_confidence_weighted(self, offline_value: float, realtime_value: float, 
                                  offline_conf: float, realtime_conf: float) -> float:
        """Weighted fusion based on confidence scores"""
        total_conf = offline_conf + realtime_conf
        if total_conf > 0:
            return (offline_value * offline_conf + realtime_value * realtime_conf) / total_conf
        else:
            return offline_value


def demo_feature_store_fusion():
    """
    Demonstrate feature store and offline/online fusion
    """
    print("🏪 Feature Store and Offline/Online Fusion Demo")
    print("="*60)
    
    # Initialize feature store
    store = FeatureStore()
    fusion = OnlineOfflineFusion(store)
    
    # Register feature schemas
    print("Registering feature schemas...")
    
    user_features = [
        FeatureSpec('total_interactions_7d', 'numerical', 'offline', '1day', 86400, 0),
        FeatureSpec('conversion_rate_7d', 'numerical', 'offline', '1day', 86400, 0.0),
        FeatureSpec('engagement_score_300s', 'numerical', 'realtime', '5min', 300, 0.0),
        FeatureSpec('purchase_intent_high', 'categorical', 'realtime', '30min', 1800, False),
        FeatureSpec('total_lifetime_actions', 'numerical', 'hybrid', '1hour', 3600, 0)
    ]
    
    item_features = [
        FeatureSpec('popularity_score_7d', 'numerical', 'offline', '1day', 86400, 0.0),
        FeatureSpec('trending_score', 'numerical', 'realtime', '5min', 300, 0.0),
        FeatureSpec('purchase_rate_7d', 'numerical', 'offline', '1day', 86400, 0.0)
    ]
    
    for spec in user_features + item_features:
        store.register_feature(spec)
    
    # Simulate offline batch feature writes
    print("\nWriting offline batch features...")
    store.write_offline_features('user_1', {
        'total_interactions_7d': 150,
        'conversion_rate_7d': 0.05,
        'popularity_score_7d': 850.0,
        'total_lifetime_actions': 2500
    })
    
    store.write_offline_features('game_1', {
        'popularity_score_7d': 1250.0,
        'purchase_rate_7d': 0.08
    })
    
    # Simulate real-time stream feature writes
    print("Writing real-time stream features...")
    store.write_realtime_features('user_1', {
        'engagement_score_300s': 25.5,
        'purchase_intent_high': True,
        'total_lifetime_actions': 2502  # Updated in real-time
    })
    
    store.write_realtime_features('game_1', {
        'trending_score': 95.2
    })
    
    # Test feature retrieval
    print("\n📊 Testing Feature Retrieval:")
    
    # Get user features
    user_features = store.get_user_features('user_1', 'serving')
    print(f"User features: {user_features}")
    
    # Get item features
    item_features = store.get_item_features('game_1', 'serving')
    print(f"Item features: {item_features}")
    
    # Test recommendation feature fusion
    print("\n🔄 Testing Recommendation Feature Fusion:")
    context = {
        'time_of_day': 20,  # 8 PM
        'platform': 'PS5',
        'session_id': 'session_123'
    }
    
    recommendation_features = fusion.get_recommendation_features(
        'user_1', ['game_1', 'game_2'], context
    )
    
    print(f"Recommendation features keys: {list(recommendation_features.keys())}")
    print(f"Cross features: {recommendation_features['cross_features']}")
    print(f"Hybrid features: {recommendation_features['hybrid_features']}")
    
    # Generate freshness report
    print("\n📈 Feature Freshness Report:")
    report = store.get_feature_freshness_report()
    print(f"Total features: {report['total_features']}")
    print(f"By source: {dict(report['by_source'])}")
    print(f"By freshness: {dict(report['by_freshness'])}")
    print(f"Metrics: {report['metrics']}")
    
    print("\n✅ Feature store fusion demo completed!")


if __name__ == "__main__":
    demo_feature_store_fusion()