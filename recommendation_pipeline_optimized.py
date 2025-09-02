"""
Optimized Recommendation System Pipeline - Complete Implementation
Recall 10000 -> Coarse Rank 1000 -> Fine Rank 100 -> Redis Cache
Author: Yang Liu
"""

import json
import time
import pickle
import redis
import asyncio
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

# Configuration parameters
RECALL_CANDIDATES = 10000  # Number of recall candidates
COARSE_RANK_OUTPUT = 1000  # Coarse ranking output size
FINE_RANK_OUTPUT = 100     # Fine ranking output size
FINAL_DISPLAY = 20          # Final display count

# Storage path configuration
RECALL_DATA_PATH = "/data/recall_candidates/"  # Recall data file storage path
REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db': 0,
    'decode_responses': False
}

@dataclass
class RecommendationConfig:
    """Recommendation system configuration"""
    recall_size: int = RECALL_CANDIDATES
    coarse_rank_size: int = COARSE_RANK_OUTPUT
    fine_rank_size: int = FINE_RANK_OUTPUT
    display_size: int = FINAL_DISPLAY
    cache_ttl: int = 3600  # Redis cache TTL in seconds
    enable_file_cache: bool = True  # Enable file cache for recall data


class RecallDataManager:
    """Recall data manager - Handles file storage and retrieval"""
    
    def __init__(self, base_path: str = RECALL_DATA_PATH):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def save_recall_candidates(self, user_id: str, candidates: List[Dict]) -> str:
        """
        Save recall candidates to file
        
        Args:
            user_id: User ID
            candidates: List of recalled candidate items
            
        Returns:
            File path
        """
        # Store partitioned by date
        date_str = datetime.now().strftime("%Y%m%d")
        date_path = self.base_path / date_str
        date_path.mkdir(exist_ok=True)
        
        # User recall data file path
        file_path = date_path / f"user_{user_id}_recall.json"
        
        # Save data
        recall_data = {
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'total_candidates': len(candidates),
            'candidates': candidates[:RECALL_CANDIDATES]  # Ensure not exceeding 10000
        }
        
        with open(file_path, 'w') as f:
            json.dump(recall_data, f)
            
        return str(file_path)
    
    def load_recall_candidates(self, user_id: str, date: Optional[str] = None) -> Optional[List[Dict]]:
        """
        Load recall candidates from file
        
        Args:
            user_id: User ID
            date: Date string (YYYYMMDD), defaults to today
            
        Returns:
            List of recall candidates, None if not exists
        """
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
            
        file_path = self.base_path / date / f"user_{user_id}_recall.json"
        
        if not file_path.exists():
            return None
            
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Check data freshness (optional)
        timestamp = datetime.fromisoformat(data['timestamp'])
        if datetime.now() - timestamp > timedelta(hours=24):
            print(f"Warning: Recall data for user {user_id} is older than 24 hours")
            
        return data['candidates']
    
    def batch_save_recall_candidates(self, user_candidates: Dict[str, List[Dict]]) -> Dict[str, str]:
        """
        Batch save recall candidates for multiple users
        
        Args:
            user_candidates: Dictionary of {user_id: candidates}
            
        Returns:
            Dictionary of {user_id: file_path}
        """
        file_paths = {}
        for user_id, candidates in user_candidates.items():
            file_path = self.save_recall_candidates(user_id, candidates)
            file_paths[user_id] = file_path
            
        return file_paths


class RedisManager:
    """Redis cache manager"""
    
    def __init__(self, config: Dict = REDIS_CONFIG):
        # Use real Redis in production
        # self.client = redis.Redis(**config)
        # Using dictionary for simulation here
        self.cache = {}
        self.ttl_tracker = {}
        
    def save_recommendations(self, user_id: str, recommendations: List[Dict], ttl: int = 3600):
        """
        Save recommendation results to Redis
        
        Args:
            user_id: User ID
            recommendations: List of recommendation results (100 after fine ranking)
            ttl: Time to live in seconds
        """
        key = f"rec:user:{user_id}"
        
        # Save as sorted set with recommendation score
        zadd_data = {}
        for idx, item in enumerate(recommendations[:FINE_RANK_OUTPUT]):
            item_key = f"{item['item_id']}:{item.get('score', 0)}"
            zadd_data[item_key] = float(FINE_RANK_OUTPUT - idx)  # Higher score for higher rank
            
        self.cache[key] = zadd_data
        self.ttl_tracker[key] = time.time() + ttl
        
        # Save metadata
        meta_key = f"rec:meta:user:{user_id}"
        meta_data = {
            'timestamp': datetime.now().isoformat(),
            'total_items': len(recommendations),
            'algorithm_version': 'v2.0',
            'experiment_id': 'default'
        }
        self.cache[meta_key] = json.dumps(meta_data)
        self.ttl_tracker[meta_key] = time.time() + ttl
        
    def get_recommendations(self, user_id: str, top_k: int = 20) -> Optional[List[Dict]]:
        """
        Get recommendation results from Redis
        
        Args:
            user_id: User ID
            top_k: Return top k results
            
        Returns:
            List of recommendations, None if not exists or expired
        """
        key = f"rec:user:{user_id}"
        
        # Check if expired
        if key not in self.cache:
            return None
            
        if time.time() > self.ttl_tracker.get(key, 0):
            del self.cache[key]
            return None
            
        # Get recommendation results
        items = self.cache[key]
        sorted_items = sorted(items.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Parse results
        recommendations = []
        for item_str, score in sorted_items:
            item_id, item_score = item_str.rsplit(':', 1)
            recommendations.append({
                'item_id': item_id,
                'score': float(item_score),
                'rank': len(recommendations) + 1
            })
            
        return recommendations
    
    def cache_exists(self, user_id: str) -> bool:
        """Check if cache exists and is valid"""
        key = f"rec:user:{user_id}"
        if key not in self.cache:
            return False
        return time.time() <= self.ttl_tracker.get(key, 0)


class RecommendationPipeline:
    """Complete recommendation pipeline"""
    
    def __init__(self, config: RecommendationConfig = RecommendationConfig()):
        self.config = config
        self.recall_manager = RecallDataManager()
        self.redis_manager = RedisManager()
        
        # Import existing modules
        from rga_recall_system import MultiChannelRecallSystem
        from deepfm_ranking_model import DeepFMRankingSystem
        from bert4rec_complete_ranking import BERT4RecRankingService
        
        self.recall_system = MultiChannelRecallSystem()
        self.coarse_ranker = DeepFMRankingSystem()
        self.fine_ranker = BERT4RecRankingService()
        
    async def recommend(self, user_id: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Complete recommendation flow
        
        Args:
            user_id: User ID
            use_cache: Whether to use cache
            
        Returns:
            Dictionary of recommendation results
        """
        start_time = time.time()
        metrics = {}
        
        # 1. Check Redis cache
        if use_cache and self.redis_manager.cache_exists(user_id):
            cached_results = self.redis_manager.get_recommendations(
                user_id, 
                self.config.display_size
            )
            if cached_results:
                return {
                    'user_id': user_id,
                    'recommendations': cached_results,
                    'from_cache': True,
                    'latency_ms': (time.time() - start_time) * 1000
                }
        
        # 2. Recall stage (10000 candidates)
        recall_start = time.time()
        
        # Try loading from file first
        candidates = None
        if self.config.enable_file_cache:
            candidates = self.recall_manager.load_recall_candidates(user_id)
            
        if candidates is None:
            # Real-time recall
            recall_results = self.recall_system.multi_channel_recall(
                user_id=user_id,
                target_count=self.config.recall_size
            )
            
            # Convert format
            candidates = [
                {
                    'item_id': item_id,
                    'recall_score': score,
                    'recall_source': source
                }
                for item_id, score, source in recall_results
            ]
            
            # Save to file
            if self.config.enable_file_cache:
                self.recall_manager.save_recall_candidates(user_id, candidates)
                
        metrics['recall_time_ms'] = (time.time() - recall_start) * 1000
        metrics['recall_count'] = len(candidates)
        
        # 3. Coarse ranking stage (10000 -> 1000)
        coarse_start = time.time()
        
        # Prepare coarse ranking input format
        recalled_items = [
            (c['item_id'], c['recall_score'], c['recall_source']) 
            for c in candidates
        ]
        
        coarse_ranked = self.coarse_ranker.coarse_ranking(
            recalled_items=recalled_items[:self.config.recall_size],
            user_id=user_id,
            user_features=self._get_user_features(user_id)
        )
        
        # Keep only top 1000
        coarse_ranked = coarse_ranked[:self.config.coarse_rank_size]
        
        metrics['coarse_rank_time_ms'] = (time.time() - coarse_start) * 1000
        metrics['coarse_rank_count'] = len(coarse_ranked)
        
        # 4. Fine ranking stage (1000 -> 100)
        fine_start = time.time()
        
        # Prepare fine ranking candidates
        candidate_ids = [int(item[0].split('_')[1]) for item in coarse_ranked]
        
        # BERT4Rec fine ranking
        fine_ranked = self.fine_ranker.rank_candidates(
            user_id=int(user_id.split('_')[1]) if isinstance(user_id, str) else user_id,
            candidates=candidate_ids[:self.config.coarse_rank_size],
            top_k=self.config.fine_rank_size
        )
        
        metrics['fine_rank_time_ms'] = (time.time() - fine_start) * 1000
        metrics['fine_rank_count'] = len(fine_ranked)
        
        # 5. Post-processing and business rule adjustments
        final_results = self._post_processing(fine_ranked, user_id)
        
        # 6. Cache to Redis
        cache_start = time.time()
        self.redis_manager.save_recommendations(
            user_id,
            final_results,
            ttl=self.config.cache_ttl
        )
        metrics['cache_time_ms'] = (time.time() - cache_start) * 1000
        
        # 7. Return final results
        total_time = time.time() - start_time
        
        return {
            'user_id': user_id,
            'recommendations': final_results[:self.config.display_size],
            'from_cache': False,
            'metrics': {
                **metrics,
                'total_time_ms': total_time * 1000,
                'pipeline_stages': {
                    'recall': f"{self.config.recall_size} candidates",
                    'coarse_rank': f"{self.config.coarse_rank_size} candidates", 
                    'fine_rank': f"{self.config.fine_rank_size} candidates",
                    'display': f"{self.config.display_size} items"
                }
            }
        }
    
    def _get_user_features(self, user_id: str) -> Dict[str, Any]:
        """Get user features"""
        # Should get real features from feature store here
        return {
            'user_level': 'vip',
            'age_group': '25-34',
            'gender': 'M',
            'city': 'Shanghai',
            'registration_days': 180,
            'last_30d_orders': 5,
            'last_30d_gmv': 2500.0
        }
    
    def _post_processing(self, ranked_items: List[Dict], user_id: str) -> List[Dict]:
        """
        Post-processing: Business rule adjustments
        - Deduplication
        - Diversify same category items
        - Insert operational items
        - Filter blacklist
        """
        processed = []
        seen_categories = set()
        seen_brands = set()
        
        for item in ranked_items:
            # Simple diversity control
            category = item.get('category', 'unknown')
            brand = item.get('brand', 'unknown')
            
            # Avoid consecutive same category
            if len(processed) > 0 and category == processed[-1].get('category'):
                continue
                
            # Limit same brand count
            if seen_brands.count(brand) >= 3:
                continue
                
            processed.append(item)
            seen_categories.add(category)
            seen_brands.add(brand)
            
            if len(processed) >= self.config.fine_rank_size:
                break
                
        return processed


async def main():
    """Test main function"""
    print("=" * 80)
    print("*** E-commerce Recommendation System Pipeline Test ***")
    print("=" * 80)
    
    # Create Pipeline
    pipeline = RecommendationPipeline()
    
    # Test users
    test_users = ['user_001', 'user_002', 'user_003']
    
    for user_id in test_users:
        print(f"\n*** Generating recommendations for user {user_id}...")
        
        # First request (no cache)
        result1 = await pipeline.recommend(user_id, use_cache=True)
        print(f"   * First request completed (from_cache={result1['from_cache']})")
        print(f"   * Recommended {len(result1['recommendations'])} items")
        print(f"   * Total latency: {result1.get('metrics', {}).get('total_time_ms', 0):.2f}ms")
        
        if not result1['from_cache']:
            print("\n   *** Pipeline stage latencies:")
            metrics = result1.get('metrics', {})
            print(f"      - Recall: {metrics.get('recall_time_ms', 0):.2f}ms ({metrics.get('recall_count', 0)} candidates)")
            print(f"      - Coarse rank: {metrics.get('coarse_rank_time_ms', 0):.2f}ms ({metrics.get('coarse_rank_count', 0)} candidates)")
            print(f"      - Fine rank: {metrics.get('fine_rank_time_ms', 0):.2f}ms ({metrics.get('fine_rank_count', 0)} candidates)")
            print(f"      - Cache: {metrics.get('cache_time_ms', 0):.2f}ms")
        
        # Second request (with cache)
        result2 = await pipeline.recommend(user_id, use_cache=True)
        print(f"\n   * Second request completed (from_cache={result2['from_cache']})")
        print(f"   * Cache hit latency: {result2.get('latency_ms', 0):.2f}ms")
        
        # Display top 5 recommendations
        print(f"\n   *** Top 5 recommended items:")
        for item in result1['recommendations'][:5]:
            print(f"      - {item['item_id']}: score={item.get('score', 0):.4f}, rank={item['rank']}")
    
    print("\n" + "=" * 80)
    print("*** Pipeline test completed! ***")
    print("=" * 80)


if __name__ == "__main__":
    # Run async main function
    asyncio.run(main())