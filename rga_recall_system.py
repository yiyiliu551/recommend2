"""
RGA Vector Recall System - E-commerce Recommendation Recall Layer
Author: Yang Liu
Description: Product recall system based on RGA (Real-time Graph Attention) vector database, recalls 10000 candidate products per user
Tech Stack: Faiss vector search + Real-time graph attention network + Multi-channel recall strategy
Experience: Large-scale recall system practices from Weibo 400M DAU, Qunar, SHAREit
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict
import json
import pickle
import hashlib
from datetime import datetime, timedelta
import random

# Mock faiss (use real faiss in production environment)
class MockFaiss:
    """Mock Faiss vector search library"""
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors = []
        self.ids = []
        
    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray):
        self.vectors.extend(vectors.tolist())
        self.ids.extend(ids.tolist())
    
    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.vectors) == 0:
            return np.array([]), np.array([])
        
        # Simple cosine similarity search
        vectors = np.array(self.vectors)
        similarities = np.dot(vectors, query.T).flatten()
        similarities = similarities / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query))
        
        # Get top-k
        top_k_idx = np.argsort(similarities)[::-1][:k]
        scores = similarities[top_k_idx]
        ids = np.array([self.ids[i] for i in top_k_idx])
        
        return scores.reshape(1, -1), ids.reshape(1, -1)


@dataclass
class RGAEmbedding:
    """RGA vector embedding"""
    item_id: str
    embedding: np.ndarray  # 128-dimensional vector
    category_id: int
    update_time: datetime
    
    # RGA-specific attributes
    attention_weights: Dict[str, float] = None  # Attention weights
    graph_neighbors: List[str] = None           # Graph neighbor nodes
    popularity_score: float = 0.0               # Popularity score


class RGAVectorStore:
    """RGA vector storage and retrieval system"""
    
    def __init__(self, embedding_dim: int = 128):
        """Initialize RGA vector storage"""
        self.embedding_dim = embedding_dim
        
        # Faiss index - use real faiss in production environment
        self.item_index = MockFaiss(embedding_dim)
        self.user_index = MockFaiss(embedding_dim)
        
        # Vector storage
        self.item_embeddings = {}  # item_id -> RGAEmbedding
        self.user_embeddings = {}  # user_id -> np.ndarray
        
        # Graph structure storage
        self.item_graph = defaultdict(set)  # item -> related items
        self.category_graph = defaultdict(set)  # category -> items
        
        # Real-time update cache
        self.update_queue = []
        self.last_update_time = datetime.now()
        
        print("*** RGA vector storage system initialized ***")
    
    def add_item_embedding(self, item_id: str, embedding: np.ndarray,
                          category_id: int, graph_neighbors: List[str] = None):
        """Add item vector embedding"""
        
        rga_embedding = RGAEmbedding(
            item_id=item_id,
            embedding=embedding,
            category_id=category_id,
            update_time=datetime.now(),
            graph_neighbors=graph_neighbors or [],
            attention_weights={}
        )
        
        self.item_embeddings[item_id] = rga_embedding
        
        # Update Faiss index
        self.item_index.add_with_ids(
            embedding.reshape(1, -1), 
            np.array([hash(item_id) % 1000000])
        )
        
        # Update graph structure
        self.category_graph[category_id].add(item_id)
        if graph_neighbors:
            for neighbor in graph_neighbors:
                self.item_graph[item_id].add(neighbor)
                self.item_graph[neighbor].add(item_id)
    
    def add_user_embedding(self, user_id: str, embedding: np.ndarray):
        """Add user vector embedding"""
        self.user_embeddings[user_id] = embedding
        
        # Update Faiss index
        self.user_index.add_with_ids(
            embedding.reshape(1, -1),
            np.array([hash(user_id) % 1000000])
        )
    
    def update_attention_weights(self, item_id: str, interactions: List[Dict]):
        """Update RGA attention weights - Real-time graph attention mechanism"""
        if item_id not in self.item_embeddings:
            return
        
        attention_weights = {}
        
        # Calculate attention weights based on interactions
        for interaction in interactions:
            related_item = interaction.get('related_item_id')
            action_type = interaction.get('action', 'click')
            timestamp = datetime.fromisoformat(interaction.get('timestamp', '2024-01-01'))
            
            # Time decay
            days_ago = (datetime.now() - timestamp).days
            time_decay = np.exp(-days_ago * 0.1)
            
            # Action weights
            action_weights = {'view': 1.0, 'click': 2.0, 'cart': 3.0, 'purchase': 5.0}
            action_weight = action_weights.get(action_type, 1.0)
            
            # Calculate attention weight
            attention_score = action_weight * time_decay
            attention_weights[related_item] = attention_weights.get(related_item, 0) + attention_score
        
        # Normalize attention weights
        total_weight = sum(attention_weights.values())
        if total_weight > 0:
            attention_weights = {k: v/total_weight for k, v in attention_weights.items()}
        
        self.item_embeddings[item_id].attention_weights = attention_weights
    
    def vector_similarity_recall(self, user_id: str, top_k: int = 10000) -> List[Tuple[str, float]]:
        """Vector similarity recall"""
        if user_id not in self.user_embeddings:
            return []
        
        user_embedding = self.user_embeddings[user_id]
        
        # Faiss vector search
        scores, item_indices = self.item_index.search(
            user_embedding.reshape(1, -1), 
            min(top_k, len(self.item_embeddings))
        )
        
        results = []
        if len(scores[0]) > 0:
            for score, idx in zip(scores[0], item_indices[0]):
                # Reverse lookup item_id through hash (simplified implementation)
                for item_id in self.item_embeddings.keys():
                    if hash(item_id) % 1000000 == idx:
                        results.append((item_id, float(score)))
                        break
        
        return results


class MultiChannelRecallSystem:
    """Multi-channel recall system - Integrating multiple recall strategies"""
    
    def __init__(self):
        """Initialize multi-channel recall system"""
        self.rga_store = RGAVectorStore(embedding_dim=128)
        
        # Recall strategy weights
        self.recall_weights = {
            'vector_similarity': 0.4,    # Vector similarity
            'graph_attention': 0.25,     # Graph attention
            'category_cf': 0.15,         # Category collaborative filtering  
            'hot_items': 0.1,            # Hot items
            'new_items': 0.05,           # New items
            'seasonal_items': 0.05       # Seasonal items
        }
        
        # Recall diversity controls
        self.diversity_controls = {
            'max_same_category': 2000,   # Max recall count per category
            'max_same_brand': 1000,      # Max recall count per brand
            'min_categories': 5,         # Minimum number of categories
            'popularity_decay': 0.8      # Popularity decay
        }
        
        print("*** Multi-channel recall system initialized ***")
    
    def recall_by_vector_similarity(self, user_id: str, user_embedding: np.ndarray,
                                   top_k: int = 4000) -> List[Tuple[str, float]]:
        """Vector similarity recall"""
        return self.rga_store.vector_similarity_recall(user_id, top_k)
    
    def recall_by_graph_attention(self, user_id: str, user_history: List[str],
                                 top_k: int = 2500) -> List[Tuple[str, float]]:
        """RGA graph attention based recall"""
        candidates = []
        
        # Iterate through user's historical interactions
        for hist_item_id in user_history:
            if hist_item_id not in self.rga_store.item_embeddings:
                continue
                
            rga_embedding = self.rga_store.item_embeddings[hist_item_id]
            attention_weights = rga_embedding.attention_weights or {}
            
            # Recommend related items based on attention weights
            for related_item_id, attention_score in attention_weights.items():
                if related_item_id in self.rga_store.item_embeddings:
                    # Combine graph neighbor information
                    neighbor_boost = 1.2 if related_item_id in rga_embedding.graph_neighbors else 1.0
                    final_score = attention_score * neighbor_boost
                    candidates.append((related_item_id, final_score))
            
            # Graph neighbor recall
            for neighbor_id in rga_embedding.graph_neighbors:
                if neighbor_id in self.rga_store.item_embeddings:
                    # Score based on neighbor's popularity
                    neighbor_embedding = self.rga_store.item_embeddings[neighbor_id]
                    score = neighbor_embedding.popularity_score * 0.5
                    candidates.append((neighbor_id, score))
        
        # Deduplicate and sort
        candidate_scores = defaultdict(float)
        for item_id, score in candidates:
            candidate_scores[item_id] += score
        
        sorted_candidates = sorted(candidate_scores.items(), 
                                 key=lambda x: x[1], reverse=True)
        
        return sorted_candidates[:top_k]
    
    def recall_by_category_cf(self, user_id: str, user_categories: List[int],
                             top_k: int = 1500) -> List[Tuple[str, float]]:
        """Category collaborative filtering recall"""
        candidates = []
        
        for category_id in user_categories:
            category_items = self.rga_store.category_graph.get(category_id, set())
            
            # Sort items in category by popularity
            category_candidates = []
            for item_id in category_items:
                if item_id in self.rga_store.item_embeddings:
                    popularity = self.rga_store.item_embeddings[item_id].popularity_score
                    category_candidates.append((item_id, popularity))
            
            # Take top items from each category
            category_candidates.sort(key=lambda x: x[1], reverse=True)
            candidates.extend(category_candidates[:top_k//len(user_categories)])
        
        return candidates[:top_k]
    
    def recall_hot_items(self, top_k: int = 1000) -> List[Tuple[str, float]]:
        """Hot items recall"""
        hot_items = []
        
        for item_id, embedding in self.rga_store.item_embeddings.items():
            hot_items.append((item_id, embedding.popularity_score))
        
        hot_items.sort(key=lambda x: x[1], reverse=True)
        return hot_items[:top_k]
    
    def recall_new_items(self, days: int = 7, top_k: int = 500) -> List[Tuple[str, float]]:
        """New items recall"""
        cutoff_time = datetime.now() - timedelta(days=days)
        new_items = []
        
        for item_id, embedding in self.rga_store.item_embeddings.items():
            if embedding.update_time >= cutoff_time:
                # New item score = popularity * freshness
                freshness = (embedding.update_time - cutoff_time).total_seconds() / (86400 * days)
                score = embedding.popularity_score * (1 + freshness)
                new_items.append((item_id, score))
        
        new_items.sort(key=lambda x: x[1], reverse=True)
        return new_items[:top_k]
    
    def recall_seasonal_items(self, season: int, top_k: int = 500) -> List[Tuple[str, float]]:
        """Seasonal items recall (simplified implementation)"""
        # Mock seasonal items logic
        seasonal_items = []
        
        for item_id, embedding in self.rga_store.item_embeddings.items():
            # Simplified: determine seasonality based on category_id
            seasonal_boost = 1.0
            if season == 1 and embedding.category_id in [2, 5]:  # Summer clothing, beauty
                seasonal_boost = 1.5
            elif season == 3 and embedding.category_id in [2, 4]:  # Winter clothing, home
                seasonal_boost = 1.3
            
            if seasonal_boost > 1.0:
                score = embedding.popularity_score * seasonal_boost
                seasonal_items.append((item_id, score))
        
        seasonal_items.sort(key=lambda x: x[1], reverse=True)
        return seasonal_items[:top_k]
    
    def multi_channel_recall(self, user_id: str, user_profile: Dict[str, Any],
                           target_count: int = 10000) -> List[Tuple[str, float, str]]:
        """Multi-channel recall main function - Recall 10000 candidate items"""
        
        print(f"*** Starting multi-channel recall for user {user_id}...")
        
        all_candidates = []
        
        # 1. Vector similarity recall
        if user_id in self.rga_store.user_embeddings:
            vector_candidates = self.recall_by_vector_similarity(
                user_id, self.rga_store.user_embeddings[user_id], 
                top_k=int(target_count * self.recall_weights['vector_similarity'])
            )
            all_candidates.extend([(item_id, score, 'vector_similarity') 
                                 for item_id, score in vector_candidates])
            print(f"   Vector similarity recall: {len(vector_candidates)} items")
        
        # 2. Graph attention recall
        user_history = user_profile.get('history_items', [])
        if user_history:
            graph_candidates = self.recall_by_graph_attention(
                user_id, user_history,
                top_k=int(target_count * self.recall_weights['graph_attention'])
            )
            all_candidates.extend([(item_id, score, 'graph_attention')
                                 for item_id, score in graph_candidates])
            print(f"   Graph attention recall: {len(graph_candidates)} items")
        
        # 3. Category collaborative filtering recall
        user_categories = user_profile.get('preferred_categories', [])
        if user_categories:
            cf_candidates = self.recall_by_category_cf(
                user_id, user_categories,
                top_k=int(target_count * self.recall_weights['category_cf'])
            )
            all_candidates.extend([(item_id, score, 'category_cf')
                                 for item_id, score in cf_candidates])
            print(f"   Category collaborative filtering recall: {len(cf_candidates)} items")
        
        # 4. Hot items recall
        hot_candidates = self.recall_hot_items(
            top_k=int(target_count * self.recall_weights['hot_items'])
        )
        all_candidates.extend([(item_id, score, 'hot_items')
                             for item_id, score in hot_candidates])
        print(f"   Hot items recall: {len(hot_candidates)} items")
        
        # 5. New items recall
        new_candidates = self.recall_new_items(
            top_k=int(target_count * self.recall_weights['new_items'])
        )
        all_candidates.extend([(item_id, score, 'new_items')
                             for item_id, score in new_candidates])
        print(f"   New items recall: {len(new_candidates)} items")
        
        # 6. Seasonal items recall
        current_season = (datetime.now().month - 1) // 3  # 0-3
        seasonal_candidates = self.recall_seasonal_items(
            current_season,
            top_k=int(target_count * self.recall_weights['seasonal_items'])
        )
        all_candidates.extend([(item_id, score, 'seasonal_items')
                             for item_id, score in seasonal_candidates])
        print(f"   Seasonal items recall: {len(seasonal_candidates)} items")
        
        # Deduplication and diversity control
        final_candidates = self._apply_diversity_control(all_candidates, target_count)
        
        print(f"*** Multi-channel recall completed, recalled {len(final_candidates)} candidate items in total ***")
        return final_candidates
    
    def _apply_diversity_control(self, candidates: List[Tuple[str, float, str]], 
                               target_count: int) -> List[Tuple[str, float, str]]:
        """Apply diversity control strategy"""
        
        # Deduplication - merge by score
        candidate_scores = defaultdict(lambda: {'score': 0.0, 'sources': []})
        
        for item_id, score, source in candidates:
            candidate_scores[item_id]['score'] = max(candidate_scores[item_id]['score'], score)
            candidate_scores[item_id]['sources'].append(source)
        
        # Convert to list and sort
        unique_candidates = [
            (item_id, info['score'], ','.join(info['sources']))
            for item_id, info in candidate_scores.items()
        ]
        unique_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Diversity control
        final_candidates = []
        category_count = defaultdict(int)
        brand_count = defaultdict(int)
        
        for item_id, score, sources in unique_candidates:
            if len(final_candidates) >= target_count:
                break
            
            if item_id in self.rga_store.item_embeddings:
                embedding = self.rga_store.item_embeddings[item_id]
                category_id = embedding.category_id
                
                # Simplified: use category_id as brand_id
                brand_id = category_id
                
                # Diversity check
                if (category_count[category_id] < self.diversity_controls['max_same_category'] and
                    brand_count[brand_id] < self.diversity_controls['max_same_brand']):
                    
                    final_candidates.append((item_id, score, sources))
                    category_count[category_id] += 1
                    brand_count[brand_id] += 1
        
        return final_candidates


def demo_rga_recall_system():
    """RGA recall system demonstration"""
    print("*** RGA Vector Recall System Demonstration ***")
    print("=" * 50)
    
    # Initialize recall system
    recall_system = MultiChannelRecallSystem()
    
    print("\n*** 1. Building item vector database ***")
    # Mock adding item vectors
    for i in range(1000):
        item_id = f"item_{i:04d}"
        # Random 128-dimensional vector
        embedding = np.random.rand(128).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)  # Normalize
        
        category_id = random.randint(1, 10)
        
        # Mock graph neighbor relationships
        neighbors = [f"item_{j:04d}" for j in random.sample(range(1000), 3) if j != i]
        
        recall_system.rga_store.add_item_embedding(
            item_id, embedding, category_id, neighbors
        )
        
        # Set popularity
        recall_system.rga_store.item_embeddings[item_id].popularity_score = random.random() * 5.0
    
    print("   * Added 1000 item vectors to RGA vector database")
    
    print("\n*** 2. Adding user vectors ***")
    # Add user vectors
    user_embedding = np.random.rand(128).astype(np.float32)
    user_embedding = user_embedding / np.linalg.norm(user_embedding)
    recall_system.rga_store.add_user_embedding("user_001", user_embedding)
    print("   * User vector addition completed")
    
    print("\n*** 3. Updating RGA attention weights ***")
    # Mock interaction data to update attention weights
    interactions = [
        {
            'related_item_id': 'item_0001',
            'action': 'click',
            'timestamp': '2024-08-30T10:00:00'
        },
        {
            'related_item_id': 'item_0002', 
            'action': 'purchase',
            'timestamp': '2024-08-29T15:30:00'
        }
    ]
    
    recall_system.rga_store.update_attention_weights('item_0000', interactions)
    print("   * Attention weights update completed")
    
    print("\n*** 4. Executing multi-channel recall (target 10000 items) ***")
    
    # User profile
    user_profile = {
        'history_items': ['item_0000', 'item_0001', 'item_0005'],
        'preferred_categories': [1, 2, 3],
        'preferred_brands': [10, 20, 30]
    }
    
    # Multi-channel recall
    recalled_items = recall_system.multi_channel_recall(
        user_id="user_001",
        user_profile=user_profile,
        target_count=10000
    )
    
    print(f"\n*** 5. Recall result analysis ***")
    print(f"   Total recalled items: {len(recalled_items)}")
    
    # Statistics by recall source
    source_stats = defaultdict(int)
    for _, _, sources in recalled_items:
        for source in sources.split(','):
            source_stats[source] += 1
    
    print("   Recall source distribution:")
    for source, count in source_stats.items():
        print(f"     {source}: {count} items")
    
    # Display Top 10 recall results
    print(f"\n*** Top 10 recall results: ***")
    for i, (item_id, score, sources) in enumerate(recalled_items[:10]):
        print(f"   {i+1}. {item_id} | Score: {score:.4f} | Source: {sources}")
    
    print(f"\n*** RGA recall system demonstration completed! ***")
    print("Key features:")
    print("* Real-time graph attention recall based on RGA vectors")
    print("* Multi-channel recall strategy fusion (vector+graph+CF+hot+new+seasonal)")
    print("* Diversity control ensures recall quality")
    print(f"* Successfully recalled {len(recalled_items)} candidate items for subsequent ranking")


if __name__ == "__main__":
    demo_rga_recall_system()