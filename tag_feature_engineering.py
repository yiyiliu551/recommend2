#!/usr/bin/env python3
"""
Electronic Product Tag Feature Engineering Pipeline
Integrates product tagging system into DeepFM ranking model training
Author: Yang Liu
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import logging
import time
from datetime import datetime, timedelta

@dataclass
class TagFeatureConfig:
    """Tag feature engineering configuration"""
    
    # Tag embedding settings
    tag_embedding_dim: int = 16
    tag_categories: List[str] = field(default_factory=lambda: [
        'basic_attributes', 'technical_specs', 'usage_scenarios',
        'target_audience', 'price_positioning', 'design_aesthetics',
        'features_functions', 'compatibility', 'brand_series', 'market_positioning'
    ])
    
    # Feature combination settings
    enable_tag_combinations: bool = True
    max_combination_order: int = 2
    combination_threshold: float = 0.1
    
    # Tag importance weighting
    tag_weights: Dict[str, float] = field(default_factory=lambda: {
        'basic_attributes': 1.2,
        'technical_specs': 1.1,
        'usage_scenarios': 1.0,
        'target_audience': 1.3,
        'price_positioning': 1.1,
        'design_aesthetics': 0.9,
        'features_functions': 1.2,
        'compatibility': 0.8,
        'brand_series': 1.0,
        'market_positioning': 1.1
    })
    
    # Temporal decay settings
    enable_temporal_decay: bool = True
    decay_half_life: int = 30  # days
    
    # Cross-category feature settings
    enable_cross_category: bool = True
    cross_category_pairs: List[Tuple[str, str]] = field(default_factory=lambda: [
        ('camera', 'audio'), ('gaming', 'tv_display'), 
        ('audio', 'mobile'), ('camera', 'professional')
    ])


class ProductTagFeatureExtractor:
    """Product tag feature extraction and engineering"""
    
    def __init__(self, tag_definitions_path: str, config: TagFeatureConfig):
        """Initialize tag feature extractor"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Load tag definitions
        with open(tag_definitions_path, 'r', encoding='utf-8') as f:
            self.tag_definitions = json.load(f)
        
        # Build tag mappings
        self.tag_id_to_info = {}
        self.dimension_to_tags = defaultdict(list)
        self.category_to_tags = defaultdict(list)
        
        for tag_id, tag_info in self.tag_definitions['tags'].items():
            self.tag_id_to_info[tag_id] = tag_info
            self.dimension_to_tags[tag_info['dimension']].append(tag_id)
            
            category = tag_info.get('category', 'all')
            self.category_to_tags[category].append(tag_id)
        
        # Initialize feature vocabularies
        self._build_feature_vocabularies()
        
        # Initialize tag co-occurrence matrix
        self.tag_cooccurrence = defaultdict(lambda: defaultdict(int))
        self.tag_frequencies = defaultdict(int)
        
        self.logger.info(f"Tag feature extractor initialized with {len(self.tag_id_to_info)} tags")
    
    def _build_feature_vocabularies(self):
        """Build feature vocabularies for embedding"""
        
        # Individual tag vocabulary
        self.tag_vocab = {}
        for i, tag_id in enumerate(sorted(self.tag_id_to_info.keys())):
            self.tag_vocab[tag_id] = i
        
        # Tag dimension vocabulary
        self.dimension_vocab = {}
        for i, dimension in enumerate(self.config.tag_categories):
            self.dimension_vocab[dimension] = i
        
        # Category vocabulary
        self.category_vocab = {}
        categories = set()
        for tag_info in self.tag_id_to_info.values():
            categories.add(tag_info.get('category', 'all'))
        
        for i, category in enumerate(sorted(categories)):
            self.category_vocab[category] = i
        
        self.logger.info(f"Built vocabularies: {len(self.tag_vocab)} tags, "
                        f"{len(self.dimension_vocab)} dimensions, "
                        f"{len(self.category_vocab)} categories")
    
    def extract_item_tag_features(self, item_id: str, item_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract tag-based features for a single item"""
        
        # Get item tags (from auto-tagging or manual tags)
        item_tags = self._get_item_tags(item_id, item_info)
        
        # Basic tag features
        tag_features = {
            'item_tags': [self.tag_vocab.get(tag, 0) for tag in item_tags],
            'tag_count': len(item_tags),
            'tag_dimensions': self._get_tag_dimensions(item_tags),
            'tag_categories': self._get_tag_categories(item_tags)
        }
        
        # Tag combination features
        if self.config.enable_tag_combinations:
            tag_features.update(self._extract_tag_combinations(item_tags))
        
        # Cross-category features
        if self.config.enable_cross_category:
            tag_features.update(self._extract_cross_category_features(item_tags, item_info))
        
        # Weighted tag scores
        tag_features['weighted_tag_score'] = self._calculate_weighted_tag_score(item_tags)
        
        # Tag semantic embeddings (pre-computed or learned)
        tag_features['tag_semantic_vector'] = self._get_tag_semantic_embedding(item_tags)
        
        return tag_features
    
    def _get_item_tags(self, item_id: str, item_info: Dict[str, Any]) -> List[str]:
        """Get tags for an item using auto-tagging rules"""
        
        assigned_tags = []
        
        # Price-based auto-tagging
        price = item_info.get('price', 0)
        for rule in self.tag_definitions.get('auto_tagging_rules', {}).get('price_based', []):
            condition = rule['condition']
            if eval(condition.replace('price', str(price))):
                assigned_tags.extend(rule['assign_tags'])
        
        # Keyword-based auto-tagging
        item_name = item_info.get('name', '').lower()
        item_description = item_info.get('description', '').lower()
        item_text = f"{item_name} {item_description}"
        
        for rule in self.tag_definitions.get('auto_tagging_rules', {}).get('keyword_based', []):
            for keyword in rule['keywords']:
                if keyword.lower() in item_text:
                    assigned_tags.extend(rule['assign_tags'])
        
        # Series-based auto-tagging
        for rule in self.tag_definitions.get('auto_tagging_rules', {}).get('series_based', []):
            import re
            pattern = rule['model_pattern']
            if re.search(pattern, item_id, re.IGNORECASE) or re.search(pattern, item_name, re.IGNORECASE):
                assigned_tags.extend(rule['assign_tags'])
        
        # Category-based default tags
        category = item_info.get('category')
        if category:
            assigned_tags.extend(self.category_to_tags.get(category, []))
        
        # Remove duplicates and return
        return list(set(assigned_tags))
    
    def _get_tag_dimensions(self, item_tags: List[str]) -> List[int]:
        """Get tag dimensions for item tags"""
        dimensions = set()
        for tag in item_tags:
            if tag in self.tag_id_to_info:
                dimension = self.tag_id_to_info[tag]['dimension']
                if dimension in self.dimension_vocab:
                    dimensions.add(self.dimension_vocab[dimension])
        return list(dimensions)
    
    def _get_tag_categories(self, item_tags: List[str]) -> List[int]:
        """Get tag categories for item tags"""
        categories = set()
        for tag in item_tags:
            if tag in self.tag_id_to_info:
                category = self.tag_id_to_info[tag].get('category', 'all')
                if category in self.category_vocab:
                    categories.add(self.category_vocab[category])
        return list(categories)
    
    def _extract_tag_combinations(self, item_tags: List[str]) -> Dict[str, Any]:
        """Extract tag combination features"""
        
        combinations = {}
        
        # Second-order combinations
        if len(item_tags) >= 2 and self.config.max_combination_order >= 2:
            tag_pairs = []
            for i, tag1 in enumerate(item_tags):
                for j, tag2 in enumerate(item_tags):
                    if i < j:
                        # Create combination ID
                        pair_id = f"{tag1}_{tag2}"
                        tag_pairs.append(hash(pair_id) % 10000)  # Hash to fixed vocabulary
            
            combinations['tag_pairs'] = tag_pairs[:10]  # Limit to top 10 pairs
        
        # Cross-dimension combinations
        dimension_tags = defaultdict(list)
        for tag in item_tags:
            if tag in self.tag_id_to_info:
                dimension = self.tag_id_to_info[tag]['dimension']
                dimension_tags[dimension].append(tag)
        
        cross_dimension_features = []
        dimensions = list(dimension_tags.keys())
        for i, dim1 in enumerate(dimensions):
            for j, dim2 in enumerate(dimensions):
                if i < j and len(dimension_tags[dim1]) > 0 and len(dimension_tags[dim2]) > 0:
                    # Create cross-dimension feature
                    cross_feature = f"{dim1}_{dim2}"
                    cross_dimension_features.append(hash(cross_feature) % 1000)
        
        combinations['cross_dimension_features'] = cross_dimension_features[:5]
        
        return combinations
    
    def _extract_cross_category_features(self, item_tags: List[str], 
                                       item_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract cross-category features"""
        
        cross_features = {}
        item_category = item_info.get('category')
        
        # Check if item has tags from multiple categories
        tag_categories = set()
        for tag in item_tags:
            if tag in self.tag_id_to_info:
                tag_category = self.tag_id_to_info[tag].get('category', 'all')
                if tag_category != 'all':
                    tag_categories.add(tag_category)
        
        cross_features['is_cross_category'] = len(tag_categories) > 1
        cross_features['cross_category_count'] = len(tag_categories)
        
        # Specific cross-category pair features
        cross_category_pairs = []
        for pair in self.config.cross_category_pairs:
            if pair[0] in tag_categories and pair[1] in tag_categories:
                pair_id = f"{pair[0]}_{pair[1]}"
                cross_category_pairs.append(hash(pair_id) % 100)
        
        cross_features['cross_category_pairs'] = cross_category_pairs
        
        return cross_features
    
    def _calculate_weighted_tag_score(self, item_tags: List[str]) -> float:
        """Calculate weighted tag score based on tag importance"""
        
        total_score = 0.0
        for tag in item_tags:
            if tag in self.tag_id_to_info:
                dimension = self.tag_id_to_info[tag]['dimension']
                weight = self.config.tag_weights.get(dimension, 1.0)
                
                # Consider business value
                business_value_keywords = [
                    'high_end', 'premium', 'professional', 'flagship',
                    'core', 'standard', 'differentiation'
                ]
                
                business_value = self.tag_id_to_info[tag].get('business_value', '').lower()
                business_bonus = 1.0
                
                for keyword in business_value_keywords:
                    if keyword in business_value:
                        business_bonus += 0.1
                
                total_score += weight * business_bonus
        
        return total_score / max(len(item_tags), 1)  # Normalize by tag count
    
    def _get_tag_semantic_embedding(self, item_tags: List[str]) -> np.ndarray:
        """Get semantic embedding for item tags"""
        
        # Initialize embedding vector
        embedding_dim = self.config.tag_embedding_dim
        semantic_vector = np.zeros(embedding_dim, dtype=np.float32)
        
        if not item_tags:
            return semantic_vector
        
        # Simple approach: average of tag embeddings (normally would use pre-trained embeddings)
        for tag in item_tags:
            if tag in self.tag_vocab:
                # Simulate tag embedding (in practice would load from pre-trained embeddings)
                tag_id = self.tag_vocab[tag]
                
                # Create pseudo-embedding based on tag properties
                tag_info = self.tag_id_to_info.get(tag, {})
                
                # Use tag dimension and type to create distinctive embeddings
                dimension_hash = hash(tag_info.get('dimension', '')) % embedding_dim
                type_hash = hash(tag_info.get('type', '')) % embedding_dim
                category_hash = hash(tag_info.get('category', '')) % embedding_dim
                
                tag_embedding = np.zeros(embedding_dim)
                tag_embedding[dimension_hash % embedding_dim] += 1.0
                tag_embedding[type_hash % embedding_dim] += 0.5
                tag_embedding[category_hash % embedding_dim] += 0.3
                
                semantic_vector += tag_embedding
        
        # Normalize
        if np.linalg.norm(semantic_vector) > 0:
            semantic_vector = semantic_vector / np.linalg.norm(semantic_vector)
        
        return semantic_vector
    
    def extract_user_tag_preferences(self, user_id: str, 
                                   user_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract user tag preferences from interaction history"""
        
        # Tag interaction statistics
        tag_interactions = defaultdict(int)
        tag_positive_interactions = defaultdict(int)
        dimension_preferences = defaultdict(float)
        category_preferences = defaultdict(float)
        
        total_interactions = len(user_history)
        
        for interaction in user_history:
            item_info = interaction.get('item_info', {})
            interaction_type = interaction.get('type', 'click')
            
            # Get item tags
            item_tags = self._get_item_tags(interaction['item_id'], item_info)
            
            # Weight interactions differently
            interaction_weight = {
                'click': 1.0,
                'add_to_cart': 2.0,
                'purchase': 3.0,
                'favorite': 1.5,
                'share': 1.2
            }.get(interaction_type, 1.0)
            
            for tag in item_tags:
                tag_interactions[tag] += 1
                
                # Positive interactions (purchases, favorites, etc.)
                if interaction_type in ['purchase', 'add_to_cart', 'favorite']:
                    tag_positive_interactions[tag] += interaction_weight
                
                # Update dimension preferences
                if tag in self.tag_id_to_info:
                    dimension = self.tag_id_to_info[tag]['dimension']
                    dimension_preferences[dimension] += interaction_weight
                    
                    category = self.tag_id_to_info[tag].get('category', 'all')
                    if category != 'all':
                        category_preferences[category] += interaction_weight
        
        # Calculate preference scores
        user_tag_features = {
            'total_interactions': total_interactions,
            'unique_tags_interacted': len(tag_interactions),
            'preferred_tags': self._get_top_preferences(tag_positive_interactions, 10),
            'preferred_dimensions': self._get_top_preferences(dimension_preferences, 5),
            'preferred_categories': self._get_top_preferences(category_preferences, 3),
        }
        
        # Tag diversity score
        if total_interactions > 0:
            user_tag_features['tag_diversity'] = len(tag_interactions) / total_interactions
        else:
            user_tag_features['tag_diversity'] = 0.0
        
        # Tag preference vector
        user_tag_features['tag_preference_vector'] = self._create_user_tag_vector(
            tag_positive_interactions
        )
        
        return user_tag_features
    
    def _get_top_preferences(self, preferences: Dict[str, float], top_k: int) -> List[Tuple[str, float]]:
        """Get top preferences sorted by score"""
        sorted_prefs = sorted(preferences.items(), key=lambda x: x[1], reverse=True)
        return sorted_prefs[:top_k]
    
    def _create_user_tag_vector(self, tag_preferences: Dict[str, float]) -> np.ndarray:
        """Create user tag preference vector"""
        
        # Initialize user vector with same dimension as tag semantic embeddings
        user_vector = np.zeros(self.config.tag_embedding_dim, dtype=np.float32)
        
        total_score = sum(tag_preferences.values())
        if total_score == 0:
            return user_vector
        
        # Weighted average of preferred tag embeddings
        for tag, score in tag_preferences.items():
            if tag in self.tag_id_to_info:
                tag_embedding = self._get_tag_semantic_embedding([tag])
                weight = score / total_score
                user_vector += weight * tag_embedding
        
        return user_vector
    
    def compute_tag_similarity(self, item_tags: List[str], 
                              user_tag_preferences: Dict[str, Any]) -> float:
        """Compute tag-based similarity between item and user preferences"""
        
        if not item_tags or not user_tag_preferences:
            return 0.0
        
        # Get item tag vector
        item_tag_vector = self._get_tag_semantic_embedding(item_tags)
        
        # Get user preference vector
        user_tag_vector = user_tag_preferences.get('tag_preference_vector', 
                                                  np.zeros(self.config.tag_embedding_dim))
        
        # Compute cosine similarity
        if np.linalg.norm(item_tag_vector) == 0 or np.linalg.norm(user_tag_vector) == 0:
            return 0.0
        
        similarity = np.dot(item_tag_vector, user_tag_vector) / (
            np.linalg.norm(item_tag_vector) * np.linalg.norm(user_tag_vector)
        )
        
        return max(0.0, similarity)  # Ensure non-negative similarity
    
    def update_tag_cooccurrence(self, items: List[Dict[str, Any]]):
        """Update tag co-occurrence statistics for better feature engineering"""
        
        for item in items:
            item_tags = self._get_item_tags(item['item_id'], item)
            
            # Update individual tag frequencies
            for tag in item_tags:
                self.tag_frequencies[tag] += 1
            
            # Update tag co-occurrence
            for i, tag1 in enumerate(item_tags):
                for j, tag2 in enumerate(item_tags):
                    if i != j:
                        self.tag_cooccurrence[tag1][tag2] += 1
        
        self.logger.info(f"Updated tag statistics: {len(self.tag_frequencies)} unique tags")


def demonstrate_tag_feature_engineering():
    """Demonstrate tag feature engineering capabilities"""
    
    print("=" * 80)
    print("Electronic Product Tag Feature Engineering Demo")
    print("=" * 80)
    
    # Initialize configuration
    config = TagFeatureConfig(
        tag_embedding_dim=16,
        enable_tag_combinations=True,
        enable_cross_category=True
    )
    
    # Initialize feature extractor
    tag_definitions_path = '/Users/yangliu/Desktop/小kimi/electronic-interview-projects/product_tag_definitions.json'
    extractor = ProductTagFeatureExtractor(tag_definitions_path, config)
    
    print(f"\n1. Tag Feature Extractor Initialized")
    print(f"   Tag vocabulary size: {len(extractor.tag_vocab)}")
    print(f"   Dimension vocabulary size: {len(extractor.dimension_vocab)}")
    print(f"   Category vocabulary size: {len(extractor.category_vocab)}")
    
    # Demo item feature extraction
    print(f"\n2. Item Tag Feature Extraction Demo")
    demo_items = [
        {
            'item_id': 'electronic_a7r5',
            'name': 'Electronic Alpha 7R V',
            'category': 'camera',
            'price': 3899.99,
            'description': 'Professional full-frame mirrorless camera with 8K video'
        },
        {
            'item_id': 'electronic_wh1000xm5',
            'name': 'Electronic WH-1000XM5',
            'category': 'audio',
            'price': 399.99,
            'description': 'Wireless noise canceling headphones with LDAC'
        },
        {
            'item_id': 'ps5_console',
            'name': 'PlayStation 5',
            'category': 'gaming',
            'price': 499.99,
            'description': '4K gaming console with ray tracing support'
        }
    ]
    
    for item in demo_items:
        print(f"\n   Item: {item['name']}")
        
        # Extract tag features
        tag_features = extractor.extract_item_tag_features(item['item_id'], item)
        
        print(f"   Tags found: {len(tag_features['item_tags'])}")
        print(f"   Tag dimensions: {tag_features['tag_dimensions']}")
        print(f"   Weighted tag score: {tag_features['weighted_tag_score']:.3f}")
        print(f"   Cross-category: {tag_features.get('is_cross_category', False)}")
        
        if 'tag_pairs' in tag_features:
            print(f"   Tag combinations: {len(tag_features['tag_pairs'])} pairs")
    
    # Demo user preference extraction
    print(f"\n3. User Tag Preference Extraction Demo")
    
    demo_user_history = [
        {'item_id': 'electronic_a7r5', 'type': 'purchase', 'item_info': demo_items[0]},
        {'item_id': 'electronic_wh1000xm5', 'type': 'click', 'item_info': demo_items[1]},
        {'item_id': 'ps5_console', 'type': 'add_to_cart', 'item_info': demo_items[2]},
    ]
    
    user_preferences = extractor.extract_user_tag_preferences('demo_user', demo_user_history)
    
    print(f"   Total interactions: {user_preferences['total_interactions']}")
    print(f"   Unique tags: {user_preferences['unique_tags_interacted']}")
    print(f"   Tag diversity: {user_preferences['tag_diversity']:.3f}")
    print(f"   Top preferred dimensions: {user_preferences['preferred_dimensions'][:3]}")
    
    # Demo tag similarity computation
    print(f"\n4. Tag Similarity Computation Demo")
    
    for item in demo_items:
        item_tags = extractor._get_item_tags(item['item_id'], item)
        similarity = extractor.compute_tag_similarity(item_tags, user_preferences)
        print(f"   {item['name']}: similarity = {similarity:.3f}")
    
    print(f"\n" + "=" * 80)
    print("Tag Feature Engineering System Features:")
    print("✅ 46-tag comprehensive tagging system with 10 dimensions")
    print("✅ Auto-tagging rules based on price, keywords, and series patterns")
    print("✅ Tag combination features for second-order interactions")
    print("✅ Cross-category features for ecosystem products")
    print("✅ Weighted tag scoring based on business value")
    print("✅ User tag preference modeling from interaction history")
    print("✅ Tag semantic similarity computation")
    print("✅ Tag co-occurrence statistics for feature optimization")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_tag_feature_engineering()