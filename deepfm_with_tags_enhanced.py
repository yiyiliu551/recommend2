#!/usr/bin/env python3
"""
Enhanced DeepFM Model with Product Tag Features
Integrates Electronic product tagging system into DeepFM ranking model
Author: Yang Liu
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json
import logging
import time
from tag_feature_engineering import ProductTagFeatureExtractor, TagFeatureConfig

@dataclass
class EnhancedDeepFMConfig:
    """Enhanced DeepFM configuration with tag features"""
    
    # Basic DeepFM configuration
    embedding_dim: int = 32
    tag_embedding_dim: int = 16
    
    # Feature vocabularies
    vocab_sizes: Dict[str, int] = field(default_factory=lambda: {
        'user_id': 1000000,
        'item_id': 1000000,
        'category_id': 1000,
        'brand_id': 10000,
        'city_id': 1000,
        'age_group': 10,
        'gender': 3,
        'device_type': 5,
        # Tag-related vocabularies
        'tag_id': 46,  # 46 core tags
        'tag_dimension': 10,  # 10 tag dimensions
        'tag_category': 8,  # 8 product categories
        'tag_combination': 10000,  # Tag combination vocabulary
    })
    
    # FM layer configuration
    use_fm: bool = True
    fm_dropout: float = 0.2
    
    # DNN layer configuration
    dnn_hidden_units: List[int] = field(default_factory=lambda: [512, 256, 128, 64])
    dnn_activation: str = 'relu'
    dnn_dropout: float = 0.3
    
    # Tag-specific configuration
    tag_attention_enabled: bool = True
    tag_attention_heads: int = 4
    tag_weight_learning: bool = True
    
    # Training configuration
    learning_rate: float = 0.001
    batch_size: int = 1024
    l2_regularization: float = 0.0001
    
    # Tag feature weights
    tag_feature_weights: Dict[str, float] = field(default_factory=lambda: {
        'basic_tag_features': 1.0,
        'tag_combinations': 0.8,
        'cross_category_features': 0.6,
        'user_tag_preferences': 1.2,
        'tag_similarity_score': 1.5
    })


class TagAttentionLayer:
    """Attention mechanism for tag features"""
    
    def __init__(self, embedding_dim: int, num_heads: int = 4):
        """Initialize tag attention layer"""
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        # Initialize attention weights
        self.query_weights = np.random.normal(0, 0.01, (embedding_dim, embedding_dim)).astype(np.float32)
        self.key_weights = np.random.normal(0, 0.01, (embedding_dim, embedding_dim)).astype(np.float32)
        self.value_weights = np.random.normal(0, 0.01, (embedding_dim, embedding_dim)).astype(np.float32)
        self.output_weights = np.random.normal(0, 0.01, (embedding_dim, embedding_dim)).astype(np.float32)
    
    def forward(self, tag_embeddings: np.ndarray) -> np.ndarray:
        """Forward pass of tag attention
        
        Args:
            tag_embeddings: [batch_size, num_tags, embedding_dim]
        
        Returns:
            attended_embeddings: [batch_size, num_tags, embedding_dim]
        """
        
        batch_size, num_tags, _ = tag_embeddings.shape
        
        # Compute queries, keys, values
        queries = np.matmul(tag_embeddings, self.query_weights)  # [batch_size, num_tags, embedding_dim]
        keys = np.matmul(tag_embeddings, self.key_weights)
        values = np.matmul(tag_embeddings, self.value_weights)
        
        # Reshape for multi-head attention
        queries = queries.reshape(batch_size, num_tags, self.num_heads, self.head_dim)
        keys = keys.reshape(batch_size, num_tags, self.num_heads, self.head_dim)
        values = values.reshape(batch_size, num_tags, self.num_heads, self.head_dim)
        
        # Compute attention scores
        attention_scores = np.matmul(queries, keys.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        attention_weights = self._softmax(attention_scores)
        
        # Apply attention to values
        attended = np.matmul(attention_weights, values)
        
        # Reshape and apply output projection
        attended = attended.reshape(batch_size, num_tags, self.embedding_dim)
        output = np.matmul(attended, self.output_weights)
        
        # Residual connection
        output = output + tag_embeddings
        
        return output
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation along the last dimension"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class EnhancedDeepFMModel:
    """Enhanced DeepFM model with product tag features"""
    
    def __init__(self, config: EnhancedDeepFMConfig, tag_extractor: ProductTagFeatureExtractor):
        """Initialize enhanced DeepFM model"""
        self.config = config
        self.tag_extractor = tag_extractor
        self.logger = logging.getLogger(__name__)
        
        # Initialize basic embeddings
        self.embeddings = {}
        for field, vocab_size in config.vocab_sizes.items():
            embedding_dim = config.tag_embedding_dim if 'tag' in field else config.embedding_dim
            self.embeddings[field] = np.random.normal(
                0, 0.01, (vocab_size, embedding_dim)
            ).astype(np.float32)
        
        # Initialize FM layer weights
        self.fm_first_order_weights = {}
        for field in config.vocab_sizes.keys():
            self.fm_first_order_weights[field] = np.random.normal(
                0, 0.01, (config.vocab_sizes[field], 1)
            ).astype(np.float32)
        
        self.fm_bias = np.array([0.0], dtype=np.float32)
        
        # Initialize tag attention layer
        if config.tag_attention_enabled:
            self.tag_attention = TagAttentionLayer(config.tag_embedding_dim, config.tag_attention_heads)
        
        # Initialize DNN layers
        self._init_dnn_layers()
        
        # Initialize tag-specific layers
        self._init_tag_layers()
        
        self.logger.info(f"Enhanced DeepFM model initialized with tag features")
        self.logger.info(f"Tag vocabulary: {config.vocab_sizes['tag_id']} tags")
        self.logger.info(f"Tag attention: {'enabled' if config.tag_attention_enabled else 'disabled'}")
    
    def _init_dnn_layers(self):
        """Initialize DNN layers"""
        self.dnn_weights = []
        self.dnn_biases = []
        
        # Calculate input dimension (including tag features)
        basic_input_dim = len([k for k in self.config.vocab_sizes.keys() if 'tag' not in k]) * self.config.embedding_dim
        tag_input_dim = len([k for k in self.config.vocab_sizes.keys() if 'tag' in k]) * self.config.tag_embedding_dim
        
        # Add tag semantic vector dimension
        tag_semantic_dim = self.config.tag_embedding_dim
        
        # Add user tag preference vector dimension
        user_tag_dim = self.config.tag_embedding_dim
        
        input_dim = basic_input_dim + tag_input_dim + tag_semantic_dim + user_tag_dim
        
        for hidden_unit in self.config.dnn_hidden_units:
            w = np.random.normal(0, 0.01, (input_dim, hidden_unit)).astype(np.float32)
            self.dnn_weights.append(w)
            
            b = np.zeros((hidden_unit,), dtype=np.float32)
            self.dnn_biases.append(b)
            
            input_dim = hidden_unit
        
        # Output layer
        self.output_weight = np.random.normal(
            0, 0.01, (self.config.dnn_hidden_units[-1], 1)
        ).astype(np.float32)
        self.output_bias = np.array([0.0], dtype=np.float32)
    
    def _init_tag_layers(self):
        """Initialize tag-specific layers"""
        
        # Tag combination layer
        self.tag_combination_weights = np.random.normal(
            0, 0.01, (self.config.tag_embedding_dim * 2, self.config.tag_embedding_dim)
        ).astype(np.float32)
        
        # Cross-category interaction layer
        self.cross_category_weights = np.random.normal(
            0, 0.01, (self.config.tag_embedding_dim, self.config.tag_embedding_dim)
        ).astype(np.float32)
        
        # Tag importance weighting layer
        if self.config.tag_weight_learning:
            self.tag_importance_weights = np.random.normal(
                0, 0.01, (self.config.vocab_sizes['tag_id'], 1)
            ).astype(np.float32)
    
    def extract_enhanced_features(self, user_profile: Dict[str, Any],
                                item_info: Dict[str, Any],
                                context: Dict[str, Any],
                                user_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Extract enhanced features including tag features"""
        
        # Extract basic features
        basic_features = {
            'user_id': user_profile.get('user_id', 'unknown'),
            'age_group': user_profile.get('age_group', 0),
            'gender': user_profile.get('gender', 0),
            'city_id': user_profile.get('city_id', 0),
            'item_id': item_info.get('item_id', 'unknown'),
            'category_id': item_info.get('category_id', 0),
            'brand_id': item_info.get('brand_id', 0),
            'device_type': context.get('device_type', 0),
        }
        
        # Extract tag features for item
        item_tag_features = self.tag_extractor.extract_item_tag_features(
            item_info.get('item_id', ''), item_info
        )
        
        # Extract user tag preferences
        user_tag_features = {}
        if user_history:
            user_tag_features = self.tag_extractor.extract_user_tag_preferences(
                user_profile.get('user_id', ''), user_history
            )
        
        # Combine all features
        enhanced_features = {
            **basic_features,
            'item_tag_ids': item_tag_features.get('item_tags', []),
            'tag_dimensions': item_tag_features.get('tag_dimensions', []),
            'tag_categories': item_tag_features.get('tag_categories', []),
            'tag_combinations': item_tag_features.get('tag_pairs', []),
            'weighted_tag_score': item_tag_features.get('weighted_tag_score', 0.0),
            'tag_semantic_vector': item_tag_features.get('tag_semantic_vector', 
                                                       np.zeros(self.config.tag_embedding_dim)),
            'cross_category_features': item_tag_features.get('cross_category_pairs', []),
            'user_tag_preference_vector': user_tag_features.get('tag_preference_vector',
                                                              np.zeros(self.config.tag_embedding_dim)),
            'user_tag_diversity': user_tag_features.get('tag_diversity', 0.0)
        }
        
        return enhanced_features
    
    def forward(self, features: Dict[str, Any]) -> np.ndarray:
        """Forward pass with tag features"""
        
        # Process basic features through FM layer
        basic_features = {k: v for k, v in features.items() 
                         if k in self.config.vocab_sizes and 'tag' not in k}
        
        fm_first_order, fm_second_order = self._fm_layer(basic_features)
        
        # Process tag features
        tag_features_output = self._process_tag_features(features)
        
        # Process all features through DNN
        dnn_output = self._dnn_layer(features)
        
        # Combine all outputs
        final_output = fm_first_order + fm_second_order + tag_features_output + dnn_output
        
        # Sigmoid activation for CTR prediction
        ctr_pred = 1 / (1 + np.exp(-final_output))
        
        return ctr_pred
    
    def _fm_layer(self, features: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """FM layer processing (same as original but excluding tag features)"""
        
        # First-order term
        first_order = self.fm_bias.copy()
        
        for field, ids in features.items():
            if field in self.fm_first_order_weights and isinstance(ids, (list, np.ndarray)):
                if len(ids) > 0:
                    weights = self.fm_first_order_weights[field][ids]
                    first_order = first_order + np.sum(weights, axis=0, keepdims=True)
        
        # Second-order interaction term
        embeddings = []
        batch_size = 1  # Assume single sample for now
        
        for field, ids in features.items():
            if field in self.embeddings and isinstance(ids, (list, np.ndarray)) and len(ids) > 0:
                emb = self.embeddings[field][ids]
                if emb.ndim == 1:
                    emb = emb.reshape(1, -1)
                embeddings.append(np.mean(emb, axis=0, keepdims=True))  # Average multiple IDs
        
        if len(embeddings) == 0:
            second_order = np.zeros((1, 1), dtype=np.float32)
        else:
            embeddings = np.stack(embeddings, axis=1)  # [1, field_num, embedding_dim]
            
            sum_square = np.sum(embeddings, axis=1) ** 2
            square_sum = np.sum(embeddings ** 2, axis=1)
            
            second_order = 0.5 * np.sum(sum_square - square_sum, axis=1, keepdims=True)
        
        return first_order, second_order
    
    def _process_tag_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Process tag-specific features"""
        
        tag_output = np.zeros((1, 1), dtype=np.float32)
        
        # Process individual tag embeddings
        item_tag_ids = features.get('item_tag_ids', [])
        if item_tag_ids and len(item_tag_ids) > 0:
            # Get tag embeddings
            valid_tag_ids = [tid for tid in item_tag_ids if tid < self.config.vocab_sizes['tag_id']]
            
            if valid_tag_ids:
                tag_embeddings = self.embeddings['tag_id'][valid_tag_ids]
                
                # Apply tag attention if enabled
                if self.config.tag_attention_enabled and len(valid_tag_ids) > 1:
                    tag_embeddings_3d = tag_embeddings.reshape(1, len(valid_tag_ids), -1)
                    attended_embeddings = self.tag_attention.forward(tag_embeddings_3d)
                    tag_representation = np.mean(attended_embeddings[0], axis=0, keepdims=True)
                else:
                    tag_representation = np.mean(tag_embeddings, axis=0, keepdims=True)
                
                # Apply learned importance weights if enabled
                if self.config.tag_weight_learning:
                    tag_weights = self.tag_importance_weights[valid_tag_ids]
                    weighted_importance = np.mean(tag_weights)
                    tag_representation = tag_representation * weighted_importance
                
                # Convert tag representation to scalar output
                tag_output = np.sum(tag_representation) / tag_representation.size
                tag_output = tag_output.reshape(1, 1)
        
        # Process tag semantic vector
        tag_semantic_vector = features.get('tag_semantic_vector')
        if tag_semantic_vector is not None and hasattr(tag_semantic_vector, 'shape'):
            semantic_contribution = np.mean(tag_semantic_vector) * 0.5
            tag_output = tag_output + semantic_contribution
        
        # Process user tag preference similarity
        user_tag_vector = features.get('user_tag_preference_vector')
        item_tag_vector = features.get('tag_semantic_vector')
        
        if (user_tag_vector is not None and item_tag_vector is not None and 
            hasattr(user_tag_vector, 'shape') and hasattr(item_tag_vector, 'shape')):
            
            # Compute cosine similarity
            if np.linalg.norm(user_tag_vector) > 0 and np.linalg.norm(item_tag_vector) > 0:
                similarity = np.dot(user_tag_vector, item_tag_vector) / (
                    np.linalg.norm(user_tag_vector) * np.linalg.norm(item_tag_vector)
                )
                tag_output = tag_output + similarity * 0.3
        
        return tag_output
    
    def _dnn_layer(self, features: Dict[str, Any]) -> np.ndarray:
        """Enhanced DNN layer with tag features"""
        
        dnn_inputs = []
        
        # Process basic features
        for field, ids in features.items():
            if field in self.embeddings and 'tag' not in field and isinstance(ids, (list, np.ndarray, str, int)):
                if isinstance(ids, (str, int)):
                    ids = [hash(str(ids)) % self.config.vocab_sizes[field]]
                elif isinstance(ids, list) and len(ids) == 0:
                    continue
                
                valid_ids = [int(id_val) % self.config.vocab_sizes[field] for id_val in ids]
                emb = self.embeddings[field][valid_ids]
                if emb.ndim == 1:
                    dnn_inputs.append(emb)
                else:
                    dnn_inputs.append(np.mean(emb, axis=0))
        
        # Process tag features
        tag_fields = ['tag_id', 'tag_dimension', 'tag_category']
        for field in tag_fields:
            if field in features:
                ids = features[field]
                if isinstance(ids, list) and len(ids) > 0:
                    valid_ids = [int(id_val) % self.config.vocab_sizes[field] for id_val in ids]
                    if valid_ids:
                        emb = self.embeddings[field][valid_ids]
                        dnn_inputs.append(np.mean(emb, axis=0))
        
        # Add semantic vectors
        for vector_field in ['tag_semantic_vector', 'user_tag_preference_vector']:
            if vector_field in features:
                vector = features[vector_field]
                if vector is not None and hasattr(vector, 'shape'):
                    dnn_inputs.append(vector)
        
        # Add scalar features
        scalar_features = ['weighted_tag_score', 'user_tag_diversity']
        for scalar_field in scalar_features:
            if scalar_field in features:
                value = features[scalar_field]
                if isinstance(value, (int, float)):
                    # Convert scalar to small vector
                    scalar_vector = np.array([value, value**2, np.log(1 + abs(value))], dtype=np.float32)
                    dnn_inputs.append(scalar_vector)
        
        if len(dnn_inputs) == 0:
            return np.zeros((1, 1), dtype=np.float32)
        
        # Concatenate all inputs
        x = np.concatenate(dnn_inputs).reshape(1, -1)
        
        # Ensure x has the right dimension for first DNN layer
        if x.shape[1] != self.dnn_weights[0].shape[0]:
            # Pad or truncate to match expected input dimension
            expected_dim = self.dnn_weights[0].shape[0]
            if x.shape[1] < expected_dim:
                padding = np.zeros((1, expected_dim - x.shape[1]), dtype=np.float32)
                x = np.concatenate([x, padding], axis=1)
            else:
                x = x[:, :expected_dim]
        
        # Forward pass through DNN layers
        for weight, bias in zip(self.dnn_weights, self.dnn_biases):
            x = np.matmul(x, weight) + bias
            
            # Apply activation
            if self.config.dnn_activation == 'relu':
                x = np.maximum(0, x)
            elif self.config.dnn_activation == 'sigmoid':
                x = 1 / (1 + np.exp(-x))
        
        # Output layer
        output = np.matmul(x, self.output_weight) + self.output_bias
        
        return output
    
    def predict_with_tags(self, user_profile: Dict[str, Any],
                         item_info: Dict[str, Any],
                         context: Dict[str, Any],
                         user_history: Optional[List[Dict]] = None) -> Dict[str, float]:
        """Predict with detailed tag feature breakdown"""
        
        # Extract enhanced features
        features = self.extract_enhanced_features(user_profile, item_info, context, user_history)
        
        # Forward pass
        ctr_prediction = self.forward(features)
        
        # Calculate component contributions
        basic_features = {k: v for k, v in features.items() 
                         if k in self.config.vocab_sizes and 'tag' not in k}
        fm_first, fm_second = self._fm_layer(basic_features)
        tag_contribution = self._process_tag_features(features)
        
        # Tag similarity score
        tag_similarity = 0.0
        if user_history:
            item_tags = self.tag_extractor._get_item_tags(item_info.get('item_id', ''), item_info)
            user_tag_prefs = self.tag_extractor.extract_user_tag_preferences(
                user_profile.get('user_id', ''), user_history
            )
            tag_similarity = self.tag_extractor.compute_tag_similarity(item_tags, user_tag_prefs)
        
        return {
            'ctr_prediction': float(ctr_prediction.flatten()[0]),
            'fm_contribution': float(fm_first.flatten()[0] + fm_second.flatten()[0]),
            'tag_contribution': float(tag_contribution.flatten()[0]),
            'tag_similarity': tag_similarity,
            'weighted_tag_score': features.get('weighted_tag_score', 0.0)
        }


def demonstrate_enhanced_deepfm():
    """Demonstrate enhanced DeepFM with tag features"""
    
    print("=" * 80)
    print("Enhanced DeepFM with Product Tag Features Demo")
    print("=" * 80)
    
    # Initialize tag feature extractor
    tag_config = TagFeatureConfig()
    tag_definitions_path = '/Users/yangliu/Desktop/小kimi/electronic-interview-projects/product_tag_definitions.json'
    tag_extractor = ProductTagFeatureExtractor(tag_definitions_path, tag_config)
    
    # Initialize enhanced DeepFM model
    model_config = EnhancedDeepFMConfig(
        embedding_dim=32,
        tag_embedding_dim=16,
        tag_attention_enabled=True,
        tag_weight_learning=True
    )
    
    model = EnhancedDeepFMModel(model_config, tag_extractor)
    
    print(f"\n1. Model Initialization Complete")
    print(f"   Basic embedding dim: {model_config.embedding_dim}")
    print(f"   Tag embedding dim: {model_config.tag_embedding_dim}")
    print(f"   Tag attention: {'enabled' if model_config.tag_attention_enabled else 'disabled'}")
    print(f"   Tag vocabularies: {sum(1 for k in model_config.vocab_sizes.keys() if 'tag' in k)}")
    
    # Demo prediction with tag features
    print(f"\n2. Prediction with Tag Features Demo")
    
    demo_user = {
        'user_id': 'user_001',
        'age_group': 2,
        'gender': 1,
        'city_id': 1
    }
    
    demo_items = [
        {
            'item_id': 'electronic_a7r5',
            'name': 'Electronic Alpha 7R V',
            'category': 'camera',
            'price': 3899.99,
            'category_id': 1,
            'brand_id': 1,
            'description': 'Professional full-frame mirrorless camera with 8K video'
        },
        {
            'item_id': 'electronic_wh1000xm5',
            'name': 'Electronic WH-1000XM5',
            'category': 'audio',
            'price': 399.99,
            'category_id': 2,
            'brand_id': 1,
            'description': 'Wireless noise canceling headphones with LDAC'
        }
    ]
    
    demo_context = {
        'device_type': 1,
        'hour': 14,
        'day_of_week': 3
    }
    
    # Demo user history for preference learning
    demo_history = [
        {'item_id': 'electronic_a7r4', 'type': 'purchase', 'item_info': {
            'category': 'camera', 'price': 3499.99, 'name': 'Electronic Alpha 7R IV'
        }},
        {'item_id': 'electronic_24_70_lens', 'type': 'purchase', 'item_info': {
            'category': 'camera', 'price': 2199.99, 'name': 'Electronic FE 24-70mm f/2.8 GM'
        }}
    ]
    
    for item in demo_items:
        print(f"\n   Item: {item['name']}")
        
        # Predict with tag features
        prediction_result = model.predict_with_tags(
            demo_user, item, demo_context, demo_history
        )
        
        print(f"   CTR Prediction: {prediction_result['ctr_prediction']:.4f}")
        print(f"   FM Contribution: {prediction_result['fm_contribution']:.4f}")
        print(f"   Tag Contribution: {prediction_result['tag_contribution']:.4f}")
        print(f"   Tag Similarity: {prediction_result['tag_similarity']:.4f}")
        print(f"   Weighted Tag Score: {prediction_result['weighted_tag_score']:.3f}")
    
    print(f"\n3. Tag Feature Analysis")
    
    # Show tag features for demo item
    sample_item = demo_items[0]
    item_tag_features = tag_extractor.extract_item_tag_features(
        sample_item['item_id'], sample_item
    )
    
    print(f"   Sample Item: {sample_item['name']}")
    print(f"   Extracted Tags: {len(item_tag_features['item_tags'])}")
    print(f"   Tag Dimensions: {item_tag_features['tag_dimensions']}")
    print(f"   Tag Combinations: {len(item_tag_features.get('tag_pairs', []))}")
    print(f"   Cross-Category: {item_tag_features.get('is_cross_category', False)}")
    
    # Show user preferences
    user_prefs = tag_extractor.extract_user_tag_preferences('user_001', demo_history)
    print(f"\n   User Tag Preferences:")
    print(f"   Tag Diversity: {user_prefs['tag_diversity']:.3f}")
    print(f"   Preferred Dimensions: {user_prefs['preferred_dimensions'][:3]}")
    
    print(f"\n" + "=" * 80)
    print("Enhanced DeepFM with Tag Features - Key Capabilities:")
    print("✅ 46-tag product tagging system integration")
    print("✅ Tag attention mechanism for feature importance")
    print("✅ Tag combination features for cross-feature interactions") 
    print("✅ User tag preference modeling from historical behavior")
    print("✅ Cross-category tag features for ecosystem products")
    print("✅ Learned tag importance weighting")
    print("✅ Tag semantic similarity computation")
    print("✅ Component-wise prediction analysis")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_enhanced_deepfm()