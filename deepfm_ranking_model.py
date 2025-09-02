"""
DeepFM Coarse Ranking Model - E-commerce Recommendation Coarse Ranking Layer
Author: Yang Liu
Description: DeepFM model implementation, coarse ranking to filter 10000 recalled items down to 1000 items
Tech Stack: TensorFlow/PyTorch + Wide&Deep architecture + FM feature crossing + DNN deep features
Experience: Large-scale CTR prediction practices from Weibo 400M DAU, Qunar, SHAREit
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import json
import pickle
import random
from datetime import datetime

# Mock deep learning framework
class MockTensor:
    """Mock tensor class"""
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float32)
        self.shape = self.data.shape
    
    def __matmul__(self, other):
        return MockTensor(np.matmul(self.data, other.data))
    
    def __add__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self.data + other.data)
        return MockTensor(self.data + other)
    
    def __mul__(self, other):
        if isinstance(other, MockTensor):
            return MockTensor(self.data * other.data)
        return MockTensor(self.data * other)
    
    def sigmoid(self):
        return MockTensor(1 / (1 + np.exp(-self.data)))
    
    def relu(self):
        return MockTensor(np.maximum(0, self.data))
    
    def sum(self, axis=None):
        return MockTensor(np.sum(self.data, axis=axis))


@dataclass
class DeepFMConfig:
    """DeepFM model configuration"""
    
    # Embedding layer configuration
    embedding_dim: int = 32
    vocab_sizes: Dict[str, int] = None
    
    # FM layer configuration
    use_fm: bool = True
    fm_dropout: float = 0.2
    
    # DNN layer configuration
    dnn_hidden_units: List[int] = None
    dnn_activation: str = 'relu'
    dnn_dropout: float = 0.3
    
    # Training configuration
    learning_rate: float = 0.001
    batch_size: int = 1024
    l2_regularization: float = 0.0001
    
    def __post_init__(self):
        if self.vocab_sizes is None:
            self.vocab_sizes = {
                'user_id': 1000000,
                'item_id': 1000000, 
                'category_id': 1000,
                'brand_id': 10000,
                'city_id': 1000,
                'age_group': 10,
                'gender': 3,
                'device_type': 5
            }
        
        if self.dnn_hidden_units is None:
            self.dnn_hidden_units = [512, 256, 128, 64]


class DeepFMModel:
    """DeepFM model implementation"""
    
    def __init__(self, config: DeepFMConfig):
        """Initialize DeepFM model"""
        self.config = config
        
        # Initialize embedding layers
        self.embeddings = {}
        for field, vocab_size in config.vocab_sizes.items():
            # Embedding matrix: [vocab_size, embedding_dim]
            self.embeddings[field] = np.random.normal(
                0, 0.01, (vocab_size, config.embedding_dim)
            ).astype(np.float32)
        
        # Initialize FM layer weights
        self.fm_first_order_weights = {}  # First-order weights
        for field in config.vocab_sizes.keys():
            self.fm_first_order_weights[field] = np.random.normal(
                0, 0.01, (config.vocab_sizes[field], 1)
            ).astype(np.float32)
        
        self.fm_bias = np.array([0.0], dtype=np.float32)  # Bias term
        
        # Initialize DNN layer weights
        self.dnn_weights = []
        self.dnn_biases = []
        
        # Input dimension = number of embedding fields * embedding dimension
        input_dim = len(config.vocab_sizes) * config.embedding_dim
        
        for hidden_unit in config.dnn_hidden_units:
            # Weight matrix
            w = np.random.normal(0, 0.01, (input_dim, hidden_unit)).astype(np.float32)
            self.dnn_weights.append(w)
            
            # Bias vector
            b = np.zeros((hidden_unit,), dtype=np.float32)
            self.dnn_biases.append(b)
            
            input_dim = hidden_unit
        
        # Output layer
        self.output_weight = np.random.normal(
            0, 0.01, (config.dnn_hidden_units[-1], 1)
        ).astype(np.float32)
        self.output_bias = np.array([0.0], dtype=np.float32)
        
        print(f"🤖 DeepFM model initialization completed")
        print(f"   Embedding dimension: {config.embedding_dim}")
        print(f"   DNN structure: {config.dnn_hidden_units}")
        print(f"   Feature fields: {list(config.vocab_sizes.keys())}")
    
    def get_embedding(self, field: str, ids: np.ndarray) -> np.ndarray:
        """Get field embedding vectors"""
        if field not in self.embeddings:
            raise ValueError(f"Unknown field: {field}")
        
        # Handle out-of-bounds IDs
        vocab_size = self.config.vocab_sizes[field]
        ids = np.clip(ids, 0, vocab_size - 1)
        
        return self.embeddings[field][ids]
    
    def fm_layer(self, features: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """FM layer computation - first-order terms + second-order interaction terms"""
        
        # First-order term computation
        first_order = self.fm_bias.copy()
        
        for field, ids in features.items():
            if field in self.fm_first_order_weights:
                # Get first-order weights
                weights = self.fm_first_order_weights[field][ids]  # [batch_size, 1]
                first_order = first_order + np.sum(weights, axis=0, keepdims=True)
        
        # Second-order interaction term computation
        # Get embedding vectors for all fields
        embeddings = []  # [batch_size, field_num, embedding_dim]
        
        batch_size = len(list(features.values())[0])
        
        for field, ids in features.items():
            if field in self.embeddings:
                emb = self.get_embedding(field, ids)  # [batch_size, embedding_dim]
                embeddings.append(emb)
        
        if len(embeddings) == 0:
            second_order = np.zeros((batch_size, 1), dtype=np.float32)
        else:
            embeddings = np.stack(embeddings, axis=1)  # [batch_size, field_num, embedding_dim]
            
            # FM second-order interaction: 0.5 * ((sum(x))^2 - sum(x^2))
            sum_square = np.sum(embeddings, axis=1) ** 2  # [batch_size, embedding_dim]
            square_sum = np.sum(embeddings ** 2, axis=1)   # [batch_size, embedding_dim]
            
            second_order = 0.5 * np.sum(sum_square - square_sum, axis=1, keepdims=True)  # [batch_size, 1]
        
        return first_order, second_order
    
    def dnn_layer(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """DNN layer computation - deep neural network"""
        
        batch_size = len(list(features.values())[0])
        
        # Concatenate all embedding vectors as DNN input
        dnn_inputs = []
        
        for field, ids in features.items():
            if field in self.embeddings:
                emb = self.get_embedding(field, ids)  # [batch_size, embedding_dim]
                dnn_inputs.append(emb.reshape(batch_size, -1))
        
        if len(dnn_inputs) == 0:
            return np.zeros((batch_size, 1), dtype=np.float32)
        
        # Concatenate all embedding vectors
        x = np.concatenate(dnn_inputs, axis=1)  # [batch_size, total_embedding_dim]
        
        # Forward propagation
        for i, (weight, bias) in enumerate(zip(self.dnn_weights, self.dnn_biases)):
            x = np.matmul(x, weight) + bias  # Linear transformation
            
            # Activation function
            if self.config.dnn_activation == 'relu':
                x = np.maximum(0, x)
            elif self.config.dnn_activation == 'sigmoid':
                x = 1 / (1 + np.exp(-x))
            
            # Dropout (skipped during inference)
        
        # Output layer
        output = np.matmul(x, self.output_weight) + self.output_bias  # [batch_size, 1]
        
        return output
    
    def forward(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """Forward propagation - complete DeepFM inference"""
        
        # FM layer computation
        fm_first_order, fm_second_order = self.fm_layer(features)
        
        # DNN layer computation
        dnn_output = self.dnn_layer(features)
        
        # Combine FM and DNN outputs
        final_output = fm_first_order + fm_second_order + dnn_output
        
        # Sigmoid activation for CTR prediction
        ctr_pred = 1 / (1 + np.exp(-final_output))
        
        return ctr_pred
    
    def predict_batch(self, batch_features: List[Dict[str, Any]]) -> np.ndarray:
        """Batch prediction"""
        
        if len(batch_features) == 0:
            return np.array([])
        
        # Convert to model input format
        model_features = defaultdict(list)
        
        for sample in batch_features:
            for field, value in sample.items():
                if field in self.config.vocab_sizes:
                    # Convert to ID (simplified processing)
                    if isinstance(value, str):
                        feature_id = hash(value) % self.config.vocab_sizes[field]
                    else:
                        feature_id = int(value) % self.config.vocab_sizes[field]
                    
                    model_features[field].append(feature_id)
        
        # Convert to numpy arrays
        for field in model_features:
            model_features[field] = np.array(model_features[field], dtype=np.int32)
        
        # Forward propagation
        predictions = self.forward(model_features)
        
        return predictions.flatten()


class DeepFMRankingSystem:
    """DeepFM coarse ranking system"""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize DeepFM coarse ranking system"""
        
        # Initialize model configuration
        self.config = DeepFMConfig(
            embedding_dim=32,
            dnn_hidden_units=[512, 256, 128, 64],
            dnn_dropout=0.3,
            learning_rate=0.001
        )
        
        # Initialize model
        self.model = DeepFMModel(self.config)
        
        # Load pre-trained model (if provided)
        if model_path:
            self.load_model(model_path)
        
        # Coarse ranking configuration
        self.ranking_config = {
            'target_count': 1000,        # Coarse ranking output item count
            'batch_size': 256,           # Batch processing size
            'score_threshold': 0.001,    # Minimum score threshold
            'diversity_control': True,   # Diversity control
            'category_quota': 200        # Maximum items per category
        }
        
        print("🎯 DeepFM coarse ranking system initialization completed")
    
    def extract_ranking_features(self, user_profile: Dict[str, Any],
                                item_info: Dict[str, Any],
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract coarse ranking features"""
        
        features = {
            # User features
            'user_id': user_profile.get('user_id', 'unknown'),
            'age_group': user_profile.get('age_group', 0),
            'gender': user_profile.get('gender', 0),
            'city_id': user_profile.get('city_id', 0),
            
            # Item features
            'item_id': item_info.get('item_id', 'unknown'),
            'category_id': item_info.get('category_id', 0),
            'brand_id': item_info.get('brand_id', 0),
            
            # Context features
            'device_type': context.get('device_type', 0),
            'hour': context.get('hour', 12),
            'day_of_week': context.get('day_of_week', 1)
        }
        
        return features
    
    def coarse_ranking(self, recalled_items: List[Tuple[str, float, str]],
                      user_profile: Dict[str, Any],
                      context: Dict[str, Any]) -> List[Tuple[str, float, Dict]]:
        """DeepFM coarse ranking main function"""
        
        print(f"🔄 Starting DeepFM coarse ranking, input {len(recalled_items)} items...")
        
        if len(recalled_items) == 0:
            return []
        
        # Prepare batch features
        batch_features = []
        item_infos = []
        
        for item_id, recall_score, sources in recalled_items:
            # Simulate item information
            item_info = {
                'item_id': item_id,
                'category_id': random.randint(1, 20),
                'brand_id': random.randint(1, 100),
                'price': random.uniform(10, 1000),
                'ctr_7d': random.uniform(0.001, 0.1),
                'cvr_7d': random.uniform(0.001, 0.05)
            }
            
            # Extract features
            features = self.extract_ranking_features(user_profile, item_info, context)
            batch_features.append(features)
            item_infos.append((item_id, recall_score, sources, item_info))
        
        # Batch prediction
        all_predictions = []
        batch_size = self.ranking_config['batch_size']
        
        for i in range(0, len(batch_features), batch_size):
            batch = batch_features[i:i + batch_size]
            predictions = self.model.predict_batch(batch)
            all_predictions.extend(predictions)
        
        print(f"   ✓ DeepFM model prediction completed, total {len(all_predictions)} prediction scores")
        
        # Combine recall score and CTR prediction score
        ranked_items = []
        
        for idx, (item_id, recall_score, sources, item_info) in enumerate(item_infos):
            if idx < len(all_predictions):
                ctr_pred = float(all_predictions[idx])
                
                # Coarse ranking score combination (recall score * 0.3 + CTR prediction * 0.7)
                coarse_score = recall_score * 0.3 + ctr_pred * 0.7
                
                # Add detailed information
                ranking_info = {
                    'recall_score': recall_score,
                    'ctr_prediction': ctr_pred,
                    'coarse_score': coarse_score,
                    'sources': sources,
                    'item_info': item_info
                }
                
                ranked_items.append((item_id, coarse_score, ranking_info))
        
        # Sort by coarse ranking score
        ranked_items.sort(key=lambda x: x[1], reverse=True)
        
        print(f"   ✓ Item sorting completed")
        
        # Apply diversity control
        if self.ranking_config['diversity_control']:
            final_items = self._apply_coarse_ranking_diversity(ranked_items)
        else:
            final_items = ranked_items
        
        # Truncate to target count
        target_count = self.ranking_config['target_count']
        final_items = final_items[:target_count]
        
        print(f"✅ DeepFM coarse ranking completed, output {len(final_items)} items")
        
        return final_items
    
    def _apply_coarse_ranking_diversity(self, ranked_items: List[Tuple[str, float, Dict]]) -> List[Tuple[str, float, Dict]]:
        """Apply coarse ranking diversity control"""
        
        final_items = []
        category_count = defaultdict(int)
        category_quota = self.ranking_config['category_quota']
        score_threshold = self.ranking_config['score_threshold']
        
        for item_id, score, info in ranked_items:
            # Score threshold filtering
            if score < score_threshold:
                continue
            
            # Category diversity control
            category_id = info['item_info'].get('category_id', 0)
            
            if category_count[category_id] < category_quota:
                final_items.append((item_id, score, info))
                category_count[category_id] += 1
                
                # Stop when target count is reached
                if len(final_items) >= self.ranking_config['target_count']:
                    break
        
        return final_items
    
    def load_model(self, model_path: str):
        """Load pre-trained model"""
        try:
            # Simplified implementation - should load real model weights
            print(f"   📁 Loading model: {model_path}")
        except Exception as e:
            print(f"   ❌ Model loading failed: {e}")
    
    def save_model(self, model_path: str):
        """Save model"""
        try:
            # Simplified implementation
            print(f"   💾 Saving model: {model_path}")
        except Exception as e:
            print(f"   ❌ Model saving failed: {e}")


def demo_deepfm_ranking():
    """DeepFM coarse ranking demo"""
    print("🎯 DeepFM Coarse Ranking System Demo")
    print("=" * 50)
    
    # Initialize coarse ranking system
    ranking_system = DeepFMRankingSystem()
    
    print("\n📊 1. Simulate recalled items data")
    # Simulate 10000 items recalled by RGA (simplified to 100 for demo)
    recalled_items = []
    for i in range(100):
        item_id = f"item_{i:04d}"
        recall_score = random.uniform(0.1, 10.0)
        sources = random.choice([
            'vector_similarity',
            'graph_attention', 
            'category_cf',
            'hot_items,new_items',
            'vector_similarity,hot_items'
        ])
        recalled_items.append((item_id, recall_score, sources))
    
    print(f"   ✓ Generated {len(recalled_items)} recalled items")
    
    print("\n👤 2. User Profile Preparation")
    user_profile = {
        'user_id': 'user_001',
        'age_group': 2,     # 26-35years old
        'gender': 1,        # Female
        'city_id': 1,       # Tier 1 city
        'preferred_categories': [1, 3, 5],
        'avg_order_value': 200.0
    }
    print(f"   ✓ User profile: Age group {user_profile['age_group']}, Tier {user_profile['city_id']} city")
    
    print("\n🌍 3. Context Information")
    context = {
        'device_type': 1,   # Mobile
        'hour': 14,         # 2 PM
        'day_of_week': 3,   # Wednesday
        'is_weekend': False
    }
    print(f"   ✓ Context: Device {context['device_type']}, Hour {context['hour']}, Day {context['day_of_week']+1}")
    
    print("\n🤖 4. DeepFM Coarse Ranking Processing")
    ranked_items = ranking_system.coarse_ranking(
        recalled_items=recalled_items,
        user_profile=user_profile,
        context=context
    )
    
    print(f"\n📈 5. Coarse Ranking Results Analysis")
    print(f"   Coarse ranking output item count: {len(ranked_items)}")
    
    # Score distribution statistics
    scores = [score for _, score, _ in ranked_items]
    print(f"   Score range: {min(scores):.4f} ~ {max(scores):.4f}")
    print(f"   Average score: {np.mean(scores):.4f}")
    
    # Category diversity statistics
    category_dist = defaultdict(int)
    for _, _, info in ranked_items:
        category_id = info['item_info'].get('category_id', 0)
        category_dist[category_id] += 1
    
    print(f"   Category distribution: {len(category_dist)} different categories")
    print(f"   Category balance: {min(category_dist.values())} ~ {max(category_dist.values())} items/category")
    
    print(f"\n🏆 Top 10 Coarse Ranking Results:")
    for i, (item_id, score, info) in enumerate(ranked_items[:10]):
        recall_score = info['recall_score']
        ctr_pred = info['ctr_prediction']
        category = info['item_info']['category_id']
        sources = info['sources']
        
        print(f"   {i+1}. {item_id} | Score:{score:.4f} | CTR:{ctr_pred:.4f} | Category:{category} | Source:{sources}")
    
    print(f"\n✅ DeepFM coarse ranking demo completed!")
    print("Key Features:")
    print("• Wide&Deep architecture combines linear features and deep features")
    print("• FM layer automatically learns feature crosses")
    print("• DNN layer learns high-order non-linear features")
    print("• Diversity control ensures category balance")
    print(f"• Successfully filtered recalled items from {len(recalled_items)} to {len(ranked_items)} ")


if __name__ == "__main__":
    demo_deepfm_ranking()