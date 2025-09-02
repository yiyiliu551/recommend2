"""
Incremental Model Updates for Real-time Recommendations
Author: Yang Liu
Description: Online learning and incremental model updates for production recommendation system
Experience: Built similar system at Weibo enabling real-time model adaptation for 400M users
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import threading
import queue
import pickle
import hashlib
from abc import ABC, abstractmethod
import logging


@dataclass
class ModelUpdate:
    """Represents an incremental model update"""
    update_type: str  # 'gradient', 'embedding', 'weight', 'bias'
    model_component: str  # 'user_embedding', 'item_embedding', 'interaction_layer'
    entity_id: str  # user_id or item_id
    update_vector: np.ndarray
    learning_rate: float
    timestamp: float
    confidence: float = 1.0


@dataclass
class FeedbackEvent:
    """User feedback event for model learning"""
    user_id: str
    item_id: str
    feedback_type: str  # 'implicit', 'explicit'
    action: str  # 'click', 'purchase', 'like', 'dislike', 'rating'
    value: float  # rating value, engagement time, etc.
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)


class OnlineLearningModel(ABC):
    """Abstract base class for online learning models"""
    
    @abstractmethod
    def predict(self, user_id: str, item_id: str, context: Dict[str, Any]) -> float:
        pass
    
    @abstractmethod
    def update(self, feedback: FeedbackEvent) -> List[ModelUpdate]:
        pass
    
    @abstractmethod
    def get_model_state(self) -> Dict[str, Any]:
        pass


class IncrementalMatrixFactorization(OnlineLearningModel):
    """
    Incremental Matrix Factorization for real-time collaborative filtering
    """
    
    def __init__(self, embedding_dim: int = 128, learning_rate: float = 0.01,
                 regularization: float = 0.001):
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.regularization = regularization
        
        # Model parameters
        self.user_embeddings = {}
        self.item_embeddings = {}
        self.user_bias = {}
        self.item_bias = {}
        self.global_bias = 3.5  # Average rating
        
        # Optimization state
        self.user_gradient_history = defaultdict(lambda: np.zeros(embedding_dim))
        self.item_gradient_history = defaultdict(lambda: np.zeros(embedding_dim))
        
        # Model statistics
        self.update_count = 0
        self.last_update_time = 0
        
        # Thread safety
        self.lock = threading.Lock()
    
    def _init_user_embedding(self, user_id: str):
        """Initialize embedding for new user"""
        if user_id not in self.user_embeddings:
            self.user_embeddings[user_id] = np.random.normal(0, 0.1, self.embedding_dim)
            self.user_bias[user_id] = 0.0
    
    def _init_item_embedding(self, item_id: str):
        """Initialize embedding for new item"""
        if item_id not in self.item_embeddings:
            self.item_embeddings[item_id] = np.random.normal(0, 0.1, self.embedding_dim)
            self.item_bias[item_id] = 0.0
    
    def predict(self, user_id: str, item_id: str, context: Dict[str, Any] = None) -> float:
        """Predict rating for user-item pair"""
        with self.lock:
            self._init_user_embedding(user_id)
            self._init_item_embedding(item_id)
            
            # Dot product + biases
            user_emb = self.user_embeddings[user_id]
            item_emb = self.item_embeddings[item_id]
            
            rating = (
                self.global_bias +
                self.user_bias[user_id] +
                self.item_bias[item_id] +
                np.dot(user_emb, item_emb)
            )
            
            return np.clip(rating, 1.0, 5.0)
    
    def update(self, feedback: FeedbackEvent) -> List[ModelUpdate]:
        """Update model based on feedback"""
        with self.lock:
            updates = []
            
            user_id = feedback.user_id
            item_id = feedback.item_id
            actual_rating = feedback.value
            
            # Initialize embeddings if needed
            self._init_user_embedding(user_id)
            self._init_item_embedding(item_id)
            
            # Current prediction
            predicted_rating = self.predict(user_id, item_id, feedback.context)
            error = actual_rating - predicted_rating
            
            # Adaptive learning rate based on confidence
            adaptive_lr = self.learning_rate * feedback.confidence if hasattr(feedback, 'confidence') else self.learning_rate
            
            # SGD updates
            user_emb = self.user_embeddings[user_id].copy()
            item_emb = self.item_embeddings[item_id].copy()
            
            # Gradient computation
            user_grad = error * item_emb - self.regularization * user_emb
            item_grad = error * user_emb - self.regularization * item_emb
            
            # AdaGrad-style adaptive learning
            self.user_gradient_history[user_id] += user_grad ** 2
            self.item_gradient_history[item_id] += item_grad ** 2
            
            user_adaptive_lr = adaptive_lr / (1e-8 + np.sqrt(self.user_gradient_history[user_id]))
            item_adaptive_lr = adaptive_lr / (1e-8 + np.sqrt(self.item_gradient_history[item_id]))
            
            # Update embeddings
            self.user_embeddings[user_id] += user_adaptive_lr * user_grad
            self.item_embeddings[item_id] += item_adaptive_lr * item_grad
            
            # Update biases
            self.user_bias[user_id] += adaptive_lr * (error - self.regularization * self.user_bias[user_id])
            self.item_bias[item_id] += adaptive_lr * (error - self.regularization * self.item_bias[item_id])
            
            # Create update records
            updates.append(ModelUpdate(
                update_type='embedding',
                model_component='user_embedding',
                entity_id=user_id,
                update_vector=self.user_embeddings[user_id] - user_emb,
                learning_rate=adaptive_lr,
                timestamp=feedback.timestamp
            ))
            
            updates.append(ModelUpdate(
                update_type='embedding', 
                model_component='item_embedding',
                entity_id=item_id,
                update_vector=self.item_embeddings[item_id] - item_emb,
                learning_rate=adaptive_lr,
                timestamp=feedback.timestamp
            ))
            
            self.update_count += 1
            self.last_update_time = feedback.timestamp
            
            return updates
    
    def get_model_state(self) -> Dict[str, Any]:
        """Get current model state for monitoring"""
        with self.lock:
            return {
                'model_type': 'IncrementalMatrixFactorization',
                'embedding_dim': self.embedding_dim,
                'n_users': len(self.user_embeddings),
                'n_items': len(self.item_embeddings),
                'update_count': self.update_count,
                'last_update_time': self.last_update_time,
                'global_bias': self.global_bias,
                'learning_rate': self.learning_rate
            }


class DeepIncrementalModel(OnlineLearningModel):
    """
    Incremental Deep Neural Network for recommendations
    """
    
    def __init__(self, user_vocab_size: int = 100000, item_vocab_size: int = 50000,
                 embedding_dim: int = 64, hidden_dims: List[int] = [128, 64]):
        self.user_vocab_size = user_vocab_size
        self.item_vocab_size = item_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        
        # Model parameters
        self.user_embeddings = {}
        self.item_embeddings = {}
        
        # Neural network weights (simplified)
        self.W1 = np.random.normal(0, 0.1, (embedding_dim * 2, hidden_dims[0]))
        self.b1 = np.zeros(hidden_dims[0])
        self.W2 = np.random.normal(0, 0.1, (hidden_dims[0], hidden_dims[1]))
        self.b2 = np.zeros(hidden_dims[1])
        self.W_out = np.random.normal(0, 0.1, (hidden_dims[1], 1))
        self.b_out = np.zeros(1)
        
        # Optimizer state (Adam-like)
        self.m_W1 = np.zeros_like(self.W1)
        self.v_W1 = np.zeros_like(self.W1)
        self.m_W2 = np.zeros_like(self.W2)
        self.v_W2 = np.zeros_like(self.W2)
        
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.learning_rate = 0.001
        
        self.update_count = 0
        self.lock = threading.Lock()
    
    def _get_user_embedding(self, user_id: str) -> np.ndarray:
        """Get or initialize user embedding"""
        if user_id not in self.user_embeddings:
            self.user_embeddings[user_id] = np.random.normal(0, 0.1, self.embedding_dim)
        return self.user_embeddings[user_id]
    
    def _get_item_embedding(self, item_id: str) -> np.ndarray:
        """Get or initialize item embedding"""
        if item_id not in self.item_embeddings:
            self.item_embeddings[item_id] = np.random.normal(0, 0.1, self.embedding_dim)
        return self.item_embeddings[item_id]
    
    def _forward(self, user_emb: np.ndarray, item_emb: np.ndarray) -> Tuple[float, Dict]:
        """Forward pass through network"""
        # Concatenate embeddings
        x = np.concatenate([user_emb, item_emb])
        
        # First hidden layer
        z1 = np.dot(x, self.W1) + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        
        # Second hidden layer
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = np.maximum(0, z2)  # ReLU
        
        # Output layer
        output = np.dot(a2, self.W_out) + self.b_out
        rating = 1 + 4 * (1 / (1 + np.exp(-output[0])))  # Sigmoid scaled to [1,5]
        
        # Store activations for backprop
        cache = {'x': x, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2, 'output': output}
        
        return rating, cache
    
    def predict(self, user_id: str, item_id: str, context: Dict[str, Any] = None) -> float:
        """Predict rating using deep model"""
        with self.lock:
            user_emb = self._get_user_embedding(user_id)
            item_emb = self._get_item_embedding(item_id)
            
            rating, _ = self._forward(user_emb, item_emb)
            return rating
    
    def update(self, feedback: FeedbackEvent) -> List[ModelUpdate]:
        """Update deep model with backpropagation"""
        with self.lock:
            updates = []
            
            user_id = feedback.user_id
            item_id = feedback.item_id
            target = feedback.value
            
            user_emb = self._get_user_embedding(user_id)
            item_emb = self._get_item_embedding(item_id)
            
            # Forward pass
            predicted, cache = self._forward(user_emb, item_emb)
            
            # Compute loss and gradients
            loss = (predicted - target) ** 2
            dloss_dpred = 2 * (predicted - target)
            
            # Backpropagation (simplified)
            # Output layer gradients
            dW_out = dloss_dpred * cache['a2'].reshape(-1, 1)
            db_out = np.array([dloss_dpred])
            
            # Hidden layer gradients
            da2 = dloss_dpred * self.W_out.flatten()
            dz2 = da2 * (cache['z2'] > 0).astype(float)  # ReLU derivative
            
            dW2 = np.outer(cache['a1'], dz2)
            db2 = dz2
            
            da1 = np.dot(dz2, self.W2.T)
            dz1 = da1 * (cache['z1'] > 0).astype(float)
            
            dW1 = np.outer(cache['x'], dz1)
            db1 = dz1
            
            # Adam optimizer updates
            self.update_count += 1
            t = self.update_count
            
            # Update W1
            self.m_W1 = self.beta1 * self.m_W1 + (1 - self.beta1) * dW1
            self.v_W1 = self.beta2 * self.v_W1 + (1 - self.beta2) * (dW1 ** 2)
            m_hat_W1 = self.m_W1 / (1 - self.beta1 ** t)
            v_hat_W1 = self.v_W1 / (1 - self.beta2 ** t)
            
            self.W1 -= self.learning_rate * m_hat_W1 / (np.sqrt(v_hat_W1) + self.epsilon)
            self.b1 -= self.learning_rate * db1
            
            # Update W2
            self.m_W2 = self.beta1 * self.m_W2 + (1 - self.beta1) * dW2
            self.v_W2 = self.beta2 * self.v_W2 + (1 - self.beta2) * (dW2 ** 2)
            m_hat_W2 = self.m_W2 / (1 - self.beta1 ** t)
            v_hat_W2 = self.v_W2 / (1 - self.beta2 ** t)
            
            self.W2 -= self.learning_rate * m_hat_W2 / (np.sqrt(v_hat_W2) + self.epsilon)
            self.b2 -= self.learning_rate * db2
            
            # Update output layer
            self.W_out -= self.learning_rate * dW_out
            self.b_out -= self.learning_rate * db_out
            
            # Update embeddings
            dx = np.dot(dz1, self.W1.T)
            duser_emb = dx[:self.embedding_dim]
            ditem_emb = dx[self.embedding_dim:]
            
            self.user_embeddings[user_id] -= self.learning_rate * duser_emb
            self.item_embeddings[item_id] -= self.learning_rate * ditem_emb
            
            # Create update records
            updates.append(ModelUpdate(
                update_type='weight',
                model_component='hidden_layer_1',
                entity_id='global',
                update_vector=dW1.flatten(),
                learning_rate=self.learning_rate,
                timestamp=feedback.timestamp
            ))
            
            return updates
    
    def get_model_state(self) -> Dict[str, Any]:
        """Get current model state"""
        with self.lock:
            return {
                'model_type': 'DeepIncrementalModel',
                'n_users': len(self.user_embeddings),
                'n_items': len(self.item_embeddings),
                'update_count': self.update_count,
                'embedding_dim': self.embedding_dim,
                'hidden_dims': self.hidden_dims,
                'learning_rate': self.learning_rate
            }


class IncrementalModelManager:
    """
    Manages incremental model updates and serving
    """
    
    def __init__(self, model: OnlineLearningModel):
        self.model = model
        self.feedback_queue = queue.Queue()
        self.update_buffer = deque(maxlen=10000)
        
        # Model serving
        self.model_versions = {}
        self.current_version = "v1.0"
        
        # Update scheduling
        self.batch_size = 50
        self.update_interval = 30  # seconds
        self.is_running = False
        self.update_thread = None
        
        # Monitoring
        self.metrics = {
            'total_updates': 0,
            'avg_update_latency': 0.0,
            'model_drift_score': 0.0,
            'prediction_errors': deque(maxlen=1000)
        }
        
        # A/B Testing
        self.ab_test_config = {
            'enabled': False,
            'traffic_split': 0.5,  # 50/50 split
            'champion_model': None,
            'challenger_model': None
        }
    
    def start_updates(self):
        """Start incremental update processing"""
        self.is_running = True
        self.update_thread = threading.Thread(target=self._update_worker)
        self.update_thread.start()
        logging.info("Incremental model updates started")
    
    def stop_updates(self):
        """Stop update processing"""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join()
        logging.info("Incremental model updates stopped")
    
    def add_feedback(self, feedback: FeedbackEvent):
        """Add user feedback for model learning"""
        self.feedback_queue.put(feedback)
    
    def predict(self, user_id: str, item_id: str, context: Dict[str, Any] = None) -> float:
        """Get prediction from current model"""
        # A/B testing logic
        if self.ab_test_config['enabled']:
            # Simple hash-based traffic splitting
            hash_val = int(hashlib.md5(f"{user_id}_{item_id}".encode()).hexdigest()[:8], 16)
            if (hash_val % 100) < (self.ab_test_config['traffic_split'] * 100):
                model = self.ab_test_config['champion_model']
            else:
                model = self.ab_test_config['challenger_model']
        else:
            model = self.model
        
        return model.predict(user_id, item_id, context)
    
    def _update_worker(self):
        """Background worker for processing updates"""
        feedback_batch = []
        last_update_time = time.time()
        
        while self.is_running:
            try:
                # Collect batch of feedback
                timeout = max(0.1, self.update_interval - (time.time() - last_update_time))
                
                try:
                    feedback = self.feedback_queue.get(timeout=timeout)
                    feedback_batch.append(feedback)
                except queue.Empty:
                    pass
                
                # Process batch when conditions are met
                should_update = (
                    len(feedback_batch) >= self.batch_size or
                    (len(feedback_batch) > 0 and time.time() - last_update_time > self.update_interval)
                )
                
                if should_update:
                    self._process_feedback_batch(feedback_batch)
                    feedback_batch.clear()
                    last_update_time = time.time()
                    
            except Exception as e:
                logging.error(f"Error in update worker: {e}")
    
    def _process_feedback_batch(self, feedback_batch: List[FeedbackEvent]):
        """Process a batch of feedback events"""
        update_start_time = time.time()
        
        all_updates = []
        prediction_errors = []
        
        for feedback in feedback_batch:
            # Get prediction before update for error calculation
            predicted = self.model.predict(feedback.user_id, feedback.item_id, feedback.context)
            error = abs(predicted - feedback.value)
            prediction_errors.append(error)
            
            # Update model
            updates = self.model.update(feedback)
            all_updates.extend(updates)
        
        # Update metrics
        self.metrics['total_updates'] += len(feedback_batch)
        self.metrics['avg_update_latency'] = time.time() - update_start_time
        self.metrics['prediction_errors'].extend(prediction_errors)
        
        # Store updates in buffer
        self.update_buffer.extend(all_updates)
        
        # Calculate model drift
        if len(prediction_errors) > 0:
            recent_error = np.mean(prediction_errors)
            historical_error = np.mean(list(self.metrics['prediction_errors']))
            self.metrics['model_drift_score'] = recent_error / max(historical_error, 0.001)
        
        logging.info(f"Processed {len(feedback_batch)} feedback events, "
                    f"generated {len(all_updates)} model updates, "
                    f"avg error: {np.mean(prediction_errors):.4f}")
    
    def create_model_checkpoint(self, version: str = None) -> str:
        """Create a model checkpoint for versioning"""
        if version is None:
            version = f"v{len(self.model_versions) + 1}.0"
        
        checkpoint = {
            'version': version,
            'timestamp': time.time(),
            'model_state': self.model.get_model_state(),
            'metrics': self.metrics.copy(),
            'update_count': self.metrics['total_updates']
        }
        
        self.model_versions[version] = checkpoint
        logging.info(f"Created model checkpoint: {version}")
        
        return version
    
    def rollback_to_checkpoint(self, version: str):
        """Rollback to a previous model checkpoint"""
        if version not in self.model_versions:
            raise ValueError(f"Checkpoint {version} not found")
        
        # In production, would restore model weights
        logging.warning(f"Rollback to {version} requested - implement model restoration")
        self.current_version = version
    
    def setup_ab_test(self, champion_model: OnlineLearningModel,
                      challenger_model: OnlineLearningModel,
                      traffic_split: float = 0.5):
        """Setup A/B test between two models"""
        self.ab_test_config = {
            'enabled': True,
            'traffic_split': traffic_split,
            'champion_model': champion_model,
            'challenger_model': challenger_model
        }
        logging.info(f"A/B test enabled with {traffic_split*100}% traffic split")
    
    def get_model_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive model performance report"""
        recent_errors = list(self.metrics['prediction_errors'])
        
        return {
            'model_info': self.model.get_model_state(),
            'performance_metrics': {
                'total_updates': self.metrics['total_updates'],
                'avg_update_latency': self.metrics['avg_update_latency'],
                'recent_mae': np.mean(recent_errors) if recent_errors else 0.0,
                'recent_rmse': np.sqrt(np.mean(np.array(recent_errors)**2)) if recent_errors else 0.0,
                'model_drift_score': self.metrics['model_drift_score']
            },
            'model_versions': list(self.model_versions.keys()),
            'current_version': self.current_version,
            'ab_test_status': self.ab_test_config['enabled'],
            'update_buffer_size': len(self.update_buffer),
            'timestamp': time.time()
        }


def demo_incremental_model_updates():
    """Demonstrate incremental model updates"""
    
    print("🔄 Incremental Model Updates Demo")
    print("="*50)
    
    # Initialize models
    print("Initializing incremental matrix factorization model...")
    mf_model = IncrementalMatrixFactorization(embedding_dim=64)
    manager = IncrementalModelManager(mf_model)
    
    # Start update processing
    manager.start_updates()
    
    # Generate sample feedback events
    print("Generating sample feedback events...")
    
    import random
    import time
    
    users = [f"user_{i}" for i in range(100)]
    items = [f"game_{i}" for i in range(50)]
    
    # Simulate feedback stream
    for i in range(500):
        feedback = FeedbackEvent(
            user_id=random.choice(users),
            item_id=random.choice(items),
            feedback_type='implicit',
            action='rating',
            value=random.uniform(1, 5),
            timestamp=time.time(),
            context={'platform': random.choice(['PS5', 'PS4'])}
        )
        
        manager.add_feedback(feedback)
        
        if i % 100 == 0:
            print(f"Generated {i} feedback events...")
    
    # Wait for updates to process
    time.sleep(5)
    
    # Test predictions
    print("\nTesting model predictions...")
    test_predictions = []
    
    for i in range(10):
        user = f"user_{i}"
        item = f"game_{i}"
        prediction = manager.predict(user, item)
        test_predictions.append((user, item, prediction))
        print(f"Prediction for {user}, {item}: {prediction:.3f}")
    
    # Create checkpoint
    print("\nCreating model checkpoint...")
    checkpoint_version = manager.create_model_checkpoint()
    
    # Generate performance report
    print("\nGenerating performance report...")
    report = manager.get_model_performance_report()
    
    print(f"Model Type: {report['model_info']['model_type']}")
    print(f"Total Updates: {report['performance_metrics']['total_updates']}")
    print(f"Recent MAE: {report['performance_metrics']['recent_mae']:.4f}")
    print(f"Model Drift Score: {report['performance_metrics']['model_drift_score']:.4f}")
    print(f"Current Version: {report['current_version']}")
    print(f"Available Checkpoints: {', '.join(report['model_versions'])}")
    
    # Demonstrate A/B testing
    print("\nSetting up A/B test...")
    challenger_model = DeepIncrementalModel()
    manager.setup_ab_test(mf_model, challenger_model, traffic_split=0.3)
    
    # Test A/B predictions
    ab_predictions = []
    for i in range(5):
        user = f"user_{i}"
        item = f"game_{i}"
        prediction = manager.predict(user, item)
        ab_predictions.append((user, item, prediction))
        print(f"A/B Prediction for {user}, {item}: {prediction:.3f}")
    
    # Stop updates
    manager.stop_updates()
    
    print("\n✅ Incremental model updates demo completed!")
    print("Key features demonstrated:")
    print("  • Real-time model learning from user feedback")
    print("  • Incremental weight updates with adaptive learning rates")
    print("  • Model versioning and checkpointing")
    print("  • A/B testing framework")
    print("  • Performance monitoring and drift detection")


if __name__ == "__main__":
    # Import time module
    import time
    demo_incremental_model_updates()