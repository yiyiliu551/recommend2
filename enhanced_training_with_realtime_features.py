#!/usr/bin/env python3
"""
Enhanced Training with Real-time Features Integration
Uses historical real-time features for training and online learning updates
Author: Yang Liu
"""

import numpy as np
import pandas as pd
import time
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, deque
from dataclasses import dataclass
import threading
import queue
import redis


@dataclass
class Trainingsample:
    """Training sample with real-time features"""
    user_id: str
    item_id: str
    timestamp: float
    label: int  # 0: no interaction, 1: click, 2: purchase
    
    # Static features
    user_age_group: int
    user_gender: int
    user_city_level: int
    item_category: int
    item_brand: int
    item_price: float
    
    # Historical real-time features (at sample timestamp)
    user_clicks_1h: int
    user_clicks_6h: int
    user_clicks_1d: int
    user_clicks_7d: int
    user_purchases_1h: int
    user_purchases_6h: int
    user_purchases_1d: int
    user_purchases_7d: int
    user_avg_price_1d: float
    user_session_length_1h: int
    
    # Item historical real-time features
    item_views_1h: int
    item_views_6h: int
    item_views_1d: int
    item_clicks_1h: int
    item_clicks_6h: int
    item_clicks_1d: int
    item_purchases_1h: int
    item_purchases_6h: int
    item_purchases_1d: int
    item_ctr_1h: float
    item_ctr_6h: float
    item_ctr_1d: float
    item_cvr_1d: float
    
    # Cross features
    user_item_category_affinity: float
    user_price_sensitivity: float
    item_popularity_vs_user_activity: float
    
    # Context features
    hour_of_day: int
    day_of_week: int
    is_weekend: bool
    season: int


class HistoricalRealtimeFeatureGenerator:
    """Generate training data with historical real-time features"""
    
    def __init__(self, data_start_date: str = "2024-01-01", data_end_date: str = "2024-08-31"):
        self.start_time = datetime.strptime(data_start_date, "%Y-%m-%d").timestamp()
        self.end_time = datetime.strptime(data_end_date, "%Y-%m-%d").timestamp()
        
        # Simulate user and item pools
        self.num_users = 10000
        self.num_items = 50000
        self.num_categories = 20
        self.num_brands = 500
        
        # Historical behavior storage
        self.user_behavior_history = defaultdict(list)  # {user_id: [(timestamp, item_id, action, price)]}
        self.item_interaction_history = defaultdict(list)  # {item_id: [(timestamp, user_id, action)]}
        
        # Generate base interactions
        self._generate_historical_interactions()
    
    def _generate_historical_interactions(self):
        """Generate 8 months of historical user interactions"""
        print("🔄 Generating 8 months of historical interactions...")
        
        total_interactions = 0
        current_time = self.start_time
        
        while current_time < self.end_time:
            # Generate interactions for this hour
            hour_interactions = np.random.poisson(1000)  # Average 1000 interactions per hour
            
            for _ in range(hour_interactions):
                user_id = f"user_{random.randint(1, self.num_users)}"
                item_id = f"item_{random.randint(1, self.num_items)}"
                
                # Action weights: view(70%), click(20%), purchase(10%)
                action = np.random.choice(['view', 'click', 'purchase'], p=[0.7, 0.2, 0.1])
                price = random.uniform(10, 1000)
                
                # Add to history
                interaction = (current_time, item_id, action, price)
                self.user_behavior_history[user_id].append(interaction)
                
                item_interaction = (current_time, user_id, action)
                self.item_interaction_history[item_id].append(item_interaction)
                
                total_interactions += 1
            
            current_time += 3600  # Move to next hour
            
            if total_interactions % 100000 == 0:
                progress = (current_time - self.start_time) / (self.end_time - self.start_time) * 100
                print(f"   Progress: {progress:.1f}% - {total_interactions:,} interactions generated")
        
        print(f"✅ Generated {total_interactions:,} historical interactions")
        print(f"   Users: {len(self.user_behavior_history)}")
        print(f"   Items: {len(self.item_interaction_history)}")
    
    def calculate_user_realtime_features(self, user_id: str, timestamp: float, 
                                       windows: List[int] = [3600, 21600, 86400, 604800]) -> Dict:
        """Calculate user's real-time features at given timestamp"""
        user_history = self.user_behavior_history.get(user_id, [])
        features = {}
        
        for window_seconds in windows:
            window_start = timestamp - window_seconds
            window_name = self._seconds_to_window_name(window_seconds)
            
            # Filter interactions in this window
            window_interactions = [
                (t, item, action, price) for t, item, action, price in user_history
                if window_start <= t < timestamp
            ]
            
            # Calculate features
            total_actions = len(window_interactions)
            clicks = sum(1 for _, _, action, _ in window_interactions if action == 'click')
            purchases = sum(1 for _, _, action, _ in window_interactions if action == 'purchase')
            avg_price = np.mean([price for _, _, _, price in window_interactions]) if window_interactions else 0
            unique_items = len(set(item for _, item, _, _ in window_interactions))
            
            features.update({
                f'user_total_actions_{window_name}': total_actions,
                f'user_clicks_{window_name}': clicks,
                f'user_purchases_{window_name}': purchases,
                f'user_avg_price_{window_name}': avg_price,
                f'user_unique_items_{window_name}': unique_items,
            })
        
        return features
    
    def calculate_item_realtime_features(self, item_id: str, timestamp: float,
                                       windows: List[int] = [3600, 21600, 86400]) -> Dict:
        """Calculate item's real-time features at given timestamp"""
        item_history = self.item_interaction_history.get(item_id, [])
        features = {}
        
        for window_seconds in windows:
            window_start = timestamp - window_seconds  
            window_name = self._seconds_to_window_name(window_seconds)
            
            # Filter interactions in this window
            window_interactions = [
                (t, user, action) for t, user, action in item_history
                if window_start <= t < timestamp
            ]
            
            # Calculate features
            views = sum(1 for _, _, action in window_interactions if action == 'view')
            clicks = sum(1 for _, _, action in window_interactions if action == 'click')
            purchases = sum(1 for _, _, action in window_interactions if action == 'purchase')
            unique_users = len(set(user for _, user, _ in window_interactions))
            
            ctr = clicks / max(views, 1)
            cvr = purchases / max(views, 1)
            
            features.update({
                f'item_views_{window_name}': views,
                f'item_clicks_{window_name}': clicks,
                f'item_purchases_{window_name}': purchases,
                f'item_unique_users_{window_name}': unique_users,
                f'item_ctr_{window_name}': ctr,
                f'item_cvr_{window_name}': cvr,
            })
        
        return features
    
    def _seconds_to_window_name(self, seconds: int) -> str:
        """Convert seconds to readable window name"""
        if seconds == 3600:
            return "1h"
        elif seconds == 21600:
            return "6h"
        elif seconds == 86400:
            return "1d"
        elif seconds == 604800:
            return "7d"
        else:
            return f"{seconds}s"
    
    def generate_training_samples(self, num_samples: int = 100000) -> List[TrainingSample]:
        """Generate training samples with historical real-time features"""
        print(f"🎯 Generating {num_samples:,} training samples with real-time features...")
        
        samples = []
        sample_timestamps = np.random.uniform(
            self.start_time + 86400*7,  # Start after 7 days to have history
            self.end_time - 86400,      # End before last day
            num_samples
        )
        
        for i, timestamp in enumerate(sample_timestamps):
            user_id = f"user_{random.randint(1, self.num_users)}"
            item_id = f"item_{random.randint(1, self.num_items)}"
            
            # Generate label (simulate real user behavior)
            # Higher chance of interaction if user was recently active
            user_recent_activity = len([
                1 for t, _, _, _ in self.user_behavior_history.get(user_id, [])
                if timestamp - 3600 <= t < timestamp
            ])
            
            # Higher chance if item was recently popular
            item_recent_popularity = len([
                1 for t, _, _ in self.item_interaction_history.get(item_id, [])
                if timestamp - 3600 <= t < timestamp
            ])
            
            interaction_prob = min(0.1 + user_recent_activity * 0.01 + item_recent_popularity * 0.001, 0.8)
            label = 1 if random.random() < interaction_prob else 0
            
            # Static features
            dt = datetime.fromtimestamp(timestamp)
            
            # Calculate real-time features
            user_features = self.calculate_user_realtime_features(user_id, timestamp)
            item_features = self.calculate_item_realtime_features(item_id, timestamp)
            
            # Cross features
            user_avg_price = user_features.get('user_avg_price_1d', 100)
            item_price = random.uniform(10, 1000)
            price_diff = abs(user_avg_price - item_price)
            
            sample = TrainingSample(
                user_id=user_id,
                item_id=item_id,
                timestamp=timestamp,
                label=label,
                
                # Static features
                user_age_group=random.randint(1, 6),
                user_gender=random.randint(0, 2),
                user_city_level=random.randint(1, 4),
                item_category=random.randint(1, 20),
                item_brand=random.randint(1, 500),
                item_price=item_price,
                
                # User real-time features
                user_clicks_1h=user_features.get('user_clicks_1h', 0),
                user_clicks_6h=user_features.get('user_clicks_6h', 0), 
                user_clicks_1d=user_features.get('user_clicks_1d', 0),
                user_clicks_7d=user_features.get('user_clicks_7d', 0),
                user_purchases_1h=user_features.get('user_purchases_1h', 0),
                user_purchases_6h=user_features.get('user_purchases_6h', 0),
                user_purchases_1d=user_features.get('user_purchases_1d', 0),
                user_purchases_7d=user_features.get('user_purchases_7d', 0),
                user_avg_price_1d=user_features.get('user_avg_price_1d', 100),
                user_session_length_1h=user_features.get('user_total_actions_1h', 0),
                
                # Item real-time features
                item_views_1h=item_features.get('item_views_1h', 0),
                item_views_6h=item_features.get('item_views_6h', 0),
                item_views_1d=item_features.get('item_views_1d', 0),
                item_clicks_1h=item_features.get('item_clicks_1h', 0),
                item_clicks_6h=item_features.get('item_clicks_6h', 0),
                item_clicks_1d=item_features.get('item_clicks_1d', 0),
                item_purchases_1h=item_features.get('item_purchases_1h', 0),
                item_purchases_6h=item_features.get('item_purchases_6h', 0),
                item_purchases_1d=item_features.get('item_purchases_1d', 0),
                item_ctr_1h=item_features.get('item_ctr_1h', 0),
                item_ctr_6h=item_features.get('item_ctr_6h', 0),
                item_ctr_1d=item_features.get('item_ctr_1d', 0),
                item_cvr_1d=item_features.get('item_cvr_1d', 0),
                
                # Cross features
                user_item_category_affinity=random.uniform(0, 1),
                user_price_sensitivity=min(price_diff / max(user_avg_price, 1), 2.0),
                item_popularity_vs_user_activity=user_features.get('user_total_actions_1d', 0) * 
                                               item_features.get('item_views_1d', 0) / 10000,
                
                # Context features
                hour_of_day=dt.hour,
                day_of_week=dt.weekday(),
                is_weekend=dt.weekday() >= 5,
                season=(dt.month - 1) // 3
            )
            
            samples.append(sample)
            
            if (i + 1) % 10000 == 0:
                progress = (i + 1) / num_samples * 100
                print(f"   Progress: {progress:.1f}% - {i+1:,} samples generated")
        
        print(f"✅ Generated {len(samples):,} training samples with real-time features")
        return samples


class OnlineLearningDeepFM:
    """DeepFM model with online learning capabilities"""
    
    def __init__(self, feature_dims: int = 50):
        self.feature_dims = feature_dims
        self.learning_rate = 0.001
        self.l2_reg = 0.0001
        
        # Initialize model weights
        self.weights = {
            'linear': np.random.normal(0, 0.01, feature_dims),
            'embeddings': np.random.normal(0, 0.01, (feature_dims, 32)),
            'dnn_w1': np.random.normal(0, 0.01, (feature_dims * 32, 256)),
            'dnn_w2': np.random.normal(0, 0.01, (256, 128)),
            'dnn_w3': np.random.normal(0, 0.01, (128, 1)),
            'bias': 0.0
        }
        
        # Online learning buffer
        self.online_buffer = deque(maxlen=10000)
        self.update_counter = 0
        
        print("🤖 Online Learning DeepFM initialized")
    
    def feature_engineering(self, sample: TrainingSample) -> np.ndarray:
        """Convert TrainingSample to feature vector"""
        features = np.array([
            # Static features (normalized)
            sample.user_age_group / 6.0,
            sample.user_gender / 2.0,
            sample.user_city_level / 4.0,
            sample.item_category / 20.0,
            sample.item_brand / 500.0,
            np.log1p(sample.item_price) / 10.0,
            
            # User real-time features (normalized)
            np.log1p(sample.user_clicks_1h) / 5.0,
            np.log1p(sample.user_clicks_6h) / 8.0,
            np.log1p(sample.user_clicks_1d) / 10.0,
            np.log1p(sample.user_clicks_7d) / 15.0,
            np.log1p(sample.user_purchases_1h) / 3.0,
            np.log1p(sample.user_purchases_6h) / 5.0,
            np.log1p(sample.user_purchases_1d) / 8.0,
            np.log1p(sample.user_purchases_7d) / 12.0,
            np.log1p(sample.user_avg_price_1d) / 10.0,
            np.log1p(sample.user_session_length_1h) / 8.0,
            
            # Item real-time features (normalized)
            np.log1p(sample.item_views_1h) / 10.0,
            np.log1p(sample.item_views_6h) / 12.0,
            np.log1p(sample.item_views_1d) / 15.0,
            np.log1p(sample.item_clicks_1h) / 8.0,
            np.log1p(sample.item_clicks_6h) / 10.0,
            np.log1p(sample.item_clicks_1d) / 12.0,
            np.log1p(sample.item_purchases_1h) / 5.0,
            np.log1p(sample.item_purchases_6h) / 8.0,
            np.log1p(sample.item_purchases_1d) / 10.0,
            sample.item_ctr_1h,
            sample.item_ctr_6h,
            sample.item_ctr_1d,
            sample.item_cvr_1d,
            
            # Cross features
            sample.user_item_category_affinity,
            min(sample.user_price_sensitivity, 2.0) / 2.0,
            min(sample.item_popularity_vs_user_activity, 100.0) / 100.0,
            
            # Context features
            sample.hour_of_day / 24.0,
            sample.day_of_week / 7.0,
            float(sample.is_weekend),
            sample.season / 4.0,
            
            # Time-based features
            np.sin(2 * np.pi * sample.hour_of_day / 24),  # Cyclical hour
            np.cos(2 * np.pi * sample.hour_of_day / 24),
            np.sin(2 * np.pi * sample.day_of_week / 7),   # Cyclical day
            np.cos(2 * np.pi * sample.day_of_week / 7),
            
            # Feature interactions
            sample.user_clicks_1h * sample.item_ctr_1h,
            sample.user_purchases_1d * sample.item_cvr_1d,
            sample.user_avg_price_1d * sample.item_price / 1000000,  # Price match
            
            # Trend features
            max(sample.user_clicks_1d - sample.user_clicks_1h * 24, 0) / 100.0,  # User activity trend
            max(sample.item_views_1d - sample.item_views_1h * 24, 0) / 1000.0,   # Item popularity trend
        ], dtype=np.float32)
        
        # Pad or truncate to fixed size
        if len(features) < self.feature_dims:
            features = np.pad(features, (0, self.feature_dims - len(features)), mode='constant')
        else:
            features = features[:self.feature_dims]
            
        return features
    
    def forward(self, features: np.ndarray) -> float:
        """Forward pass of DeepFM"""
        # Linear part
        linear_output = np.dot(features, self.weights['linear']) + self.weights['bias']
        
        # FM part (simplified)
        embeddings = np.dot(features.reshape(-1, 1), 
                           self.weights['embeddings'].reshape(1, -1)).flatten()
        fm_output = 0.5 * np.sum(embeddings ** 2)
        
        # DNN part
        dnn_input = embeddings
        h1 = np.maximum(0, np.dot(dnn_input, self.weights['dnn_w1']))  # ReLU
        h2 = np.maximum(0, np.dot(h1, self.weights['dnn_w2']))
        dnn_output = np.dot(h2, self.weights['dnn_w3']).item()
        
        # Combine
        final_output = linear_output + fm_output + dnn_output
        return 1.0 / (1.0 + np.exp(-final_output))  # Sigmoid
    
    def batch_train(self, samples: List[TrainingSample], epochs: int = 5):
        """Batch training with real-time features"""
        print(f"🚀 Training DeepFM with {len(samples):,} samples for {epochs} epochs...")
        
        best_auc = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            predictions = []
            labels = []
            
            # Shuffle samples
            random.shuffle(samples)
            
            for i, sample in enumerate(samples):
                features = self.feature_engineering(sample)
                pred = self.forward(features)
                
                # Calculate loss
                loss = -sample.label * np.log(pred + 1e-8) - (1 - sample.label) * np.log(1 - pred + 1e-8)
                epoch_loss += loss
                
                # Simple gradient descent (simplified)
                error = pred - sample.label
                lr = self.learning_rate / (1 + epoch * 0.1)  # Learning rate decay
                
                # Update weights (simplified)
                self.weights['linear'] -= lr * error * features
                self.weights['bias'] -= lr * error
                
                predictions.append(pred)
                labels.append(sample.label)
                
                if (i + 1) % 10000 == 0:
                    progress = (i + 1) / len(samples) * 100
                    avg_loss = epoch_loss / (i + 1)
                    print(f"   Epoch {epoch+1}, Progress: {progress:.1f}%, Avg Loss: {avg_loss:.4f}")
            
            # Calculate AUC
            auc = self.calculate_auc(predictions, labels)
            avg_loss = epoch_loss / len(samples)
            
            print(f"   Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, AUC: {auc:.4f}")
            
            if auc > best_auc:
                best_auc = auc
                print(f"   ✅ New best AUC: {best_auc:.4f}")
        
        print(f"🎯 Training completed! Best AUC: {best_auc:.4f}")
        return best_auc
    
    def online_update(self, new_sample: TrainingSample):
        """Online learning update with single sample"""
        self.online_buffer.append(new_sample)
        
        # Update every 100 samples
        if len(self.online_buffer) >= 100:
            features_batch = [self.feature_engineering(s) for s in self.online_buffer]
            labels_batch = [s.label for s in self.online_buffer]
            
            for features, label in zip(features_batch, labels_batch):
                pred = self.forward(features)
                error = pred - label
                lr = 0.0001  # Small learning rate for online updates
                
                # Update weights
                self.weights['linear'] -= lr * error * features
                self.weights['bias'] -= lr * error
            
            self.online_buffer.clear()
            self.update_counter += 1
            
            if self.update_counter % 10 == 0:
                print(f"📊 Online update #{self.update_counter} completed")
    
    def calculate_auc(self, predictions: List[float], labels: List[int]) -> float:
        """Calculate AUC metric"""
        if len(set(labels)) < 2:
            return 0.5
        
        # Simple AUC calculation
        pos_scores = [p for p, l in zip(predictions, labels) if l == 1]
        neg_scores = [p for p, l in zip(predictions, labels) if l == 0]
        
        if not pos_scores or not neg_scores:
            return 0.5
        
        auc_sum = 0
        for pos_score in pos_scores:
            for neg_score in neg_scores:
                if pos_score > neg_score:
                    auc_sum += 1
                elif pos_score == neg_score:
                    auc_sum += 0.5
        
        return auc_sum / (len(pos_scores) * len(neg_scores))


def main():
    """Main training pipeline with real-time features"""
    print("🚀 Enhanced Training with Real-time Features")
    print("=" * 80)
    
    # 1. Generate historical data and real-time features
    feature_generator = HistoricalRealtimeFeatureGenerator()
    
    # 2. Generate training samples
    training_samples = feature_generator.generate_training_samples(50000)
    
    # 3. Split train/validation
    split_idx = int(len(training_samples) * 0.8)
    train_samples = training_samples[:split_idx]
    val_samples = training_samples[split_idx:]
    
    print(f"\n📊 Dataset split:")
    print(f"   Training samples: {len(train_samples):,}")
    print(f"   Validation samples: {len(val_samples):,}")
    
    # 4. Initialize and train model
    model = OnlineLearningDeepFM(feature_dims=50)
    best_auc = model.batch_train(train_samples, epochs=3)
    
    # 5. Validate
    print(f"\n🔍 Validation...")
    val_predictions = []
    val_labels = []
    
    for sample in val_samples[:1000]:  # Sample validation
        features = model.feature_engineering(sample)
        pred = model.forward(features)
        val_predictions.append(pred)
        val_labels.append(sample.label)
    
    val_auc = model.calculate_auc(val_predictions, val_labels)
    print(f"   Validation AUC: {val_auc:.4f}")
    
    # 6. Demonstrate online learning
    print(f"\n🔄 Demonstrating Online Learning...")
    for i in range(10):
        # Simulate new real-time sample
        new_sample = training_samples[random.randint(0, len(training_samples)-1)]
        model.online_update(new_sample)
        time.sleep(0.1)
    
    print("\n✅ Enhanced training with real-time features completed!")
    print("Key improvements:")
    print("• Historical real-time features in training")
    print("• Multi-time window feature engineering")  
    print("• Cross features and temporal patterns")
    print("• Online learning capability")
    print(f"• Final training AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()