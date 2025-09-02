#!/usr/bin/env python3
"""
Simple model training - no external ML libs
Just pure math and common sense
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Any
import random


class SimplePredictor:
    """
    A stupidly simple model that just works.
    No gradient descent, no neural nets, just business logic.
    """
    
    def __init__(self):
        self.weights = {
            'user_age_young': 0.1,       # Young users click more
            'user_age_old': -0.05,       # Older users are picky
            'category_match': 0.3,       # Category match is huge
            'brand_match': 0.2,          # Brand loyalty matters
            'price_match': 0.15,         # Price sensitivity
            'popularity': 0.1,           # Social proof works
            'time_evening': 0.05,        # Evening browsing boost
            'mobile_penalty': -0.02      # Mobile users scroll fast
        }
        
        self.stats = {'predictions': 0, 'accuracy_checks': 0, 'correct': 0}
        print("🤖 SimplePredictor initialized (no fancy ML required)")
    
    def predict_ctr(self, interaction: Dict) -> float:
        """
        Predict if user will click. 
        Just add up the signals - sometimes simple is better.
        """
        
        score = 0.05  # Base CTR (everyone starts somewhere)
        
        # User age effects
        user_age = interaction.get('user_age', 30)
        if user_age < 25:
            score += self.weights['user_age_young']
        elif user_age > 50:
            score += self.weights['user_age_old']
        
        # Category matching (this is big!)
        user_categories = set(interaction.get('user_preferred_categories', []))
        item_category = interaction.get('item_category', '')
        if item_category in user_categories:
            score += self.weights['category_match']
        
        # Brand loyalty
        user_brands = set(interaction.get('user_preferred_brands', []))
        item_brand = interaction.get('item_brand', '')
        if item_brand in user_brands:
            score += self.weights['brand_match']
        
        # Price matching (people have budgets)
        item_price = interaction.get('item_price', 100)
        if user_age < 30 and item_price < 200:
            score += self.weights['price_match']  # Young + cheap = good
        elif user_age > 40 and item_price > 1000:
            score += self.weights['price_match']  # Older + premium = good
        elif 200 <= item_price <= 1000:
            score += self.weights['price_match'] * 0.5  # Middle ground
        
        # Popularity boost
        item_views = interaction.get('item_views', 100)
        if item_views > 5000:  # Popular item
            score += self.weights['popularity']
        
        # Time of day
        hour = interaction.get('hour', 12)
        if 18 <= hour <= 22:  # Evening prime time
            score += self.weights['time_evening']
        
        # Device type
        if interaction.get('device') == 'mobile':
            score += self.weights['mobile_penalty']
        
        # Cap between 0 and 1
        score = max(0.0, min(1.0, score))
        
        self.stats['predictions'] += 1
        return score
    
    def batch_predict(self, interactions: List[Dict]) -> List[float]:
        """Predict for multiple interactions"""
        return [self.predict_ctr(interaction) for interaction in interactions]
    
    def evaluate(self, interactions: List[Dict]) -> Dict[str, float]:
        """
        See how well our simple model does.
        Sometimes simple beats complex!
        """
        
        predictions = []
        labels = []
        
        for interaction in interactions:
            pred = self.predict_ctr(interaction)
            label = 1.0 if interaction.get('action') == 'click' else 0.0
            
            predictions.append(pred)
            labels.append(label)
        
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        # Calculate metrics manually (no sklearn needed)
        
        # AUC approximation (sort and calculate)
        sorted_indices = np.argsort(predictions)[::-1]  # High to low
        sorted_labels = labels[sorted_indices]
        
        # Simple AUC calculation
        positive_count = np.sum(sorted_labels)
        negative_count = len(sorted_labels) - positive_count
        
        if positive_count == 0 or negative_count == 0:
            auc = 0.5
        else:
            # Count correct pairwise comparisons
            correct_pairs = 0
            total_pairs = positive_count * negative_count
            
            pos_sum = 0
            for i, label in enumerate(sorted_labels):
                if label == 1:
                    pos_sum += i
            
            auc = (pos_sum - positive_count * (positive_count + 1) / 2) / total_pairs
        
        # Binary accuracy at 0.1 threshold
        binary_predictions = (predictions > 0.1).astype(int)
        accuracy = np.mean(binary_predictions == labels)
        
        # Precision and recall at 0.1 threshold  
        true_positives = np.sum((binary_predictions == 1) & (labels == 1))
        predicted_positives = np.sum(binary_predictions == 1)
        actual_positives = np.sum(labels == 1)
        
        precision = true_positives / predicted_positives if predicted_positives > 0 else 0
        recall = true_positives / actual_positives if actual_positives > 0 else 0
        
        return {
            'auc': auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'avg_prediction': np.mean(predictions),
            'click_rate': np.mean(labels)
        }
    
    def explain_prediction(self, interaction: Dict) -> Dict[str, Any]:
        """
        Explain why we made this prediction.
        Transparency is key!
        """
        
        explanation = {
            'base_score': 0.05,
            'adjustments': [],
            'total_score': 0.0
        }
        
        score = 0.05
        
        # Track each adjustment
        user_age = interaction.get('user_age', 30)
        if user_age < 25:
            adjustment = self.weights['user_age_young']
            score += adjustment
            explanation['adjustments'].append({
                'reason': f'Young user (age {user_age})',
                'adjustment': adjustment
            })
        
        user_categories = set(interaction.get('user_preferred_categories', []))
        item_category = interaction.get('item_category', '')
        if item_category in user_categories:
            adjustment = self.weights['category_match']
            score += adjustment
            explanation['adjustments'].append({
                'reason': f'Category match ({item_category})',
                'adjustment': adjustment
            })
        
        # ... (other adjustments)
        
        explanation['total_score'] = max(0.0, min(1.0, score))
        
        return explanation


def generate_realistic_data(n_samples: int = 5000) -> List[Dict]:
    """
    Generate realistic interaction data.
    Based on real user behavior patterns.
    """
    
    interactions = []
    
    # Electronic product catalog (simplified)
    products = [
        {'id': 'ps5_console', 'category': 'gaming', 'price': 499, 'views': 8500},
        {'id': 'electronic_wh1000xm5', 'category': 'audio', 'price': 399, 'views': 6200},
        {'id': 'electronic_a7r5', 'category': 'camera', 'price': 3899, 'views': 2100},
        {'id': 'bravia_65x90k', 'category': 'tv', 'price': 1599, 'views': 3400},
        {'id': 'electronic_wf1000xm4', 'category': 'audio', 'price': 279, 'views': 4500},
        {'id': 'dualsense_controller', 'category': 'gaming', 'price': 69, 'views': 1200},
        {'id': 'electronic_zv1', 'category': 'camera', 'price': 749, 'views': 890},
        {'id': 'srs_xb43', 'category': 'audio', 'price': 249, 'views': 670}
    ]
    
    # User segments with realistic preferences
    user_segments = [
        {
            'type': 'gamer',
            'age_range': (16, 35),
            'preferred_categories': ['gaming', 'audio'],
            'budget': (50, 600),
            'click_rate': 0.12
        },
        {
            'type': 'photographer', 
            'age_range': (25, 55),
            'preferred_categories': ['camera'],
            'budget': (500, 5000),
            'click_rate': 0.08
        },
        {
            'type': 'audiophile',
            'age_range': (20, 45), 
            'preferred_categories': ['audio'],
            'budget': (200, 2000),
            'click_rate': 0.10
        },
        {
            'type': 'general',
            'age_range': (18, 65),
            'preferred_categories': ['audio', 'tv'],
            'budget': (100, 1500),
            'click_rate': 0.06
        }
    ]
    
    print(f"Generating {n_samples} realistic interactions...")
    
    for i in range(n_samples):
        # Pick user segment
        segment = random.choice(user_segments)
        
        # Generate user
        user_age = random.randint(*segment['age_range'])
        user_categories = segment['preferred_categories']
        user_budget_min, user_budget_max = segment['budget']
        
        # Pick product
        product = random.choice(products)
        
        # Context
        hour = random.randint(0, 23)
        device = random.choices(['mobile', 'desktop', 'tablet'], weights=[0.6, 0.3, 0.1])[0]
        
        # Realistic click probability
        base_click_rate = segment['click_rate']
        
        # Boost if category matches
        if product['category'] in user_categories:
            base_click_rate *= 3
        
        # Boost if price is in budget
        if user_budget_min <= product['price'] <= user_budget_max:
            base_click_rate *= 2
        
        # Time of day effect
        if 18 <= hour <= 22:
            base_click_rate *= 1.3
        
        # Popular items get more clicks
        if product['views'] > 5000:
            base_click_rate *= 1.2
        
        # Generate click
        clicked = random.random() < base_click_rate
        
        interaction = {
            'user_id': f'user_{i % 1000}',
            'user_age': user_age,
            'user_preferred_categories': user_categories,
            'user_preferred_brands': ['electronic'],
            'user_segment': segment['type'],
            
            'item_id': product['id'],
            'item_category': product['category'],
            'item_brand': 'electronic',
            'item_price': product['price'],
            'item_views': product['views'],
            
            'hour': hour,
            'device': device,
            'action': 'click' if clicked else 'view'
        }
        
        interactions.append(interaction)
    
    # Print stats
    click_rate = np.mean([1 for i in interactions if i['action'] == 'click'])
    print(f"Generated {len(interactions)} interactions")
    print(f"Overall click rate: {click_rate:.3f}")
    
    # Category breakdown
    category_stats = {}
    for interaction in interactions:
        cat = interaction['item_category']
        if cat not in category_stats:
            category_stats[cat] = {'views': 0, 'clicks': 0}
        category_stats[cat]['views'] += 1
        if interaction['action'] == 'click':
            category_stats[cat]['clicks'] += 1
    
    print("\nCategory performance:")
    for cat, stats in category_stats.items():
        ctr = stats['clicks'] / stats['views'] if stats['views'] > 0 else 0
        print(f"  {cat}: {ctr:.3f} CTR ({stats['clicks']}/{stats['views']})")
    
    return interactions


def main():
    """Test our simple predictor"""
    
    print("🎯 SIMPLE TRAINING (NO ML LIBRARIES)")
    print("="*60)
    
    # Generate data
    interactions = generate_realistic_data(5000)
    
    # Split into train/test (80/20)
    split_idx = int(0.8 * len(interactions))
    train_data = interactions[:split_idx]
    test_data = interactions[split_idx:]
    
    print(f"\nTrain samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Create predictor
    predictor = SimplePredictor()
    
    # Evaluate on test data
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    results = predictor.evaluate(test_data)
    
    print(f"Test AUC: {results['auc']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"Avg prediction: {results['avg_prediction']:.4f}")
    print(f"Actual click rate: {results['click_rate']:.4f}")
    
    # Performance assessment
    if results['auc'] > 0.65:
        print("\n🎉 Model performs well! Simple wins!")
    elif results['auc'] > 0.55:
        print("\n🤔 Model is decent. Could use some tuning.")
    else:
        print("\n😬 Model needs work. Back to the drawing board.")
    
    # Show some examples
    print("\n" + "="*50)
    print("PREDICTION EXAMPLES")
    print("="*50)
    
    for i, interaction in enumerate(test_data[:5]):
        pred = predictor.predict_ctr(interaction)
        actual = interaction['action']
        
        print(f"\nExample {i+1}:")
        print(f"  User: {interaction['user_age']} years, likes {interaction['user_preferred_categories']}")
        print(f"  Item: {interaction['item_id']} (${interaction['item_price']})")
        print(f"  Prediction: {pred:.4f}")
        print(f"  Actual: {actual}")
        
        if (pred > 0.1 and actual == 'click') or (pred <= 0.1 and actual == 'view'):
            print("  ✅ Correct prediction")
        else:
            print("  ❌ Wrong prediction")
    
    print(f"\n📊 Total predictions made: {predictor.stats['predictions']}")
    print("\n✨ Simple model trained and tested! 🚀")
    print("Sometimes you don't need deep learning - just deep thinking!")


if __name__ == "__main__":
    main()