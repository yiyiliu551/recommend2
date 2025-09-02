"""
Enhanced Recommendation System Features - Core High Priority Components
1. Real-time Feature Updates
2. Cold Start Problem Handling
3. Business Rules Engine
4. Real-time Recommendation Evaluation

Author: Yang Liu
"""

import json
import time
import redis
import logging
import asyncio
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserType(Enum):
    NEW_USER = "new_user"           # < 7 days, < 5 interactions
    ACTIVE_USER = "active_user"     # regular users  
    POWER_USER = "power_user"       # > 50 interactions/month
    DORMANT_USER = "dormant_user"   # no activity > 30 days

class ItemType(Enum):
    NEW_ITEM = "new_item"           # < 30 days online
    POPULAR_ITEM = "popular_item"   # high interaction rate
    NICHE_ITEM = "niche_item"      # low interaction but high satisfaction
    SEASONAL_ITEM = "seasonal_item" # time-dependent popularity


# ===== 1. REAL-TIME FEATURE UPDATES =====

@dataclass
class UserBehaviorEvent:
    """Real-time user behavior event"""
    user_id: str
    item_id: str
    action_type: str  # view, click, add_cart, purchase, like, share
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)
    value: float = 1.0  # action weight


class RealTimeFeatureUpdater:
    """Real-time feature update mechanism for immediate recommendation impact"""
    
    def __init__(self, redis_client=None, window_size: int = 1000):
        # Production: self.redis = redis.Redis(**config)
        self.redis = redis_client or {}
        self.window_size = window_size
        
        # In-memory recent behavior cache (sliding window)
        self.user_recent_behaviors: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.item_recent_interactions: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        
        # Real-time feature decay factors
        self.decay_factors = {
            'view': 0.1,
            'click': 0.3,
            'add_cart': 0.5,
            'purchase': 1.0,
            'like': 0.7,
            'share': 0.8
        }
        
    async def update_user_behavior(self, event: UserBehaviorEvent):
        """Update user real-time features based on behavior event"""
        try:
            user_id = event.user_id
            item_id = event.item_id
            current_time = time.time()
            
            # 1. Add to sliding window
            self.user_recent_behaviors[user_id].append(event)
            self.item_recent_interactions[item_id].append(event)
            
            # 2. Update user interest vector in real-time
            await self._update_user_interest_vector(user_id, event)
            
            # 3. Update item interaction statistics  
            await self._update_item_stats(item_id, event)
            
            # 4. Update contextual features
            await self._update_contextual_features(user_id, event)
            
            logger.info(f"Updated real-time features for user {user_id}, item {item_id}, action {event.action_type}")
            
        except Exception as e:
            logger.error(f"Failed to update real-time features: {e}")
    
    async def _update_user_interest_vector(self, user_id: str, event: UserBehaviorEvent):
        """Update user interest vector based on item categories/attributes"""
        # Get item categories (mock implementation)
        item_categories = self._get_item_categories(event.item_id)
        action_weight = self.decay_factors.get(event.action_type, 0.1)
        
        # Update interest scores for categories
        interest_key = f"user_interest:{user_id}"
        current_interests = self._get_current_interests(interest_key)
        
        for category in item_categories:
            current_interests[category] = current_interests.get(category, 0) + action_weight
            # Apply time decay to prevent infinite growth
            current_interests[category] = min(current_interests[category], 10.0)
            
        # Save updated interests
        self.redis[interest_key] = json.dumps(current_interests)
        
    async def _update_item_stats(self, item_id: str, event: UserBehaviorEvent):
        """Update item real-time statistics"""
        stats_key = f"item_stats:{item_id}"
        current_stats = self._get_current_item_stats(stats_key)
        
        # Update interaction counts
        current_stats['total_interactions'] += 1
        current_stats[f'{event.action_type}_count'] = current_stats.get(f'{event.action_type}_count', 0) + 1
        current_stats['last_interaction'] = event.timestamp
        
        # Calculate real-time CTR, conversion rate etc
        if event.action_type == 'click':
            current_stats['clicks'] += 1
        elif event.action_type == 'purchase':
            current_stats['purchases'] += 1
            
        # Update CTR and CVR
        current_stats['ctr'] = current_stats['clicks'] / max(current_stats.get('views', 1), 1)
        current_stats['cvr'] = current_stats['purchases'] / max(current_stats['clicks'], 1)
        
        self.redis[stats_key] = json.dumps(current_stats)
        
    async def _update_contextual_features(self, user_id: str, event: UserBehaviorEvent):
        """Update contextual features like time, device, location patterns"""
        context_key = f"user_context:{user_id}"
        current_context = self._get_current_context(context_key)
        
        # Extract context from event
        hour_of_day = datetime.fromtimestamp(event.timestamp).hour
        day_of_week = datetime.fromtimestamp(event.timestamp).weekday()
        
        # Update time-based patterns
        current_context['active_hours'] = current_context.get('active_hours', {})
        current_context['active_hours'][str(hour_of_day)] = current_context['active_hours'].get(str(hour_of_day), 0) + 1
        
        current_context['active_days'] = current_context.get('active_days', {})
        current_context['active_days'][str(day_of_week)] = current_context['active_days'].get(str(day_of_week), 0) + 1
        
        # Update session-based features
        current_context['session_length'] = event.context.get('session_length', 0)
        current_context['device_type'] = event.context.get('device_type', 'unknown')
        
        self.redis[context_key] = json.dumps(current_context)
    
    def get_user_realtime_features(self, user_id: str) -> Dict[str, Any]:
        """Get real-time features for recommendation"""
        # Get recent behaviors (last 10 actions)
        recent_behaviors = list(self.user_recent_behaviors[user_id])[-10:]
        
        # Get current interests
        interest_key = f"user_interest:{user_id}"
        interests = self._get_current_interests(interest_key)
        
        # Get contextual features
        context_key = f"user_context:{user_id}"
        context = self._get_current_context(context_key)
        
        return {
            'recent_behaviors': [
                {
                    'item_id': b.item_id,
                    'action': b.action_type,
                    'timestamp': b.timestamp,
                    'value': b.value
                } for b in recent_behaviors
            ],
            'current_interests': interests,
            'context_patterns': context,
            'last_update': time.time()
        }
    
    def _get_item_categories(self, item_id: str) -> List[str]:
        """Get item categories (mock implementation)"""
        # In production, query from item database
        category_hash = hash(item_id) % 10
        return [f"category_{category_hash}", f"subcategory_{category_hash % 3}"]
    
    def _get_current_interests(self, key: str) -> Dict[str, float]:
        """Get current user interests with time decay"""
        if key not in self.redis:
            return {}
        try:
            return json.loads(self.redis[key])
        except:
            return {}
    
    def _get_current_item_stats(self, key: str) -> Dict[str, Any]:
        """Get current item statistics"""
        if key not in self.redis:
            return {'total_interactions': 0, 'clicks': 0, 'purchases': 0, 'views': 0}
        try:
            return json.loads(self.redis[key])
        except:
            return {'total_interactions': 0, 'clicks': 0, 'purchases': 0, 'views': 0}
    
    def _get_current_context(self, key: str) -> Dict[str, Any]:
        """Get current user context"""
        if key not in self.redis:
            return {}
        try:
            return json.loads(self.redis[key])
        except:
            return {}


# ===== 2. COLD START PROBLEM HANDLING =====

class ColdStartHandler:
    """Comprehensive cold start problem handler for new users and items"""
    
    def __init__(self, feature_updater: RealTimeFeatureUpdater):
        self.feature_updater = feature_updater
        
        # Cold start strategies configuration
        self.new_user_strategies = {
            'demographic_based': 0.3,      # 30% weight
            'popularity_based': 0.4,       # 40% weight  
            'diversity_exploration': 0.2,   # 20% weight
            'category_sampling': 0.1        # 10% weight
        }
        
        self.new_item_strategies = {
            'content_based': 0.5,          # 50% weight
            'category_popularity': 0.3,     # 30% weight
            'expert_curation': 0.2         # 20% weight
        }
        
    async def handle_new_user_recommendations(self, user_id: str, user_profile: Dict[str, Any], 
                                            target_count: int = 100) -> List[Dict[str, Any]]:
        """Generate recommendations for new users using multiple strategies"""
        
        logger.info(f"Generating cold start recommendations for new user {user_id}")
        
        all_recommendations = []
        
        # Strategy 1: Demographic-based recommendations
        demo_count = int(target_count * self.new_user_strategies['demographic_based'])
        demographic_recs = await self._demographic_based_recommendations(user_profile, demo_count)
        all_recommendations.extend([(rec, 'demographic', score) for rec, score in demographic_recs])
        
        # Strategy 2: Popularity-based recommendations  
        pop_count = int(target_count * self.new_user_strategies['popularity_based'])
        popularity_recs = await self._popularity_based_recommendations(pop_count)
        all_recommendations.extend([(rec, 'popularity', score) for rec, score in popularity_recs])
        
        # Strategy 3: Diversity exploration
        div_count = int(target_count * self.new_user_strategies['diversity_exploration'])
        diversity_recs = await self._diversity_exploration_recommendations(div_count)
        all_recommendations.extend([(rec, 'diversity', score) for rec, score in diversity_recs])
        
        # Strategy 4: Category sampling
        cat_count = target_count - len(all_recommendations)
        category_recs = await self._category_sampling_recommendations(cat_count)
        all_recommendations.extend([(rec, 'category', score) for rec, score in category_recs])
        
        # Merge and rank recommendations
        final_recs = self._merge_cold_start_recommendations(all_recommendations, target_count)
        
        logger.info(f"Generated {len(final_recs)} cold start recommendations for user {user_id}")
        return final_recs
    
    async def handle_new_item_recommendations(self, item_id: str, item_features: Dict[str, Any], 
                                            target_users: int = 100) -> List[str]:
        """Find target users for new items using multiple strategies"""
        
        logger.info(f"Finding target users for new item {item_id}")
        
        target_user_candidates = []
        
        # Strategy 1: Content-based user targeting
        content_count = int(target_users * self.new_item_strategies['content_based'])
        content_users = await self._content_based_user_targeting(item_features, content_count)
        target_user_candidates.extend(content_users)
        
        # Strategy 2: Category popularity targeting
        category_count = int(target_users * self.new_item_strategies['category_popularity'])
        category_users = await self._category_popularity_targeting(item_features, category_count)
        target_user_candidates.extend(category_users)
        
        # Strategy 3: Expert curation targeting
        expert_count = target_users - len(target_user_candidates)
        expert_users = await self._expert_curation_targeting(item_features, expert_count)
        target_user_candidates.extend(expert_users)
        
        # Remove duplicates and return
        unique_users = list(set(target_user_candidates))[:target_users]
        
        logger.info(f"Found {len(unique_users)} target users for new item {item_id}")
        return unique_users
    
    def classify_user_type(self, user_id: str, user_stats: Dict[str, Any]) -> UserType:
        """Classify user type for different cold start strategies"""
        
        registration_days = user_stats.get('registration_days', 0)
        total_interactions = user_stats.get('total_interactions', 0)
        last_activity_days = user_stats.get('last_activity_days', 0)
        monthly_interactions = user_stats.get('monthly_interactions', 0)
        
        # New user detection
        if registration_days < 7 and total_interactions < 5:
            return UserType.NEW_USER
            
        # Dormant user detection  
        if last_activity_days > 30:
            return UserType.DORMANT_USER
            
        # Power user detection
        if monthly_interactions > 50:
            return UserType.POWER_USER
            
        return UserType.ACTIVE_USER
    
    def classify_item_type(self, item_id: str, item_stats: Dict[str, Any]) -> ItemType:
        """Classify item type for different cold start strategies"""
        
        days_online = item_stats.get('days_online', 0)
        total_interactions = item_stats.get('total_interactions', 0)
        interaction_rate = item_stats.get('daily_interaction_rate', 0)
        satisfaction_score = item_stats.get('satisfaction_score', 0)
        
        # New item detection
        if days_online < 30:
            return ItemType.NEW_ITEM
            
        # Popular item detection
        if interaction_rate > 100:  # > 100 interactions per day
            return ItemType.POPULAR_ITEM
            
        # Niche item detection (low interaction but high satisfaction)
        if interaction_rate < 10 and satisfaction_score > 4.0:
            return ItemType.NICHE_ITEM
            
        return ItemType.POPULAR_ITEM
    
    # Private methods for different recommendation strategies
    async def _demographic_based_recommendations(self, user_profile: Dict, count: int) -> List[Tuple[str, float]]:
        """Recommend based on demographic similarity"""
        age_group = user_profile.get('age_group', '25-34')
        gender = user_profile.get('gender', 'unknown')
        city = user_profile.get('city', 'unknown')
        
        # Mock: find items popular among similar demographics
        recommendations = []
        for i in range(count):
            item_id = f"demo_item_{age_group}_{gender}_{i}"
            score = 0.8 - i * 0.01  # Decreasing scores
            recommendations.append((item_id, score))
            
        return recommendations
    
    async def _popularity_based_recommendations(self, count: int) -> List[Tuple[str, float]]:
        """Recommend globally popular items"""
        recommendations = []
        for i in range(count):
            item_id = f"popular_item_{i}"
            score = 0.9 - i * 0.01  # Decreasing popularity scores
            recommendations.append((item_id, score))
            
        return recommendations
    
    async def _diversity_exploration_recommendations(self, count: int) -> List[Tuple[str, float]]:
        """Recommend diverse items from different categories"""
        categories = ['electronics', 'books', 'clothing', 'home', 'sports', 'beauty']
        recommendations = []
        
        items_per_category = max(1, count // len(categories))
        
        for cat_idx, category in enumerate(categories):
            for i in range(items_per_category):
                if len(recommendations) >= count:
                    break
                item_id = f"diverse_{category}_{i}"
                score = 0.7 - len(recommendations) * 0.005  # Slight decrease
                recommendations.append((item_id, score))
                
        return recommendations[:count]
    
    async def _category_sampling_recommendations(self, count: int) -> List[Tuple[str, float]]:
        """Sample items from various categories"""
        recommendations = []
        for i in range(count):
            category = f"category_{i % 20}"  # 20 different categories
            item_id = f"sample_{category}_{i}"
            score = 0.6 - i * 0.003
            recommendations.append((item_id, score))
            
        return recommendations
    
    async def _content_based_user_targeting(self, item_features: Dict, count: int) -> List[str]:
        """Find users based on item content similarity"""
        category = item_features.get('category', 'electronics')
        brand = item_features.get('brand', 'unknown')
        price_range = item_features.get('price_range', 'medium')
        
        # Mock: find users who like similar items
        target_users = []
        for i in range(count):
            user_id = f"user_likes_{category}_{brand}_{i}"
            target_users.append(user_id)
            
        return target_users
    
    async def _category_popularity_targeting(self, item_features: Dict, count: int) -> List[str]:
        """Target users who are active in the item's category"""
        category = item_features.get('category', 'electronics')
        
        # Mock: users active in this category
        target_users = []
        for i in range(count):
            user_id = f"active_in_{category}_{i}"
            target_users.append(user_id)
            
        return target_users
    
    async def _expert_curation_targeting(self, item_features: Dict, count: int) -> List[str]:
        """Target early adopters and trendsetters"""
        # Mock: expert users who try new products
        target_users = []
        for i in range(count):
            user_id = f"expert_user_{i}"
            target_users.append(user_id)
            
        return target_users
    
    def _merge_cold_start_recommendations(self, all_recs: List[Tuple], target_count: int) -> List[Dict]:
        """Merge recommendations from different strategies"""
        # Sort by score and remove duplicates
        unique_recs = {}
        for item_id, strategy, score in all_recs:
            if item_id not in unique_recs or unique_recs[item_id]['score'] < score:
                unique_recs[item_id] = {
                    'item_id': item_id,
                    'score': score,
                    'strategy': strategy,
                    'cold_start': True
                }
        
        # Sort by score and return top recommendations
        sorted_recs = sorted(unique_recs.values(), key=lambda x: x['score'], reverse=True)
        return sorted_recs[:target_count]


# ===== 3. BUSINESS RULES ENGINE =====

@dataclass
class BusinessRule:
    """Business rule definition"""
    rule_id: str
    name: str
    condition: str  # JSON condition
    action: str     # JSON action
    priority: int   # Higher number = higher priority
    active: bool = True
    created_time: float = field(default_factory=time.time)


class BusinessRulesEngine:
    """Flexible business rules engine for dynamic recommendation control"""
    
    def __init__(self):
        self.rules: Dict[str, BusinessRule] = {}
        self.rule_stats: Dict[str, Dict] = defaultdict(lambda: {'applied': 0, 'blocked': 0})
        
        # Initialize with common business rules
        self._initialize_default_rules()
        
    def _initialize_default_rules(self):
        """Initialize common business rules"""
        
        # Rule 1: Filter out of stock items
        self.add_rule(BusinessRule(
            rule_id="filter_out_of_stock",
            name="Filter Out of Stock Items",
            condition=json.dumps({"item_stock": {"$lte": 0}}),
            action=json.dumps({"action": "exclude"}),
            priority=100
        ))
        
        # Rule 2: Boost promoted items
        self.add_rule(BusinessRule(
            rule_id="boost_promoted",
            name="Boost Promoted Items",
            condition=json.dumps({"item_tags": {"$contains": "promoted"}}),
            action=json.dumps({"action": "boost", "factor": 1.5}),
            priority=80
        ))
        
        # Rule 3: Regional filtering
        self.add_rule(BusinessRule(
            rule_id="regional_filter",
            name="Regional Availability Filter",
            condition=json.dumps({"item_regions": {"$not_contains": "user_region"}}),
            action=json.dumps({"action": "exclude"}),
            priority=90
        ))
        
        # Rule 4: Age restriction
        self.add_rule(BusinessRule(
            rule_id="age_restriction",
            name="Age Restricted Items",
            condition=json.dumps({"item_age_limit": {"$gt": "user_age"}}),
            action=json.dumps({"action": "exclude"}),
            priority=95
        ))
        
        # Rule 5: Personalized blacklist
        self.add_rule(BusinessRule(
            rule_id="user_blacklist",
            name="User Blacklisted Items",
            condition=json.dumps({"item_id": {"$in": "user_blacklist"}}),
            action=json.dumps({"action": "exclude"}),
            priority=85
        ))
        
        # Rule 6: Time-based promotions
        self.add_rule(BusinessRule(
            rule_id="time_promotion",
            name="Time-based Promotions",
            condition=json.dumps({"current_hour": {"$between": [20, 23]}, "item_category": "food"}),
            action=json.dumps({"action": "boost", "factor": 1.3}),
            priority=60
        ))
        
    def add_rule(self, rule: BusinessRule):
        """Add or update a business rule"""
        self.rules[rule.rule_id] = rule
        logger.info(f"Added business rule: {rule.name}")
        
    def remove_rule(self, rule_id: str):
        """Remove a business rule"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Removed business rule: {rule_id}")
            
    def toggle_rule(self, rule_id: str, active: bool):
        """Enable/disable a business rule"""
        if rule_id in self.rules:
            self.rules[rule_id].active = active
            status = "enabled" if active else "disabled"
            logger.info(f"Business rule {rule_id} {status}")
    
    async def apply_business_rules(self, user_id: str, recommendations: List[Dict], 
                                 user_context: Dict) -> List[Dict]:
        """Apply business rules to filter and adjust recommendations"""
        
        logger.info(f"Applying business rules for user {user_id} on {len(recommendations)} items")
        
        # Sort rules by priority (higher first)
        active_rules = [rule for rule in self.rules.values() if rule.active]
        sorted_rules = sorted(active_rules, key=lambda x: x.priority, reverse=True)
        
        filtered_recommendations = []
        
        for rec in recommendations:
            should_include = True
            boost_factor = 1.0
            applied_rules = []
            
            # Apply each rule
            for rule in sorted_rules:
                try:
                    condition_met = self._evaluate_condition(rule.condition, rec, user_context)
                    
                    if condition_met:
                        action = json.loads(rule.action)
                        applied_rules.append(rule.rule_id)
                        
                        if action['action'] == 'exclude':
                            should_include = False
                            self.rule_stats[rule.rule_id]['blocked'] += 1
                            break  # No need to check other rules
                            
                        elif action['action'] == 'boost':
                            boost_factor *= action.get('factor', 1.0)
                            self.rule_stats[rule.rule_id]['applied'] += 1
                            
                        elif action['action'] == 'penalty':
                            boost_factor *= action.get('factor', 0.8)
                            self.rule_stats[rule.rule_id]['applied'] += 1
                            
                except Exception as e:
                    logger.warning(f"Error applying rule {rule.rule_id}: {e}")
                    continue
            
            # Include item if not excluded
            if should_include:
                # Apply boost factor to score
                adjusted_rec = rec.copy()
                original_score = adjusted_rec.get('score', 0.5)
                adjusted_rec['score'] = original_score * boost_factor
                adjusted_rec['boost_factor'] = boost_factor
                adjusted_rec['applied_rules'] = applied_rules
                
                filtered_recommendations.append(adjusted_rec)
        
        logger.info(f"Business rules filtered {len(recommendations)} -> {len(filtered_recommendations)} items")
        
        # Re-sort by adjusted scores
        filtered_recommendations.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return filtered_recommendations
    
    def _evaluate_condition(self, condition_json: str, item: Dict, context: Dict) -> bool:
        """Evaluate if an item meets the rule condition"""
        try:
            condition = json.loads(condition_json)
            
            # Build evaluation context
            eval_context = {
                **item,  # Item properties
                **context,  # User context
                'current_hour': datetime.now().hour,
                'current_day': datetime.now().weekday(),
                'user_region': context.get('user_region', 'unknown'),
                'user_age': context.get('user_age', 25),
                'user_blacklist': context.get('user_blacklist', [])
            }
            
            return self._evaluate_condition_recursive(condition, eval_context)
            
        except Exception as e:
            logger.warning(f"Error evaluating condition: {e}")
            return False
    
    def _evaluate_condition_recursive(self, condition: Dict, context: Dict) -> bool:
        """Recursively evaluate complex conditions"""
        
        for field, criteria in condition.items():
            field_value = context.get(field)
            
            if isinstance(criteria, dict):
                for operator, expected in criteria.items():
                    
                    if operator == "$gt":
                        if not (field_value is not None and field_value > expected):
                            return False
                            
                    elif operator == "$gte":
                        if not (field_value is not None and field_value >= expected):
                            return False
                            
                    elif operator == "$lt":
                        if not (field_value is not None and field_value < expected):
                            return False
                            
                    elif operator == "$lte":
                        if not (field_value is not None and field_value <= expected):
                            return False
                            
                    elif operator == "$eq":
                        if field_value != expected:
                            return False
                            
                    elif operator == "$ne":
                        if field_value == expected:
                            return False
                            
                    elif operator == "$in":
                        if field_value not in expected:
                            return False
                            
                    elif operator == "$not_in":
                        if field_value in expected:
                            return False
                            
                    elif operator == "$contains":
                        if isinstance(field_value, list):
                            if expected not in field_value:
                                return False
                        else:
                            if expected not in str(field_value):
                                return False
                                
                    elif operator == "$not_contains":
                        if isinstance(field_value, list):
                            if expected in field_value:
                                return False
                        else:
                            if expected in str(field_value):
                                return False
                                
                    elif operator == "$between":
                        if not (isinstance(expected, list) and len(expected) == 2):
                            continue
                        if not (field_value is not None and expected[0] <= field_value <= expected[1]):
                            return False
                            
            else:
                # Simple equality check
                if field_value != criteria:
                    return False
                    
        return True
    
    def get_rule_statistics(self) -> Dict[str, Dict]:
        """Get business rule application statistics"""
        stats = {}
        for rule_id, rule in self.rules.items():
            rule_stats = self.rule_stats[rule_id]
            total_applied = rule_stats['applied'] + rule_stats['blocked']
            
            stats[rule_id] = {
                'name': rule.name,
                'active': rule.active,
                'priority': rule.priority,
                'total_applied': total_applied,
                'items_blocked': rule_stats['blocked'],
                'items_boosted': rule_stats['applied'],
                'application_rate': total_applied / max(1, total_applied) * 100
            }
            
        return stats


# ===== 4. REAL-TIME RECOMMENDATION EVALUATION =====

@dataclass
class RecommendationFeedback:
    """User feedback on recommendation"""
    user_id: str
    item_id: str
    feedback_type: str  # click, view, purchase, like, dislike, not_relevant
    timestamp: float
    value: float = 1.0
    context: Dict[str, Any] = field(default_factory=dict)


class RecommendationEvaluator:
    """Real-time recommendation performance evaluation and optimization"""
    
    def __init__(self, feature_updater: RealTimeFeatureUpdater):
        self.feature_updater = feature_updater
        
        # Performance metrics storage
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.user_feedback_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Evaluation thresholds
        self.performance_thresholds = {
            'ctr_threshold': 0.05,      # 5% click-through rate
            'conversion_threshold': 0.02, # 2% conversion rate
            'satisfaction_threshold': 3.5, # 3.5/5 satisfaction score
        }
        
        # Algorithm weights (can be dynamically adjusted)
        self.algorithm_weights = {
            'recall_vector_similarity': 0.3,
            'recall_graph_attention': 0.25,
            'recall_category_cf': 0.2,
            'recall_hot_items': 0.15,
            'recall_random_explore': 0.1
        }
        
    async def track_recommendation_performance(self, user_id: str, 
                                             recommendations: List[Dict],
                                             session_id: str = None):
        """Track recommendation performance metrics"""
        
        session_id = session_id or f"{user_id}_{int(time.time())}"
        timestamp = time.time()
        
        # Store recommendation session for later evaluation
        session_data = {
            'user_id': user_id,
            'session_id': session_id,
            'recommendations': recommendations,
            'timestamp': timestamp,
            'metrics': {
                'total_recommended': len(recommendations),
                'avg_score': np.mean([r.get('score', 0) for r in recommendations]),
                'diversity_score': self._calculate_diversity_score(recommendations),
                'novelty_score': await self._calculate_novelty_score(user_id, recommendations)
            }
        }
        
        # Store in metrics history
        self.metrics_history[f"session_{session_id}"] = session_data
        
        logger.info(f"Tracked recommendation session {session_id} for user {user_id}")
        
    async def record_user_feedback(self, feedback: RecommendationFeedback):
        """Record user feedback and update performance metrics"""
        
        user_id = feedback.user_id
        
        # Add to user feedback history
        self.user_feedback_history[user_id].append(feedback)
        
        # Update real-time metrics
        await self._update_performance_metrics(feedback)
        
        # Trigger real-time feature updates
        behavior_event = UserBehaviorEvent(
            user_id=feedback.user_id,
            item_id=feedback.item_id,
            action_type=feedback.feedback_type,
            timestamp=feedback.timestamp,
            value=feedback.value,
            context=feedback.context
        )
        
        await self.feature_updater.update_user_behavior(behavior_event)
        
        # Check if algorithm weights need adjustment
        await self._check_algorithm_performance()
        
        logger.info(f"Recorded feedback: {feedback.feedback_type} for item {feedback.item_id} by user {feedback.user_id}")
        
    async def _update_performance_metrics(self, feedback: RecommendationFeedback):
        """Update system performance metrics based on feedback"""
        
        metric_key = f"performance_metrics"
        current_metrics = self._get_current_performance_metrics()
        
        # Update overall metrics
        current_metrics['total_feedback'] += 1
        current_metrics[f'{feedback.feedback_type}_count'] = current_metrics.get(f'{feedback.feedback_type}_count', 0) + 1
        
        # Calculate rates
        total_interactions = current_metrics['total_feedback']
        current_metrics['ctr'] = current_metrics.get('click_count', 0) / total_interactions
        current_metrics['conversion_rate'] = current_metrics.get('purchase_count', 0) / total_interactions
        current_metrics['satisfaction_rate'] = current_metrics.get('like_count', 0) / max(1, current_metrics.get('dislike_count', 0) + current_metrics.get('like_count', 0))
        
        # Store updated metrics
        self.metrics_history[metric_key] = current_metrics
        
    async def _check_algorithm_performance(self):
        """Check if algorithm weights need adjustment based on performance"""
        
        current_metrics = self._get_current_performance_metrics()
        
        # Check if performance is below thresholds
        ctr = current_metrics.get('ctr', 0)
        conversion_rate = current_metrics.get('conversion_rate', 0)
        satisfaction_rate = current_metrics.get('satisfaction_rate', 0)
        
        needs_adjustment = False
        
        if ctr < self.performance_thresholds['ctr_threshold']:
            logger.warning(f"CTR ({ctr:.3f}) below threshold ({self.performance_thresholds['ctr_threshold']})")
            needs_adjustment = True
            
        if conversion_rate < self.performance_thresholds['conversion_threshold']:
            logger.warning(f"Conversion rate ({conversion_rate:.3f}) below threshold ({self.performance_thresholds['conversion_threshold']})")
            needs_adjustment = True
            
        if satisfaction_rate < self.performance_thresholds['satisfaction_threshold']:
            logger.warning(f"Satisfaction rate ({satisfaction_rate:.3f}) below threshold ({self.performance_thresholds['satisfaction_threshold']})")
            needs_adjustment = True
            
        if needs_adjustment:
            await self._auto_adjust_algorithm_weights()
            
    async def _auto_adjust_algorithm_weights(self):
        """Automatically adjust algorithm weights based on performance data"""
        
        logger.info("Auto-adjusting algorithm weights based on performance")
        
        # Simple adjustment strategy: reduce poorly performing components
        # In production, this would use more sophisticated ML-based optimization
        
        performance_by_source = await self._analyze_performance_by_recall_source()
        
        # Adjust weights based on performance
        total_weight = 0
        new_weights = {}
        
        for source, performance in performance_by_source.items():
            if source in self.algorithm_weights:
                # Boost high-performing sources, reduce low-performing ones
                adjustment_factor = min(2.0, max(0.5, performance['effectiveness']))
                new_weight = self.algorithm_weights[source] * adjustment_factor
                new_weights[source] = new_weight
                total_weight += new_weight
                
        # Normalize weights to sum to 1.0
        if total_weight > 0:
            for source in new_weights:
                new_weights[source] /= total_weight
                
            # Update weights if change is significant
            significant_change = any(
                abs(new_weights[s] - self.algorithm_weights[s]) > 0.05 
                for s in new_weights
            )
            
            if significant_change:
                old_weights = self.algorithm_weights.copy()
                self.algorithm_weights.update(new_weights)
                
                logger.info("Algorithm weights updated:")
                for source in new_weights:
                    logger.info(f"  {source}: {old_weights.get(source, 0):.3f} -> {new_weights[source]:.3f}")
                    
    async def _analyze_performance_by_recall_source(self) -> Dict[str, Dict]:
        """Analyze performance metrics by recall source"""
        
        # Mock implementation - in production, analyze actual performance data
        performance_by_source = {
            'recall_vector_similarity': {'effectiveness': 0.8, 'ctr': 0.06, 'conversion': 0.025},
            'recall_graph_attention': {'effectiveness': 0.9, 'ctr': 0.07, 'conversion': 0.03},
            'recall_category_cf': {'effectiveness': 0.7, 'ctr': 0.04, 'conversion': 0.015},
            'recall_hot_items': {'effectiveness': 0.6, 'ctr': 0.08, 'conversion': 0.02},
            'recall_random_explore': {'effectiveness': 0.4, 'ctr': 0.02, 'conversion': 0.008}
        }
        
        return performance_by_source
        
    def _calculate_diversity_score(self, recommendations: List[Dict]) -> float:
        """Calculate diversity score of recommendations"""
        if not recommendations:
            return 0.0
            
        # Calculate category diversity
        categories = [r.get('category', 'unknown') for r in recommendations]
        unique_categories = len(set(categories))
        category_diversity = unique_categories / len(recommendations)
        
        # Calculate price range diversity
        prices = [r.get('price', 50) for r in recommendations if 'price' in r]
        if prices:
            price_std = np.std(prices)
            price_diversity = min(1.0, price_std / np.mean(prices))
        else:
            price_diversity = 0.0
            
        # Combined diversity score
        diversity_score = (category_diversity + price_diversity) / 2
        return diversity_score
        
    async def _calculate_novelty_score(self, user_id: str, recommendations: List[Dict]) -> float:
        """Calculate novelty score - how new/unseen are these recommendations for the user"""
        
        # Get user's recent interaction history
        user_history = self.user_feedback_history.get(user_id, [])
        seen_items = {fb.item_id for fb in user_history}
        
        # Calculate novelty
        recommended_items = [r.get('item_id') for r in recommendations]
        novel_items = [item for item in recommended_items if item not in seen_items]
        
        novelty_score = len(novel_items) / len(recommended_items) if recommended_items else 0
        return novelty_score
        
    def _get_current_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        key = "performance_metrics"
        if key in self.metrics_history:
            return self.metrics_history[key]
        else:
            return {
                'total_feedback': 0,
                'click_count': 0,
                'purchase_count': 0,
                'like_count': 0,
                'dislike_count': 0,
                'ctr': 0.0,
                'conversion_rate': 0.0,
                'satisfaction_rate': 0.0
            }
            
    def get_system_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive system performance report"""
        
        current_metrics = self._get_current_performance_metrics()
        
        # Calculate additional insights
        total_sessions = len([k for k in self.metrics_history.keys() if k.startswith('session_')])
        avg_recommendations_per_session = np.mean([
            len(session['recommendations']) 
            for session in self.metrics_history.values() 
            if isinstance(session, dict) and 'recommendations' in session
        ]) if total_sessions > 0 else 0
        
        return {
            'overall_performance': {
                'total_feedback_events': current_metrics['total_feedback'],
                'click_through_rate': current_metrics['ctr'],
                'conversion_rate': current_metrics['conversion_rate'],
                'satisfaction_rate': current_metrics['satisfaction_rate'],
                'performance_status': self._get_performance_status(current_metrics)
            },
            'algorithm_weights': self.algorithm_weights,
            'session_statistics': {
                'total_sessions': total_sessions,
                'avg_recommendations_per_session': avg_recommendations_per_session
            },
            'performance_thresholds': self.performance_thresholds,
            'last_updated': time.time()
        }
        
    def _get_performance_status(self, metrics: Dict) -> str:
        """Determine overall performance status"""
        ctr = metrics['ctr']
        conversion_rate = metrics['conversion_rate']
        satisfaction_rate = metrics['satisfaction_rate']
        
        if (ctr >= self.performance_thresholds['ctr_threshold'] and 
            conversion_rate >= self.performance_thresholds['conversion_threshold'] and
            satisfaction_rate >= self.performance_thresholds['satisfaction_threshold']):
            return "GOOD"
        elif (ctr >= self.performance_thresholds['ctr_threshold'] * 0.7 and 
              conversion_rate >= self.performance_thresholds['conversion_threshold'] * 0.7):
            return "FAIR"
        else:
            return "NEEDS_IMPROVEMENT"


# ===== INTEGRATION CLASS =====

class EnhancedRecommendationSystem:
    """Integrated enhanced recommendation system with all core features"""
    
    def __init__(self):
        # Initialize core components
        self.feature_updater = RealTimeFeatureUpdater()
        self.cold_start_handler = ColdStartHandler(self.feature_updater)
        self.business_rules = BusinessRulesEngine()
        self.evaluator = RecommendationEvaluator(self.feature_updater)
        
        logger.info("Enhanced Recommendation System initialized with all core features")
        
    async def get_enhanced_recommendations(self, user_id: str, user_profile: Dict, 
                                         context: Dict, count: int = 20) -> Dict[str, Any]:
        """Get recommendations using all enhanced features"""
        
        start_time = time.time()
        
        # 1. Determine user type for appropriate strategy
        user_type = self.cold_start_handler.classify_user_type(user_id, user_profile)
        
        # 2. Get base recommendations based on user type
        if user_type == UserType.NEW_USER:
            base_recs = await self.cold_start_handler.handle_new_user_recommendations(
                user_id, user_profile, count * 2  # Get more for filtering
            )
        else:
            # For existing users, this would call the main pipeline
            base_recs = await self._get_regular_recommendations(user_id, count * 2)
            
        # 3. Apply business rules
        filtered_recs = await self.business_rules.apply_business_rules(
            user_id, base_recs, context
        )
        
        # 4. Apply real-time features enhancement
        enhanced_recs = await self._apply_realtime_features(user_id, filtered_recs)
        
        # 5. Final selection and ranking
        final_recs = enhanced_recs[:count]
        
        # 6. Track performance
        session_id = f"{user_id}_{int(time.time())}"
        await self.evaluator.track_recommendation_performance(user_id, final_recs, session_id)
        
        total_time = time.time() - start_time
        
        return {
            'user_id': user_id,
            'user_type': user_type.value,
            'recommendations': final_recs,
            'session_id': session_id,
            'total_latency_ms': total_time * 1000,
            'pipeline_info': {
                'base_count': len(base_recs),
                'filtered_count': len(filtered_recs),
                'final_count': len(final_recs),
                'enhancement_applied': True
            }
        }
        
    async def record_user_interaction(self, user_id: str, item_id: str, 
                                    action_type: str, context: Dict = None):
        """Record user interaction for real-time learning"""
        
        # Create feedback object
        feedback = RecommendationFeedback(
            user_id=user_id,
            item_id=item_id,
            feedback_type=action_type,
            timestamp=time.time(),
            context=context or {}
        )
        
        # Record feedback
        await self.evaluator.record_user_feedback(feedback)
        
    async def _get_regular_recommendations(self, user_id: str, count: int) -> List[Dict]:
        """Get regular recommendations for existing users (mock implementation)"""
        # This would integrate with the main recommendation pipeline
        recommendations = []
        for i in range(count):
            recommendations.append({
                'item_id': f"regular_item_{i}",
                'score': 0.8 - i * 0.01,
                'category': f"category_{i % 5}",
                'price': 50 + i * 5
            })
        return recommendations
        
    async def _apply_realtime_features(self, user_id: str, recommendations: List[Dict]) -> List[Dict]:
        """Apply real-time features to enhance recommendations"""
        
        # Get user's real-time features
        rt_features = self.feature_updater.get_user_realtime_features(user_id)
        current_interests = rt_features.get('current_interests', {})
        
        # Enhance recommendations with real-time signals
        enhanced_recs = []
        for rec in recommendations:
            enhanced_rec = rec.copy()
            
            # Boost items matching current interests
            item_category = rec.get('category', 'unknown')
            interest_boost = current_interests.get(item_category, 0) * 0.1
            
            enhanced_rec['score'] = rec.get('score', 0.5) + interest_boost
            enhanced_rec['realtime_boost'] = interest_boost
            enhanced_rec['enhanced'] = True
            
            enhanced_recs.append(enhanced_rec)
            
        # Re-sort by enhanced scores
        enhanced_recs.sort(key=lambda x: x['score'], reverse=True)
        return enhanced_recs
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'feature_updater': {
                'active_users': len(self.feature_updater.user_recent_behaviors),
                'active_items': len(self.feature_updater.item_recent_interactions)
            },
            'business_rules': {
                'total_rules': len(self.business_rules.rules),
                'active_rules': sum(1 for r in self.business_rules.rules.values() if r.active),
                'rule_stats': self.business_rules.get_rule_statistics()
            },
            'evaluator': self.evaluator.get_system_performance_report(),
            'status': 'ACTIVE',
            'last_updated': time.time()
        }


# Demo function
async def main():
    """Demo of enhanced recommendation features"""
    print("=" * 80)
    print("🚀 Enhanced Recommendation System Demo")
    print("=" * 80)
    
    # Initialize system
    system = EnhancedRecommendationSystem()
    
    # Demo 1: New user recommendations
    print("\n📝 Demo 1: New User Cold Start")
    new_user_profile = {
        'age_group': '25-34',
        'gender': 'M',
        'city': 'Shanghai',
        'registration_days': 2,
        'total_interactions': 1
    }
    
    context = {
        'device_type': 'mobile',
        'current_hour': 14,
        'user_region': 'CN'
    }
    
    result = await system.get_enhanced_recommendations(
        user_id='new_user_001',
        user_profile=new_user_profile,
        context=context,
        count=10
    )
    
    print(f"User Type: {result['user_type']}")
    print(f"Recommendations: {len(result['recommendations'])}")
    print(f"Latency: {result['total_latency_ms']:.2f}ms")
    print("Top 3 recommendations:")
    for i, rec in enumerate(result['recommendations'][:3]):
        print(f"  {i+1}. {rec['item_id']} (score: {rec['score']:.3f})")
    
    # Demo 2: User interactions
    print("\n📝 Demo 2: Real-time User Interactions")
    user_id = 'new_user_001'
    
    # Simulate user interactions
    interactions = [
        ('view', 'item_electronics_1'),
        ('click', 'item_electronics_1'), 
        ('add_cart', 'item_electronics_1'),
        ('view', 'item_books_5'),
        ('purchase', 'item_electronics_1')
    ]
    
    for action, item_id in interactions:
        await system.record_user_interaction(user_id, item_id, action, context)
        print(f"  Recorded: {action} on {item_id}")
    
    # Demo 3: Business rules impact
    print("\n📝 Demo 3: Business Rules Engine")
    rules_stats = system.business_rules.get_rule_statistics()
    print("Active business rules:")
    for rule_id, stats in rules_stats.items():
        if stats['active']:
            print(f"  - {stats['name']} (priority: {stats['priority']})")
    
    # Demo 4: System status
    print("\n📝 Demo 4: System Performance")
    status = system.get_system_status()
    perf = status['evaluator']['overall_performance']
    print(f"Performance Status: {perf['performance_status']}")
    print(f"CTR: {perf['click_through_rate']:.3f}")
    print(f"Conversion Rate: {perf['conversion_rate']:.3f}")
    
    print("\n" + "=" * 80)
    print("✅ Enhanced Recommendation System Demo Complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())