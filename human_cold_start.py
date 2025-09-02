#!/usr/bin/env python3
"""
Cold start handler - because new users deserve good recs too
"""

import random
import numpy as np
from typing import List, Dict, Any, Optional
from collections import defaultdict
import json


class ColdStartHandler:
    """
    Handles cold start like a boss. New users? New items? No problem.
    We've got tricks up our sleeve.
    """
    
    def __init__(self):
        # Popular stuff that always works
        self.safe_bets = {
            'camera': ['electronic_a7c', 'electronic_zv1'],  # Entry level, not scary
            'audio': ['electronic_wh1000xm5', 'electronic_wf_c500'],  # One premium, one budget
            'gaming': ['ps5_console', 'ps5_games_bundle'],
            'tv': ['bravia_x85k', 'bravia_x90k']
        }
        
        # What different people usually want
        self.personas = {
            'student': {
                'budget': (50, 500),
                'categories': ['audio', 'gaming'],
                'items': ['wf_c500', 'ps5_digital', 'srs_xb13']
            },
            'professional': {
                'budget': (500, 5000),
                'categories': ['camera', 'audio'],
                'items': ['a7r5', 'wh1000xm5', 'a7c']
            },
            'family': {
                'budget': (200, 2000),
                'categories': ['tv', 'gaming', 'audio'],
                'items': ['bravia_x85k', 'ps5_console', 'srs_xg500']
            },
            'enthusiast': {
                'budget': (1000, 10000),
                'categories': ['camera', 'audio'],
                'items': ['a1', 'wh1000xm5', 'a7siii']
            }
        }
        
        # Track what works (simple MAB)
        self.strategy_stats = defaultdict(lambda: {'tries': 0, 'clicks': 0})
    
    def handle_new_user(self, user_info: Dict = None) -> List[Dict]:
        """
        New user shows up. What do we show them?
        Start safe, learn fast.
        """
        
        user_info = user_info or {}
        recs = []
        
        # Try to guess their persona
        persona = self._guess_persona(user_info)
        
        if persona:
            # We think we know this type of user
            strategy = f'persona_{persona}'
            recs = self._get_persona_recs(persona)
        
        elif user_info.get('came_from'):
            # They came from somewhere specific
            source = user_info['came_from']
            strategy = f'source_{source}'
            recs = self._get_source_based_recs(source)
        
        elif user_info.get('first_click'):
            # They clicked on something - that's a strong signal!
            clicked = user_info['first_click']
            strategy = 'first_click'
            recs = self._expand_from_item(clicked)
        
        else:
            # We know nothing Jon Snow
            strategy = 'popular'
            recs = self._get_popular_diverse()
        
        # Track what strategy we used
        self.strategy_stats[strategy]['tries'] += 1
        
        # Add metadata for tracking
        for rec in recs:
            rec['strategy'] = strategy
            rec['is_cold_start'] = True
        
        return recs
    
    def handle_new_item(self, item_info: Dict) -> Dict:
        """
        New item in catalog. Who should we show it to?
        """
        
        target_users = []
        
        # Find users who bought similar stuff
        category = item_info.get('category')
        if category:
            target_users.extend(self._get_category_fans(category))
        
        # Price point matching
        price = item_info.get('price', 500)
        if price > 2000:
            # Expensive stuff -> show to big spenders
            target_users.extend(self._get_big_spenders())
        elif price < 100:
            # Cheap stuff -> show to everyone
            target_users.extend(self._get_active_users())
        
        # Is it an upgrade/successor?
        if item_info.get('replaces'):
            old_item = item_info['replaces']
            target_users.extend(self._get_item_owners(old_item))
        
        return {
            'item_id': item_info['item_id'],
            'target_users': list(set(target_users)),  # dedupe
            'boost_score': self._calculate_new_item_boost(item_info),
            'strategy': 'new_item_promotion'
        }
    
    def _guess_persona(self, user_info: Dict) -> Optional[str]:
        """
        Try to figure out what kind of user this is.
        Like Sherlock Holmes but for e-commerce.
        """
        
        age = user_info.get('age')
        time_of_day = user_info.get('hour')
        device = user_info.get('device')
        
        # Young + late night + mobile = probably student
        if age and age < 25 and time_of_day and time_of_day > 22:
            return 'student'
        
        # Weekday morning + desktop = probably professional
        if time_of_day and 9 <= time_of_day <= 17 and device == 'desktop':
            return 'professional'
        
        # Weekend + tablet = probably family
        if user_info.get('is_weekend') and device == 'tablet':
            return 'family'
        
        # Has money to burn?
        if user_info.get('came_from') == 'premium_ad':
            return 'enthusiast'
        
        return None
    
    def _get_persona_recs(self, persona: str) -> List[Dict]:
        """Get recommendations for a specific persona"""
        
        if persona not in self.personas:
            return self._get_popular_diverse()
        
        persona_data = self.personas[persona]
        recs = []
        
        # Add their typical items
        for item_id in persona_data['items']:
            recs.append({
                'item_id': item_id,
                'score': 0.8 + random.random() * 0.2,  # Some randomness
                'reason': f'{persona}_preference'
            })
        
        # Add some exploration
        for category in persona_data['categories']:
            safe_items = self.safe_bets.get(category, [])
            for item in safe_items[:1]:  # One per category
                if not any(r['item_id'] == item for r in recs):
                    recs.append({
                        'item_id': item,
                        'score': 0.6 + random.random() * 0.2,
                        'reason': f'{category}_exploration'
                    })
        
        # Sort by score and return
        recs.sort(key=lambda x: x['score'], reverse=True)
        return recs[:10]
    
    def _get_source_based_recs(self, source: str) -> List[Dict]:
        """Recommendations based on where user came from"""
        
        recs = []
        
        if 'camera' in source.lower():
            # They came from camera ad/search
            recs.extend([
                {'item_id': 'electronic_a7c', 'score': 0.9, 'reason': 'camera_interest'},
                {'item_id': 'electronic_zv1', 'score': 0.85, 'reason': 'camera_interest'},
                {'item_id': 'lens_2470', 'score': 0.7, 'reason': 'camera_accessory'}
            ])
        
        elif 'gaming' in source.lower() or 'ps5' in source.lower():
            # Gaming interest
            recs.extend([
                {'item_id': 'ps5_console', 'score': 0.95, 'reason': 'gaming_interest'},
                {'item_id': 'ps5_games_bundle', 'score': 0.8, 'reason': 'gaming_bundle'},
                {'item_id': 'dualsense_controller', 'score': 0.7, 'reason': 'gaming_accessory'}
            ])
        
        elif 'audio' in source.lower() or 'headphone' in source.lower():
            # Audio interest
            recs.extend([
                {'item_id': 'electronic_wh1000xm5', 'score': 0.9, 'reason': 'audio_interest'},
                {'item_id': 'electronic_wf1000xm4', 'score': 0.85, 'reason': 'audio_alternative'},
                {'item_id': 'electronic_srs_xg500', 'score': 0.7, 'reason': 'audio_variety'}
            ])
        
        # Always add some popular items as fallback
        recs.extend(self._get_popular_diverse()[:5])
        
        # Dedupe and sort
        seen = set()
        unique_recs = []
        for rec in recs:
            if rec['item_id'] not in seen:
                seen.add(rec['item_id'])
                unique_recs.append(rec)
        
        return unique_recs[:10]
    
    def _expand_from_item(self, item_id: str) -> List[Dict]:
        """User clicked on something - expand from there"""
        
        recs = []
        
        # Add the item itself (they showed interest!)
        recs.append({
            'item_id': item_id,
            'score': 1.0,
            'reason': 'direct_interest'
        })
        
        # Add similar items
        category = self._get_category(item_id)
        if category:
            for similar in self.safe_bets.get(category, []):
                if similar != item_id:
                    recs.append({
                        'item_id': similar,
                        'score': 0.8,
                        'reason': 'similar_item'
                    })
        
        # Add accessories/bundles
        accessories = self._get_accessories(item_id)
        for acc in accessories[:2]:
            recs.append({
                'item_id': acc,
                'score': 0.7,
                'reason': 'accessory'
            })
        
        # Add some variety
        other_categories = [c for c in self.safe_bets.keys() if c != category]
        for other_cat in random.sample(other_categories, min(2, len(other_categories))):
            item = random.choice(self.safe_bets[other_cat])
            recs.append({
                'item_id': item,
                'score': 0.5,
                'reason': 'discovery'
            })
        
        return recs[:10]
    
    def _get_popular_diverse(self) -> List[Dict]:
        """
        Get a diverse set of popular items.
        One from each category, all safe bets.
        """
        
        recs = []
        
        # One bestseller from each category
        for category, items in self.safe_bets.items():
            if items:
                item = items[0]  # First one is usually the safest
                recs.append({
                    'item_id': item,
                    'score': 0.7,
                    'reason': f'{category}_bestseller'
                })
        
        # Shuffle for variety (don't always show in same order)
        random.shuffle(recs)
        
        # Add some seasonal/promotional items if needed
        if random.random() > 0.7:
            recs.insert(0, {
                'item_id': 'seasonal_special',
                'score': 0.75,
                'reason': 'promotion'
            })
        
        return recs
    
    def _get_category(self, item_id: str) -> Optional[str]:
        """Figure out item category from ID (hacky but works)"""
        
        if any(x in item_id.lower() for x in ['a7', 'zv', 'lens', 'camera']):
            return 'camera'
        elif any(x in item_id.lower() for x in ['wh', 'wf', 'srs', 'audio', 'headphone']):
            return 'audio'
        elif any(x in item_id.lower() for x in ['ps5', 'playstation', 'dualsense', 'game']):
            return 'gaming'
        elif any(x in item_id.lower() for x in ['bravia', 'tv', 'display']):
            return 'tv'
        return None
    
    def _get_accessories(self, item_id: str) -> List[str]:
        """Get accessories for an item"""
        
        # Hardcoded but in prod this would be a proper mapping
        accessories_map = {
            'ps5_console': ['dualsense_controller', 'pulse_3d_headset', 'ps5_media_remote'],
            'electronic_a7c': ['lens_2470', 'camera_bag', 'extra_battery'],
            'electronic_a7r5': ['lens_2470gm', 'lens_70200', 'vertical_grip'],
            'electronic_wh1000xm5': ['carrying_case', 'audio_cable'],
        }
        
        return accessories_map.get(item_id, [])
    
    def _get_category_fans(self, category: str) -> List[str]:
        """Get users who love this category"""
        # Mock implementation
        return [f'user_{i}' for i in range(100, 200)]
    
    def _get_big_spenders(self) -> List[str]:
        """Get users with deep pockets"""
        # Mock implementation
        return [f'premium_user_{i}' for i in range(50)]
    
    def _get_active_users(self) -> List[str]:
        """Get active users for broad reach"""
        # Mock implementation
        return [f'active_user_{i}' for i in range(500)]
    
    def _get_item_owners(self, item_id: str) -> List[str]:
        """Get users who own a specific item"""
        # Mock implementation
        return [f'owner_{item_id}_{i}' for i in range(20)]
    
    def _calculate_new_item_boost(self, item_info: Dict) -> float:
        """Calculate boost score for new item"""
        
        boost = 0.5  # Base boost for being new
        
        # Premium items get more boost
        if item_info.get('price', 0) > 2000:
            boost += 0.2
        
        # Hot categories get more boost
        if item_info.get('category') in ['gaming', 'camera']:
            boost += 0.1
        
        # Limited edition? Push it hard!
        if item_info.get('limited_edition'):
            boost += 0.3
        
        return min(boost, 1.0)
    
    def update_feedback(self, strategy: str, clicked: bool):
        """
        Update our strategy stats based on user feedback.
        Simple but effective.
        """
        
        if strategy in self.strategy_stats:
            if clicked:
                self.strategy_stats[strategy]['clicks'] += 1
            
            # Calculate success rate
            stats = self.strategy_stats[strategy]
            if stats['tries'] > 0:
                stats['success_rate'] = stats['clicks'] / stats['tries']
    
    def get_stats(self) -> Dict:
        """Get performance stats for monitoring"""
        
        stats = {}
        for strategy, data in self.strategy_stats.items():
            if data['tries'] > 0:
                stats[strategy] = {
                    'tries': data['tries'],
                    'clicks': data['clicks'],
                    'ctr': data['clicks'] / data['tries']
                }
        
        return stats


def demo():
    """Test drive our cold start handler"""
    
    handler = ColdStartHandler()
    
    print("=" * 60)
    print("COLD START DEMO")
    print("=" * 60)
    
    # Test 1: Brand new user, no info
    print("\n1. Brand new user (no info):")
    recs = handler.handle_new_user()
    for rec in recs[:5]:
        print(f"  - {rec['item_id']} (score: {rec['score']:.2f}, {rec['reason']})")
    
    # Test 2: Young user at night
    print("\n2. Young user browsing at night:")
    recs = handler.handle_new_user({'age': 22, 'hour': 23})
    for rec in recs[:5]:
        print(f"  - {rec['item_id']} (score: {rec['score']:.2f}, {rec['reason']})")
    
    # Test 3: User from camera ad
    print("\n3. User from camera advertisement:")
    recs = handler.handle_new_user({'came_from': 'camera_google_ad'})
    for rec in recs[:5]:
        print(f"  - {rec['item_id']} (score: {rec['score']:.2f}, {rec['reason']})")
    
    # Test 4: User clicked on PS5
    print("\n4. User's first click was PS5:")
    recs = handler.handle_new_user({'first_click': 'ps5_console'})
    for rec in recs[:5]:
        print(f"  - {rec['item_id']} (score: {rec['score']:.2f}, {rec['reason']})")
    
    # Test 5: New expensive camera
    print("\n5. New item: Premium camera")
    result = handler.handle_new_item({
        'item_id': 'electronic_a1_mark2',
        'category': 'camera',
        'price': 6500,
        'replaces': 'electronic_a1'
    })
    print(f"  Target {len(result['target_users'])} users")
    print(f"  Boost score: {result['boost_score']:.2f}")
    
    print("\n✨ Cold start handled like a pro!")


if __name__ == "__main__":
    demo()