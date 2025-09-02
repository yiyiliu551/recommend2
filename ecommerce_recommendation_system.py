"""
Amazon-Style E-commerce Recommendation System
Author: Yang Liu
Description: Comprehensive product recommendation system for large-scale e-commerce platform
Business Focus: Multi-category retail, personalization, cross-selling, inventory optimization
Experience: Built similar systems at Weibo (400M DAU), Qunar, SHAREit, CNKI
"""

import numpy as np
# import pandas as pd  # Commented out due to numpy compatibility issues
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import json


@dataclass
class EcommerceProduct:
    """E-commerce product for general retail platform"""
    product_id: str
    title: str
    category: str  # electronics, clothing, books, home, sports, beauty, toys
    subcategory: str  # smartphone, laptop, dress, novel, kitchen, fitness, skincare
    
    # Product attributes  
    brand: str
    
    # Pricing information
    price: float
    original_price: float
    discount_percentage: float = 0.0
    is_on_sale: bool = False
    rating: float = 0.0
    review_count: int = 0
    availability: str = "in_stock"  # in_stock, out_of_stock, pre_order, limited_stock
    
    # E-commerce specific
    shipping_required: bool = True  # True for physical products
    shipping_weight: float = 0.0
    dimensions: str = ""
    specifications: Dict[str, str] = field(default_factory=dict)
    color_options: List[str] = field(default_factory=list)
    size_options: List[str] = field(default_factory=list)
    bundle_items: List[str] = field(default_factory=list)  # Bundle components
    
    # Performance metrics
    view_count: int = 0
    add_to_cart_count: int = 0
    purchase_count: int = 0
    return_rate: float = 0.0


@dataclass
class Customer:
    """E-commerce customer profile"""
    customer_id: str
    email: str
    registration_date: datetime
    
    # Demographics
    age_group: str  # 13-17, 18-25, 26-35, 36-45, 45+
    location: str
    preferred_language: str
    
    # Purchase behavior
    total_orders: int = 0
    total_spent: float = 0.0
    avg_order_value: float = 0.0
    preferred_categories: List[str] = field(default_factory=list)
    
    # Shopping patterns
    is_price_sensitive: bool = True
    prefers_bundles: bool = False
    frequent_buyer: bool = False  # More than 5 orders per year
    seasonal_buyer: bool = False  # Only buys during sales
    
    # Digital behavior
    newsletter_subscriber: bool = False
    mobile_app_user: bool = False
    social_media_follower: bool = False


@dataclass
class ShoppingSession:
    """Customer shopping session"""
    session_id: str
    customer_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Session behavior
    pages_viewed: List[str] = field(default_factory=list)
    products_viewed: List[str] = field(default_factory=list)
    search_queries: List[str] = field(default_factory=list)
    cart_additions: List[str] = field(default_factory=list)
    cart_removals: List[str] = field(default_factory=list)
    
    # Session outcome
    purchase_made: bool = False
    cart_abandoned: bool = False
    total_session_value: float = 0.0
    
    # Context
    traffic_source: str = "direct"  # direct, search, social, email, ad
    device_type: str = "desktop"  # desktop, mobile, tablet, console
    is_returning_visitor: bool = False


class EcommerceFeatureEngineer:
    """Feature engineering for e-commerce recommendations"""
    
    def __init__(self):
        self.category_weights = {
            'electronics': 1.2,
            'clothing': 0.9,
            'books': 0.7,
            'home': 1.0,
            'sports': 0.8,
            'beauty': 0.9,
            'toys': 0.6,
            'automotive': 1.1,
            'food': 0.5
        }
        
        # Interaction weights for e-commerce
        self.interaction_weights = {
            'view': 1.0,
            'add_to_cart': 5.0,
            'purchase': 10.0,
            'review': 8.0,
            'wishlist': 3.0,
            'share': 4.0,
            'return': -5.0
        }
    
    def extract_customer_features(self, customer: Customer, 
                                 sessions: List[ShoppingSession],
                                 purchase_history: List[Dict]) -> Dict[str, Any]:
        """Extract comprehensive customer features for recommendations"""
        
        features = {
            # Basic customer profile
            'customer_tenure_days': (datetime.now() - customer.registration_date).days,
            'total_lifetime_value': customer.total_spent,
            'avg_order_value': customer.avg_order_value,
            'purchase_frequency': customer.total_orders / max((datetime.now() - customer.registration_date).days / 365, 1),
            
            # Shopping behavior patterns
            'price_sensitivity_score': 1.0 if customer.is_price_sensitive else 0.3,
            'bundle_preference': 1.0 if customer.prefers_bundles else 0.0,
            'brand_loyalty_score': self._calculate_brand_loyalty(purchase_history),
            
            # Engagement features
            'newsletter_engagement': 1.0 if customer.newsletter_subscriber else 0.0,
            'mobile_usage': 1.0 if customer.mobile_app_user else 0.0,
            'social_engagement': 1.0 if customer.social_media_follower else 0.0,
        }
        
        # Analyze recent shopping sessions
        if sessions:
            recent_sessions = [s for s in sessions if (datetime.now() - s.start_time).days <= 30]
            
            if recent_sessions:
                features.update({
                    'monthly_session_count': len(recent_sessions),
                    'avg_session_duration': np.mean([
                        (s.end_time - s.start_time).total_seconds() / 60 
                        for s in recent_sessions if s.end_time
                    ]) if any(s.end_time for s in recent_sessions) else 0,
                    'cart_abandonment_rate': sum(1 for s in recent_sessions if s.cart_abandoned) / len(recent_sessions),
                    'conversion_rate': sum(1 for s in recent_sessions if s.purchase_made) / len(recent_sessions),
                    'avg_cart_size': np.mean([len(s.cart_additions) for s in recent_sessions]),
                })
                
                # Device and channel preferences
                device_usage = defaultdict(int)
                traffic_sources = defaultdict(int)
                
                for session in recent_sessions:
                    device_usage[session.device_type] += 1
                    traffic_sources[session.traffic_source] += 1
                
                features['preferred_device'] = max(device_usage, key=device_usage.get) if device_usage else 'desktop'
                features['primary_traffic_source'] = max(traffic_sources, key=traffic_sources.get) if traffic_sources else 'direct'
        
        # Category preferences from purchase history
        category_purchases = defaultdict(float)
        for purchase in purchase_history:
            category = purchase.get('category', 'unknown')
            amount = purchase.get('amount', 0)
            category_purchases[category] += amount
        
        total_spent = sum(category_purchases.values())
        for category in self.category_weights.keys():
            features[f'category_preference_{category}'] = category_purchases.get(category, 0) / max(total_spent, 1)
        
        return features
    
    def extract_product_features(self, product: EcommerceProduct, 
                                market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract product features for recommendation scoring"""
        
        features = {
            # Basic product attributes
            'price_tier': self._categorize_price_tier(product.price, product.category),
            'discount_attractiveness': product.discount_percentage / 100,
            'review_quality_score': min(product.rating / 5.0, 1.0) if product.rating > 0 else 0.5,
            'review_volume_score': min(np.log1p(product.review_count) / 10, 1.0),
            
            # Availability and urgency
            'availability_score': self._encode_availability(product.availability),
            'urgency_factor': 1.0 if product.is_on_sale else 0.0,
            
            # Product performance
            'popularity_score': np.log1p(product.view_count + product.purchase_count),
            'conversion_rate': product.purchase_count / max(product.view_count, 1),
            'cart_conversion_rate': product.purchase_count / max(product.add_to_cart_count, 1),
            'return_risk': product.return_rate,
            
            # Category-specific features
            'category_weight': self.category_weights.get(product.category, 0.5),
            'is_digital_product': 0.0 if product.shipping_required else 1.0,
            'color_variety': len(product.color_options) / 5.0,  # Normalize by max color options
            'size_variety': len(product.size_options) / 10.0,  # Normalize by max size options
            
            # Bundle and upsell potential
            'bundle_potential': 1.0 if product.bundle_items else 0.0,
            'bundle_value': len(product.bundle_items) * 0.2 if product.bundle_items else 0.0,
        }
        
        # Brand features
        features['brand_premium'] = self._calculate_brand_premium(product.brand)
        
        # Market position features
        if 'category_stats' in market_data:
            category_stats = market_data['category_stats'].get(product.category, {})
            avg_price = category_stats.get('avg_price', product.price)
            features['price_competitiveness'] = avg_price / max(product.price, 0.01)
        
        return features
    
    def extract_contextual_features(self, customer: Customer, 
                                   session: ShoppingSession,
                                   external_context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract contextual features for real-time recommendations"""
        
        features = {
            # Temporal context
            'hour_of_day': session.start_time.hour / 24,
            'day_of_week': session.start_time.weekday() / 7,
            'is_weekend': 1.0 if session.start_time.weekday() >= 5 else 0.0,
            'is_holiday_season': self._is_holiday_season(session.start_time),
            
            # Session context
            'session_depth': len(session.products_viewed),
            'search_intent_strength': len(session.search_queries) / max(len(session.products_viewed), 1),
            'cart_activity': len(session.cart_additions) + len(session.cart_removals),
            
            # Device and channel context
            'device_mobile': 1.0 if session.device_type == 'mobile' else 0.0,
            'from_marketing_campaign': 1.0 if session.traffic_source in ['email', 'ad', 'social'] else 0.0,
            'is_returning_session': 1.0 if session.is_returning_visitor else 0.0,
            
            # External context
            'competitor_sale_active': external_context.get('competitor_promotions', 0.0),
            'inventory_pressure': external_context.get('inventory_clearance_mode', 0.0),
            'seasonal_demand': external_context.get('seasonal_multiplier', 1.0),
        }
        
        # Shopping stage inference
        if len(session.cart_additions) > 0:
            features['shopping_stage'] = 0.8  # Consideration/Purchase
        elif len(session.search_queries) > 0:
            features['shopping_stage'] = 0.5  # Research
        else:
            features['shopping_stage'] = 0.2  # Browse
        
        return features
    
    def _calculate_brand_loyalty(self, purchase_history: List[Dict]) -> float:
        """Calculate customer brand loyalty score"""
        if not purchase_history:
            return 0.0
        
        brand_purchases = defaultdict(int)
        for purchase in purchase_history:
            brand = purchase.get('brand', 'unknown')
            brand_purchases[brand] += 1
        
        if not brand_purchases:
            return 0.0
        
        total_purchases = sum(brand_purchases.values())
        max_brand_purchases = max(brand_purchases.values())
        
        return max_brand_purchases / total_purchases
    
    def _categorize_price_tier(self, price: float, category: str) -> float:
        """Categorize product into price tier (0-1 scale)"""
        # Price tiers by category (simplified)
        price_ranges = {
            'electronics': {'low': 50, 'high': 2000},
            'clothing': {'low': 20, 'high': 300},
            'books': {'low': 10, 'high': 50},
            'home': {'low': 30, 'high': 500},
            'sports': {'low': 25, 'high': 400},
            'beauty': {'low': 15, 'high': 150},
            'toys': {'low': 10, 'high': 200},
            'automotive': {'low': 50, 'high': 1000}
        }
        
        range_info = price_ranges.get(category, {'low': 10, 'high': 100})
        normalized = (price - range_info['low']) / (range_info['high'] - range_info['low'])
        
        return np.clip(normalized, 0.0, 1.0)
    
    def _encode_availability(self, availability: str) -> float:
        """Encode availability status"""
        mapping = {
            'in_stock': 1.0,
            'limited_stock': 0.7,
            'pre_order': 0.5,
            'out_of_stock': 0.0
        }
        return mapping.get(availability, 0.5)
    
    def _calculate_brand_premium(self, brand: str) -> float:
        """Calculate brand premium score"""
        # Premium brands across various categories
        premium_brands = {
            'Apple': 1.0,
            'Samsung': 0.9,
            'Nike': 0.9,
            'Adidas': 0.8,
            'Electronic': 0.9,
            'Canon': 0.8,
            'Coach': 1.0,
            'Gucci': 1.0,
            'BMW': 1.0,
            'Mercedes': 1.0,
            'Amazon': 0.7,
            'Target': 0.5
        }
        return premium_brands.get(brand, 0.3)
    
    def _is_holiday_season(self, date: datetime) -> float:
        """Determine if date is in holiday shopping season"""
        month = date.month
        
        # Holiday seasons
        if month == 11:  # Black Friday
            return 1.0
        elif month == 12:  # Christmas
            return 1.0
        elif month in [6, 7]:  # Summer sales
            return 0.7
        elif month == 3:  # Spring sales
            return 0.5
        else:
            return 0.0


class EcommerceRecommendationEngine:
    """Amazon-style e-commerce recommendation engine for multi-category retail"""
    
    def __init__(self):
        self.feature_engineer = EcommerceFeatureEngineer()
        
        # Recommendation strategies
        self.strategy_weights = {
            'collaborative_filtering': 0.35,
            'content_based': 0.25,
            'popularity_based': 0.15,
            'cross_sell': 0.15,
            'business_rules': 0.10
        }
        
        # Business objectives
        self.business_weights = {
            'revenue_optimization': 0.4,
            'customer_satisfaction': 0.3,
            'inventory_management': 0.2,
            'brand_promotion': 0.1
        }
    
    def recommend_products(self, customer: Customer,
                          session: ShoppingSession,
                          candidate_products: List[EcommerceProduct],
                          context: Dict[str, Any],
                          n_recommendations: int = 10) -> List[Tuple[str, float, Dict]]:
        """Generate personalized product recommendations"""
        
        # Extract features
        customer_features = self.feature_engineer.extract_customer_features(
            customer, [session], context.get('purchase_history', [])
        )
        
        contextual_features = self.feature_engineer.extract_contextual_features(
            customer, session, context.get('external_context', {})
        )
        
        recommendations = []
        
        for product in candidate_products:
            # Extract product features
            product_features = self.feature_engineer.extract_product_features(
                product, context.get('market_data', {})
            )
            
            # Compute recommendation score
            score_components = self._compute_recommendation_score(
                customer_features, product_features, contextual_features, 
                customer, product, session
            )
            
            final_score = sum(
                score_components[strategy] * weight
                for strategy, weight in self.strategy_weights.items()
                if strategy in score_components
            )
            
            # Apply business rules and constraints
            final_score = self._apply_business_rules(
                final_score, customer, product, session, context
            )
            
            recommendations.append((product.product_id, final_score, score_components))
        
        # Sort and return top N
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]
    
    def _compute_recommendation_score(self, customer_features: Dict, 
                                    product_features: Dict,
                                    contextual_features: Dict,
                                    customer: Customer,
                                    product: EcommerceProduct,
                                    session: ShoppingSession) -> Dict[str, float]:
        """Compute individual recommendation score components"""
        
        components = {}
        
        # 1. Collaborative Filtering Score
        category_pref = customer_features.get(f'category_preference_{product.category}', 0)
        price_match = self._calculate_price_preference_match(customer, product)
        components['collaborative_filtering'] = (category_pref * 0.6 + price_match * 0.4)
        
        # 2. Content-Based Score
        brand_match = self._calculate_brand_affinity(customer_features, product)
        quality_score = product_features['review_quality_score']
        components['content_based'] = (brand_match * 0.5 + quality_score * 0.5)
        
        # 3. Popularity Score
        components['popularity_based'] = product_features['popularity_score'] / 10.0  # Normalize
        
        # 4. Cross-sell Score
        components['cross_sell'] = self._calculate_cross_sell_potential(session, product)
        
        # 5. Business Rules Score
        components['business_rules'] = self._calculate_business_priority(product, contextual_features)
        
        return components
    
    def _calculate_price_preference_match(self, customer: Customer, product: EcommerceProduct) -> float:
        """Calculate how well product price matches customer preferences"""
        
        # If customer is price sensitive and product is on sale
        if customer.is_price_sensitive and product.is_on_sale:
            return 0.9
        
        # If customer is not price sensitive and product is premium
        elif not customer.is_price_sensitive and product.price > 50:
            return 0.8
        
        # Match average order value
        if customer.avg_order_value > 0:
            price_ratio = product.price / customer.avg_order_value
            if 0.5 <= price_ratio <= 2.0:  # Within reasonable range
                return 0.7
        
        return 0.5  # Default moderate score
    
    def _calculate_brand_affinity(self, customer_features: Dict, product: EcommerceProduct) -> float:
        """Calculate customer affinity for product brand"""
        brand_loyalty = customer_features.get('brand_loyalty_score', 0)
        brand_premium = self.feature_engineer._calculate_brand_premium(product.brand)
        
        return (brand_loyalty * 0.6 + brand_premium * 0.4)
    
    def _calculate_cross_sell_potential(self, session: ShoppingSession, product: EcommerceProduct) -> float:
        """Calculate cross-sell potential based on current session"""
        
        # If customer has items in cart, recommend complementary products
        if session.cart_additions:
            # Simplified cross-sell logic for various categories
            if product.category == 'electronics' and 'accessories' in session.cart_additions:
                return 0.8  # Electronics accessories complement main products
            elif product.category == 'clothing' and any('clothing' in item for item in session.cart_additions):
                return 0.7  # Clothing items complement each other
            elif product.category == 'home' and any('kitchen' in item for item in session.cart_additions):
                return 0.6  # Home products often bought together
        
        # If customer searched for specific terms
        if session.search_queries:
            # Simple keyword matching (in production, use more sophisticated NLP)
            for query in session.search_queries:
                if any(word in product.title.lower() for word in query.lower().split()):
                    return 0.9
        
        return 0.3  # Default low cross-sell score
    
    def _calculate_business_priority(self, product: EcommerceProduct, 
                                   contextual_features: Dict) -> float:
        """Calculate business priority score for promoting certain products"""
        
        score = 0.5  # Base score
        
        # Promote products on sale
        if product.is_on_sale:
            score += 0.2
        
        # Promote high-margin products (simplified)
        if product.category in ['beauty', 'clothing', 'toys']:
            score += 0.15
        
        # Promote during inventory clearance
        if contextual_features.get('inventory_pressure', 0) > 0.5:
            score += 0.2
        
        # Promote new releases
        if product.availability == 'pre_order':
            score += 0.1
        
        return min(score, 1.0)
    
    def _apply_business_rules(self, base_score: float, customer: Customer,
                            product: EcommerceProduct, session: ShoppingSession,
                            context: Dict) -> float:
        """Apply business rules and constraints"""
        
        final_score = base_score
        
        # Don't recommend out-of-stock items
        if product.availability == 'out_of_stock':
            final_score *= 0.1
        
        # Boost recommendations during sales
        if product.is_on_sale and customer.is_price_sensitive:
            final_score *= 1.3
        
        # Reduce score for recently returned products
        if product.return_rate > 0.2:
            final_score *= 0.8
        
        # Boost bundle recommendations for bundle-preferring customers
        if customer.prefers_bundles and product.bundle_items:
            final_score *= 1.2
        
        # Regional availability (simplified)
        if context.get('region_restricted_products') and product.product_id in context['region_restricted_products']:
            final_score *= 0.1
        
        return min(final_score, 1.0)


def demo_ecommerce_recommendations():
    """Demonstrate Amazon-style e-commerce recommendation system"""
    
    print("🛒 Amazon-Style E-commerce Recommendation System Demo")
    print("="*65)
    
    # Create sample customer
    customer = Customer(
        customer_id="cust_12345",
        email="shopper@email.com",
        registration_date=datetime.now() - timedelta(days=365),
        age_group="26-35",
        location="US",
        preferred_language="en-US",
        total_orders=25,
        total_spent=1250.0,
        avg_order_value=50.0,
        preferred_categories=["electronics", "clothing"],
        is_price_sensitive=True,
        frequent_buyer=True
    )
    
    # Create sample products (Amazon-style diverse catalog)
    products = [
        EcommerceProduct(
            product_id="iphone_15_pro",
            title="Apple iPhone 15 Pro 256GB",
            category="electronics",
            subcategory="smartphone",
            price=999.99,
            original_price=999.99,
            brand="Apple",
            rating=4.7,
            review_count=28420,
            availability="in_stock",
            shipping_required=True,
            shipping_weight=0.5,
            color_options=["Natural Titanium", "Blue Titanium", "White Titanium"],
            specifications={"Storage": "256GB", "Camera": "48MP", "Display": "6.1 inch"},
            view_count=150000,
            add_to_cart_count=15000,
            purchase_count=8200
        ),
        EcommerceProduct(
            product_id="nike_air_max_90",
            title="Nike Air Max 90 Running Shoes",
            category="clothing",
            subcategory="shoes",
            price=89.99,
            original_price=109.99,
            discount_percentage=18.2,
            is_on_sale=True,
            brand="Nike",
            rating=4.5,
            review_count=12500,
            availability="in_stock",
            shipping_required=True,
            shipping_weight=1.2,
            color_options=["White", "Black", "Red", "Blue"],
            size_options=["7", "7.5", "8", "8.5", "9", "9.5", "10"],
            view_count=45000,
            add_to_cart_count=5500,
            purchase_count=3800
        ),
        EcommerceProduct(
            product_id="instant_pot_duo",
            title="Instant Pot Duo 7-in-1 Electric Pressure Cooker",
            category="home",
            subcategory="kitchen",
            price=79.99,
            original_price=99.99,
            discount_percentage=20.0,
            is_on_sale=True,
            brand="Instant Pot",
            rating=4.8,
            review_count=45200,
            availability="in_stock",
            shipping_required=True,
            shipping_weight=12.5,
            specifications={"Capacity": "6 Quart", "Functions": "7-in-1", "Material": "Stainless Steel"},
            view_count=85000,
            add_to_cart_count=12000,
            purchase_count=9500
        ),
        EcommerceProduct(
            product_id="kindle_paperwhite",
            title="Amazon Kindle Paperwhite (11th Generation)",
            category="electronics",
            subcategory="e_reader",
            price=139.99,
            original_price=139.99,
            brand="Amazon",
            rating=4.6,
            review_count=18900,
            availability="in_stock",
            shipping_required=True,
            shipping_weight=0.7,
            specifications={"Storage": "8GB", "Display": "6.8 inch", "Battery": "10 weeks"},
            view_count=32000,
            add_to_cart_count=4500,
            purchase_count=3200
        )
    ]
    
    # Create shopping session
    session = ShoppingSession(
        session_id="sess_789",
        customer_id="cust_12345",
        start_time=datetime.now() - timedelta(minutes=30),
        products_viewed=["iphone_15_pro", "nike_air_max_90"],
        search_queries=["iPhone 15", "running shoes", "kitchen appliances"],
        cart_additions=["iphone_15_pro"],
        traffic_source="search",
        device_type="mobile"
    )
    
    # Context
    context = {
        'purchase_history': [
            {'category': 'electronics', 'amount': 800, 'brand': 'Apple'},
            {'category': 'clothing', 'amount': 120, 'brand': 'Nike'},
            {'category': 'home', 'amount': 85, 'brand': 'Instant Pot'},
            {'category': 'books', 'amount': 25, 'brand': 'Amazon'}
        ],
        'external_context': {
            'seasonal_multiplier': 1.3,  # Black Friday season
            'inventory_clearance_mode': 0.4,
            'competitor_promotions': 0.6
        },
        'market_data': {
            'category_stats': {
                'electronics': {'avg_price': 650},
                'clothing': {'avg_price': 75},
                'home': {'avg_price': 120}
            }
        }
    }
    
    # Generate recommendations
    engine = EcommerceRecommendationEngine()
    recommendations = engine.recommend_products(
        customer, session, products, context, n_recommendations=4
    )
    
    # Display results
    print(f"\n🎯 Personalized Recommendations for Customer {customer.customer_id}")
    print(f"Profile: {customer.age_group}, Frequent Buyer, Price Sensitive")
    print(f"Current Session: Viewed iPhone 15 Pro, Added to Cart, Searched for shoes")
    print("-" * 65)
    
    for i, (product_id, score, components) in enumerate(recommendations, 1):
        product = next(p for p in products if p.product_id == product_id)
        
        print(f"{i}. {product.title}")
        print(f"   💰 ${product.price:.2f}", end="")
        if product.is_on_sale:
            print(f" (was ${product.original_price:.2f}) - {product.discount_percentage:.1f}% OFF! 🔥", end="")
        print()
        print(f"   ⭐ {product.rating}/5.0 ({product.review_count:,} reviews)")
        print(f"   📦 {product.availability.replace('_', ' ').title()}")
        print(f"   🎯 Recommendation Score: {score:.3f}")
        
        print(f"   📊 Score Breakdown:")
        for component, value in components.items():
            print(f"     • {component.replace('_', ' ').title()}: {value:.3f}")
        print()
    
    print("✅ Amazon-style e-commerce recommendation demo completed!")
    print("\nKey Features Demonstrated:")
    print("• Multi-category product catalog (Electronics, Fashion, Home, Books)")
    print("• Customer lifetime value and purchase behavior analysis")
    print("• Real-time shopping session context")
    print("• Cross-selling and upselling strategies")
    print("• Dynamic pricing and promotional recommendations")
    print("• Brand affinity and premium product promotion")
    print("• Inventory optimization and clearance strategies")


if __name__ == "__main__":
    demo_ecommerce_recommendations()