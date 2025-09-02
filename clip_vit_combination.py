#!/usr/bin/env python3
"""
CLIP + ViT Combination Strategies for E-commerce Recommendation
Author: Yang Liu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, ViTImageProcessor, ViTModel
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class ProductItem:
    item_id: str
    title: str
    description: str
    image_path: str
    category: str


class Strategy1_ParallelFeatures(nn.Module):
    """Strategy 1: Parallel extraction + feature concatenation"""
    
    def __init__(self):
        super().__init__()
        # CLIP for cross-modal features
        self.clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        self.clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        
        # ViT for pure visual features
        self.vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        
        # Feature fusion layers
        self.clip_proj = nn.Linear(512, 256)  # CLIP: 512 -> 256
        self.vit_proj = nn.Linear(768, 256)   # ViT: 768 -> 256
        self.fusion = nn.Linear(512, 256)     # Combined: 512 -> 256
        
        # Freeze pre-trained models
        for param in self.clip_model.parameters():
            param.requires_grad = False
        for param in self.vit_model.parameters():
            param.requires_grad = False
    
    def forward(self, images: List[Image.Image]) -> torch.Tensor:
        """Extract both CLIP and ViT features in parallel"""
        
        # CLIP features (cross-modal capability)
        clip_inputs = self.clip_processor(images=images, return_tensors="pt")
        with torch.no_grad():
            clip_features = self.clip_model.get_image_features(**clip_inputs)
        clip_projected = self.clip_proj(clip_features)
        
        # ViT features (pure visual understanding)
        vit_inputs = self.vit_processor(images=images, return_tensors="pt")
        with torch.no_grad():
            vit_outputs = self.vit_model(**vit_inputs)
            vit_features = vit_outputs.last_hidden_state[:, 0]  # [CLS] token
        vit_projected = self.vit_proj(vit_features)
        
        # Concatenate and fuse
        combined = torch.cat([clip_projected, vit_projected], dim=1)
        fused_features = self.fusion(combined)
        
        return fused_features


class Strategy2_HierarchicalRouting(nn.Module):
    """Strategy 2: Task-specific routing - CLIP for cross-modal, ViT for visual"""
    
    def __init__(self):
        super().__init__()
        # CLIP for cross-modal tasks
        self.clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        self.clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        
        # ViT for visual similarity
        self.vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        
        # Task-specific heads
        self.cross_modal_head = nn.Linear(512, 256)
        self.visual_similarity_head = nn.Linear(768, 256)
        
        for param in self.clip_model.parameters():
            param.requires_grad = False
        for param in self.vit_model.parameters():
            param.requires_grad = False
    
    def forward_cross_modal(self, images: List[Image.Image], texts: List[str]) -> torch.Tensor:
        """Use CLIP for text-image matching tasks"""
        inputs = self.clip_processor(text=texts, images=images, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            # Get image features for downstream tasks
            image_features = outputs.image_embeds
        
        return self.cross_modal_head(image_features)
    
    def forward_visual_similarity(self, images: List[Image.Image]) -> torch.Tensor:
        """Use ViT for pure visual similarity tasks"""
        inputs = self.vit_processor(images=images, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.vit_model(**inputs)
            features = outputs.last_hidden_state[:, 0]
        
        return self.visual_similarity_head(features)


class Strategy3_AttentionFusion(nn.Module):
    """Strategy 3: Cross-attention between CLIP and ViT features"""
    
    def __init__(self):
        super().__init__()
        # Feature extractors
        self.clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        self.clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        self.vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        
        # Projection to same dimension
        self.clip_proj = nn.Linear(512, 256)
        self.vit_proj = nn.Linear(768, 256)
        
        # Cross-attention mechanism
        self.cross_attention = nn.MultiheadAttention(256, num_heads=8)
        
        # Output projection
        self.output_proj = nn.Linear(256, 256)
        
        for param in self.clip_model.parameters():
            param.requires_grad = False
        for param in self.vit_model.parameters():
            param.requires_grad = False
    
    def forward(self, images: List[Image.Image]) -> torch.Tensor:
        """Fuse CLIP and ViT features using cross-attention"""
        
        # Extract features
        clip_inputs = self.clip_processor(images=images, return_tensors="pt")
        with torch.no_grad():
            clip_features = self.clip_model.get_image_features(**clip_inputs)
        clip_projected = self.clip_proj(clip_features)
        
        vit_inputs = self.vit_processor(images=images, return_tensors="pt")
        with torch.no_grad():
            vit_outputs = self.vit_model(**vit_inputs)
            vit_features = vit_outputs.last_hidden_state[:, 0]
        vit_projected = self.vit_proj(vit_features)
        
        # Cross-attention: CLIP attends to ViT and vice versa
        clip_attended, _ = self.cross_attention(
            clip_projected.unsqueeze(1),  # Query
            vit_projected.unsqueeze(1),   # Key, Value
            vit_projected.unsqueeze(1)
        )
        
        vit_attended, _ = self.cross_attention(
            vit_projected.unsqueeze(1),   # Query
            clip_projected.unsqueeze(1),  # Key, Value
            clip_projected.unsqueeze(1)
        )
        
        # Combine attended features
        fused = (clip_attended.squeeze(1) + vit_attended.squeeze(1)) / 2
        
        return self.output_proj(fused)


class Strategy4_EnsembleScoring(nn.Module):
    """Strategy 4: Ensemble scoring with weighted combination"""
    
    def __init__(self):
        super().__init__()
        # Feature extractors
        self.clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        self.clip_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        self.vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        
        # Separate scoring heads
        self.clip_scorer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.vit_scorer = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Learnable weights
        self.clip_weight = nn.Parameter(torch.tensor(0.6))
        self.vit_weight = nn.Parameter(torch.tensor(0.4))
        
        for param in self.clip_model.parameters():
            param.requires_grad = False
        for param in self.vit_model.parameters():
            param.requires_grad = False
    
    def forward(self, images: List[Image.Image]) -> torch.Tensor:
        """Ensemble scoring from both models"""
        
        # CLIP scoring
        clip_inputs = self.clip_processor(images=images, return_tensors="pt")
        with torch.no_grad():
            clip_features = self.clip_model.get_image_features(**clip_inputs)
        clip_score = self.clip_scorer(clip_features)
        
        # ViT scoring
        vit_inputs = self.vit_processor(images=images, return_tensors="pt")
        with torch.no_grad():
            vit_outputs = self.vit_model(**vit_inputs)
            vit_features = vit_outputs.last_hidden_state[:, 0]
        vit_score = self.vit_scorer(vit_features)
        
        # Weighted ensemble
        ensemble_score = (
            torch.sigmoid(self.clip_weight) * clip_score + 
            torch.sigmoid(self.vit_weight) * vit_score
        )
        
        return ensemble_score


class CLIPViTRecommendationSystem:
    """Complete recommendation system using CLIP+ViT combination"""
    
    def __init__(self, strategy='parallel'):
        self.strategy = strategy
        
        if strategy == 'parallel':
            self.model = Strategy1_ParallelFeatures()
        elif strategy == 'routing':
            self.model = Strategy2_HierarchicalRouting()
        elif strategy == 'attention':
            self.model = Strategy3_AttentionFusion()
        elif strategy == 'ensemble':
            self.model = Strategy4_EnsembleScoring()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def extract_item_features(self, items: List[ProductItem]) -> Dict[str, np.ndarray]:
        """Extract features for all items"""
        images = []
        item_ids = []
        
        for item in items:
            try:
                img = Image.open(item.image_path).convert('RGB')
                images.append(img)
                item_ids.append(item.item_id)
            except:
                # Skip items with invalid images
                continue
        
        if not images:
            return {}
        
        # Extract features based on strategy
        with torch.no_grad():
            if self.strategy == 'routing':
                # Use visual similarity for item features
                features = self.model.forward_visual_similarity(images)
            else:
                features = self.model(images)
        
        # Convert to dictionary
        feature_dict = {}
        for i, item_id in enumerate(item_ids):
            feature_dict[item_id] = features[i].numpy()
        
        return feature_dict
    
    def text_to_image_search(self, query_text: str, item_features: Dict[str, np.ndarray], 
                           items: List[ProductItem], top_k=10) -> List[str]:
        """Search images using text query (requires CLIP)"""
        if self.strategy != 'routing':
            raise ValueError("Text-to-image search only available with routing strategy")
        
        # Get images for items with features
        valid_items = [item for item in items if item.item_id in item_features]
        images = []
        item_ids = []
        
        for item in valid_items:
            try:
                img = Image.open(item.image_path).convert('RGB')
                images.append(img)
                item_ids.append(item.item_id)
            except:
                continue
        
        if not images:
            return []
        
        # Use CLIP for cross-modal matching
        texts = [query_text] * len(images)
        with torch.no_grad():
            cross_modal_features = self.model.forward_cross_modal(images, texts)
        
        # Calculate similarities (simplified)
        similarities = []
        for i, item_id in enumerate(item_ids):
            sim = float(torch.cosine_similarity(
                cross_modal_features[i:i+1], 
                cross_modal_features[0:1]
            ))
            similarities.append((item_id, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return [item_id for item_id, _ in similarities[:top_k]]


def demonstrate_combination_strategies():
    """Demonstrate different CLIP+ViT combination strategies"""
    print("=" * 80)
    print("CLIP + ViT Combination Strategies for E-commerce")
    print("=" * 80)
    
    strategies = {
        "Strategy 1 - Parallel Features": {
            "Description": "Extract CLIP and ViT features in parallel, then concatenate",
            "Use Case": "General recommendation with both cross-modal and visual features",
            "Pros": ["Simple to implement", "Leverages both model strengths", "Good for mixed tasks"],
            "Cons": ["Higher computational cost", "May have redundant features"]
        },
        
        "Strategy 2 - Hierarchical Routing": {
            "Description": "Route tasks to appropriate model (CLIP for cross-modal, ViT for visual)",
            "Use Case": "Different models for different recommendation scenarios",
            "Pros": ["Task-optimized", "Efficient resource usage", "Clear separation of concerns"],
            "Cons": ["More complex logic", "Need to decide routing strategy"]
        },
        
        "Strategy 3 - Cross-Attention Fusion": {
            "Description": "Use attention mechanism to let models attend to each other",
            "Use Case": "When you want models to learn from each other's representations",
            "Pros": ["Sophisticated fusion", "Learning cross-model interactions", "Better feature integration"],
            "Cons": ["Most complex", "Requires training", "Higher latency"]
        },
        
        "Strategy 4 - Ensemble Scoring": {
            "Description": "Get separate scores from each model, then weighted combination",
            "Use Case": "Final ranking stage where you want both perspectives",
            "Pros": ["Interpretable", "Can tune weights", "Robust predictions"],
            "Cons": ["Need ground truth for weight tuning", "Linear combination may be limiting"]
        }
    }
    
    for strategy_name, details in strategies.items():
        print(f"\n{strategy_name}")
        print("-" * len(strategy_name))
        print(f"Description: {details['Description']}")
        print(f"Use Case: {details['Use Case']}")
        print(f"Pros: {', '.join(details['Pros'])}")
        print(f"Cons: {', '.join(details['Cons'])}")


def industry_best_practices():
    """Show how major companies combine CLIP+ViT"""
    print("\n" + "=" * 80)
    print("Industry Best Practices - How Companies Use CLIP+ViT")
    print("=" * 80)
    
    practices = [
        {
            "Company": "Pinterest",
            "Strategy": "Hierarchical Routing",
            "Implementation": "CLIP for text-to-image search, ViT for visual similarity in same category",
            "Reason": "Different user intents need different models"
        },
        {
            "Company": "Shopify",
            "Strategy": "Parallel Features + Ensemble",
            "Implementation": "Both models score products, weighted by product category",
            "Reason": "Fashion needs visual, electronics need cross-modal"
        },
        {
            "Company": "Amazon Visual Search",
            "Strategy": "Multi-stage Pipeline",
            "Implementation": "CLIP for initial recall, ViT for fine-grained ranking",
            "Reason": "Optimize for both recall and precision"
        },
        {
            "Company": "Alibaba Taobao",
            "Strategy": "Category-specific Routing",
            "Implementation": "Fashion uses ViT, general products use CLIP",
            "Reason": "Domain expertise shows visual vs cross-modal importance"
        }
    ]
    
    for practice in practices:
        print(f"\n{practice['Company']}:")
        print(f"  Strategy: {practice['Strategy']}")
        print(f"  Implementation: {practice['Implementation']}")
        print(f"  Reason: {practice['Reason']}")
    
    print("\n" + "=" * 50)
    print("Key Takeaways:")
    print("1. No single strategy fits all - depends on use case")
    print("2. Hierarchical routing is most popular in production")
    print("3. Fashion/visual products prefer ViT")
    print("4. Cross-modal search requires CLIP")
    print("5. Ensemble methods good for critical ranking stages")


if __name__ == "__main__":
    demonstrate_combination_strategies()
    industry_best_practices()