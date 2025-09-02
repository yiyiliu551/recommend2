#!/usr/bin/env python3
"""
Mainstream Image Feature Extraction Methods Comparison
Author: Yang Liu
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from transformers import CLIPProcessor, CLIPModel, ViTImageProcessor, ViTModel
from transformers import DeiTImageProcessor, DeiTModel
import timm
from PIL import Image
import numpy as np
from typing import List
import time


class CLIPImageExtractor(nn.Module):
    """CLIP Vision Encoder - Best for cross-modal tasks"""
    
    def __init__(self):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        self.model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, images: List[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        
        return image_features  # [batch_size, 512]


class ViTImageExtractor(nn.Module):
    """Vision Transformer - State-of-the-art for pure vision tasks"""
    
    def __init__(self, model_name='google/vit-base-patch16-224'):
        super().__init__()
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)
        
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, images: List[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding
            features = outputs.last_hidden_state[:, 0]
        
        return features  # [batch_size, 768]


class ResNetImageExtractor(nn.Module):
    """ResNet - Traditional CNN, still very popular in industry"""
    
    def __init__(self, model_name='resnet50'):
        super().__init__()
        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
        elif model_name == 'resnet101':
            self.model = models.resnet101(pretrained=True)
        
        # Remove final classification layer
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, images: List[Image.Image]) -> torch.Tensor:
        tensors = []
        for img in images:
            tensor = self.transform(img).unsqueeze(0)
            tensors.append(tensor)
        
        batch_tensor = torch.cat(tensors, dim=0)
        
        with torch.no_grad():
            features = self.model(batch_tensor)
            features = features.flatten(1)  # Flatten spatial dimensions
        
        return features  # [batch_size, 2048]


class EfficientNetImageExtractor(nn.Module):
    """EfficientNet - Good efficiency/performance trade-off"""
    
    def __init__(self, model_name='efficientnet_b0'):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        
        # Get the transform from timm
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**data_config, is_training=False)
        
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, images: List[Image.Image]) -> torch.Tensor:
        tensors = []
        for img in images:
            tensor = self.transform(img).unsqueeze(0)
            tensors.append(tensor)
        
        batch_tensor = torch.cat(tensors, dim=0)
        
        with torch.no_grad():
            features = self.model(batch_tensor)
        
        return features  # [batch_size, feature_dim]


class DeiTImageExtractor(nn.Module):
    """DeiT - Data-efficient Image Transformer"""
    
    def __init__(self):
        super().__init__()
        self.processor = DeiTImageProcessor.from_pretrained('facebook/deit-base-distilled-patch16-224')
        self.model = DeiTModel.from_pretrained('facebook/deit-base-distilled-patch16-224')
        
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, images: List[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token
            features = outputs.last_hidden_state[:, 0]
        
        return features  # [batch_size, 768]


def benchmark_models():
    """Benchmark different image feature extraction models"""
    print("=" * 80)
    print("Image Feature Extraction Models Benchmark")
    print("=" * 80)
    
    # Create test images
    test_images = [Image.new('RGB', (224, 224), color=f'#{i:02x}{i:02x}{i:02x}') for i in range(10)]
    
    models_config = {
        'CLIP-ViT': {
            'model': CLIPImageExtractor(),
            'feature_dim': 512,
            'use_case': 'Cross-modal, text-image matching',
            'pros': 'Best for multimodal, zero-shot classification',
            'cons': 'Smaller feature dimension'
        },
        'ViT-Base': {
            'model': ViTImageExtractor(),
            'feature_dim': 768,
            'use_case': 'Pure vision tasks, high accuracy',
            'pros': 'SOTA performance, large feature dimension',
            'cons': 'Requires more computation'
        },
        'ResNet-50': {
            'model': ResNetImageExtractor('resnet50'),
            'feature_dim': 2048,
            'use_case': 'Industry standard, reliable',
            'pros': 'Fast, mature, well-understood',
            'cons': 'CNN limitations, not SOTA'
        },
        'EfficientNet-B0': {
            'model': EfficientNetImageExtractor(),
            'feature_dim': 1280,
            'use_case': 'Efficiency-focused applications',
            'pros': 'Good accuracy/speed trade-off',
            'cons': 'Complex architecture'
        },
        'DeiT-Base': {
            'model': DeiTImageExtractor(),
            'feature_dim': 768,
            'use_case': 'Data-efficient training',
            'pros': 'Good with limited data',
            'cons': 'Less popular than ViT'
        }
    }
    
    print(f"{'Model':<15} {'Feature Dim':<12} {'Time (ms)':<10} {'Use Case'}")
    print("-" * 80)
    
    for name, config in models_config.items():
        try:
            model = config['model']
            model.eval()
            
            # Benchmark speed
            start_time = time.time()
            with torch.no_grad():
                features = model(test_images[:5])  # Test with 5 images
            end_time = time.time()
            
            inference_time = (end_time - start_time) * 1000 / 5  # ms per image
            
            print(f"{name:<15} {config['feature_dim']:<12} {inference_time:<10.1f} {config['use_case']}")
            
        except Exception as e:
            print(f"{name:<15} {'ERROR':<12} {'N/A':<10} {str(e)[:30]}...")
    
    print("\n" + "=" * 80)
    print("Detailed Analysis")
    print("=" * 80)
    
    for name, config in models_config.items():
        print(f"\n{name}:")
        print(f"  Feature Dimension: {config['feature_dim']}")
        print(f"  Use Case: {config['use_case']}")
        print(f"  Pros: {config['pros']}")
        print(f"  Cons: {config['cons']}")


def recommend_for_ecommerce():
    """Recommend best approach for e-commerce"""
    print("\n" + "=" * 80)
    print("E-commerce Recommendation System - Best Practices")
    print("=" * 80)
    
    recommendations = {
        "1. Pure Image Similarity (Fashion/Home)": {
            "Best Choice": "ViT-Base or CLIP",
            "Reason": "Need fine-grained visual understanding",
            "Implementation": "Use ViT for pure visual similarity, CLIP for text-image search"
        },
        
        "2. Cross-modal Search (Text -> Image)": {
            "Best Choice": "CLIP (Mandatory)",
            "Reason": "Only model that aligns text and image in same space",
            "Implementation": "CLIP is the industry standard for this"
        },
        
        "3. Large-scale Production (Millions of items)": {
            "Best Choice": "ResNet-50 or EfficientNet",
            "Reason": "Faster inference, mature deployment ecosystem",
            "Implementation": "Use CNN for speed, fine-tune on your data"
        },
        
        "4. Cold Start (New products with little data)": {
            "Best Choice": "CLIP",
            "Reason": "Pre-trained on web-scale data, generalizes well",
            "Implementation": "Zero-shot classification, no fine-tuning needed"
        },
        
        "5. Mobile/Edge Deployment": {
            "Best Choice": "EfficientNet-B0 or MobileNet",
            "Reason": "Optimized for resource-constrained environments",
            "Implementation": "Use quantization and pruning"
        }
    }
    
    for scenario, rec in recommendations.items():
        print(f"\n{scenario}:")
        print(f"  Best Choice: {rec['Best Choice']}")
        print(f"  Reason: {rec['Reason']}")
        print(f"  Implementation: {rec['Implementation']}")


def current_industry_trends():
    """Current trends in industry"""
    print("\n" + "=" * 80)
    print("Current Industry Trends (2024)")
    print("=" * 80)
    
    trends = [
        {
            "Company": "Amazon",
            "Approach": "CLIP + ResNet ensemble",
            "Use Case": "Product search, visual similarity"
        },
        {
            "Company": "Alibaba/Taobao",
            "Approach": "ViT + custom fashion models",
            "Use Case": "Fashion recommendation, style matching"
        },
        {
            "Company": "Pinterest",
            "Approach": "CLIP + custom visual embeddings",
            "Use Case": "Visual discovery, style inspiration"
        },
        {
            "Company": "Google Shopping",
            "Approach": "Multi-model ensemble (CLIP + EfficientNet)",
            "Use Case": "Product understanding, price comparison"
        },
        {
            "Company": "Meta (Facebook Marketplace)",
            "Approach": "CLIP for cross-modal, ResNet for categories",
            "Use Case": "Item classification, similar item recommendation"
        }
    ]
    
    print(f"{'Company':<20} {'Approach':<35} {'Use Case'}")
    print("-" * 80)
    
    for trend in trends:
        print(f"{trend['Company']:<20} {trend['Approach']:<35} {trend['Use Case']}")
    
    print("\nKey Insights:")
    print("1. CLIP dominates cross-modal applications (text-image search)")
    print("2. ResNet still widely used in production for pure speed")
    print("3. ViT gaining adoption for high-accuracy visual tasks")
    print("4. Ensemble approaches common for critical applications")
    print("5. Custom fine-tuning on domain data is standard practice")


if __name__ == "__main__":
    # Run comprehensive comparison
    benchmark_models()
    recommend_for_ecommerce()
    current_industry_trends()