#!/usr/bin/env python3
"""
Text + Image Multimodal Recommendation System
Author: Yang Liu
"""

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class MultimodalItem:
    item_id: str
    title: str
    description: str
    image_path: str
    category_id: int
    brand_id: int
    price: float


class TextFeatureExtractor(nn.Module):
    """BERT text feature extraction"""
    
    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def forward(self, text_list: List[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            text_list, padding=True, truncation=True, 
            max_length=128, return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = self.bert(**encoded)
            return outputs.last_hidden_state[:, 0, :]


class ImageFeatureExtractor(nn.Module):
    """CLIP image feature extraction"""
    
    def __init__(self):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        self.model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, images: List[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        
        with torch.no_grad():
            return self.model.get_image_features(**inputs)


class TextImageFusion(nn.Module):
    """Text-Image cross-modal fusion"""
    
    def __init__(self, text_dim=768, image_dim=512, hidden_dim=256):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, text_features, image_features):
        text_proj = self.text_proj(text_features)
        image_proj = self.image_proj(image_features)
        
        multimodal = torch.stack([text_proj, image_proj], dim=0)
        attended, _ = self.attention(multimodal, multimodal, multimodal)
        
        combined = attended.transpose(0, 1).flatten(1)
        return self.output_proj(combined)


class MultimodalDeepFM(nn.Module):
    """Multimodal DeepFM model"""
    
    def __init__(self, structured_dims: Dict[str, int]):
        super().__init__()
        self.text_extractor = TextFeatureExtractor()
        self.image_extractor = ImageFeatureExtractor()
        self.fusion = TextImageFusion()
        
        # Structured embeddings
        self.embeddings = nn.ModuleDict()
        embedding_dim = 32
        for name, vocab_size in structured_dims.items():
            self.embeddings[name] = nn.Embedding(vocab_size, embedding_dim)
        
        # FM + DNN
        total_dim = len(structured_dims) * embedding_dim + 256
        self.fm_first = nn.Linear(total_dim, 1)
        self.fm_second = nn.Parameter(torch.randn(total_dim, embedding_dim))
        
        self.dnn = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
        self.output = nn.Sigmoid()
    
    def forward(self, batch_data):
        # Extract multimodal features
        text_features = self.text_extractor(batch_data['text'])
        
        images = []
        for path in batch_data['image_paths']:
            try:
                img = Image.open(path).convert('RGB')
            except:
                img = Image.new('RGB', (224, 224), color='white')
            images.append(img)
        
        image_features = self.image_extractor(images)
        multimodal_features = self.fusion(text_features, image_features)
        
        # Structured features
        structured_features = []
        for name, values in batch_data['structured'].items():
            if name in self.embeddings:
                emb = self.embeddings[name](torch.LongTensor(values))
                structured_features.append(emb)
        
        structured_concat = torch.cat(structured_features, dim=1)
        all_features = torch.cat([structured_concat, multimodal_features], dim=1)
        
        # FM
        fm_first = self.fm_first(all_features)
        fm_embeddings = torch.matmul(all_features, self.fm_second)
        square_sum = torch.pow(torch.sum(fm_embeddings, dim=1, keepdim=True), 2)
        sum_square = torch.sum(torch.pow(fm_embeddings, 2), dim=1, keepdim=True)
        fm_second = 0.5 * (square_sum - sum_square)
        
        # DNN
        dnn_out = self.dnn(all_features)
        
        final_output = fm_first + fm_second.sum(dim=1, keepdim=True) + dnn_out
        return self.output(final_output)


if __name__ == "__main__":
    print("Text + Image Multimodal System Created")
    print("Features: BERT text + CLIP image + Cross-modal fusion")