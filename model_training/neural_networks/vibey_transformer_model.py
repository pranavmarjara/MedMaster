"""
VibeyBot Advanced Transformer Architecture
State-of-the-art transformer model for medical language understanding
Specialized architecture for clinical text analysis and medical reasoning
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import AutoModel, AutoTokenizer, BertModel, BertConfig
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime
from dataclasses import dataclass
import json
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VibeyTransformerConfig:
    """Configuration for VibeyBot Transformer model"""
    # Model architecture
    vocab_size: int = 50000
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 1024
    
    # Medical-specific parameters
    medical_vocab_size: int = 5000
    clinical_context_size: int = 256
    diagnostic_heads: int = 8
    risk_assessment_layers: int = 4
    
    # Training parameters
    dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02
    
    # Task-specific parameters
    num_disease_classes: int = 1000
    num_symptom_classes: int = 500
    num_medication_classes: int = 2000
    num_procedure_classes: int = 800
    
    # Medical reasoning parameters
    reasoning_depth: int = 6
    evidence_fusion_dim: int = 512
    confidence_estimation_layers: int = 3

class VibeyPositionalEncoding(nn.Module):
    """Advanced positional encoding for medical sequences"""
    
    def __init__(self, d_model: int, max_len: int = 1024, temperature: int = 10000):
        super(VibeyPositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(temperature) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]

class VibeyMedicalAttention(nn.Module):
    """Medical-specialized multi-head attention mechanism"""
    
    def __init__(self, config: VibeyTransformerConfig):
        super(VibeyMedicalAttention, self).__init__()
        
        self.config = config
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = config.hidden_size // config.num_attention_heads
        
        # Standard attention components
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Medical-specific attention components
        self.medical_query = nn.Linear(config.hidden_size, config.medical_vocab_size)
        self.medical_key = nn.Linear(config.hidden_size, config.medical_vocab_size)
        self.clinical_context_proj = nn.Linear(config.hidden_size, config.clinical_context_size)
        
        # Attention fusion
        self.attention_fusion = nn.Linear(config.hidden_size + config.medical_vocab_size, config.hidden_size)
        
        # Dropout
        self.attention_dropout = nn.Dropout(config.attention_dropout_prob)
        self.output_dropout = nn.Dropout(config.dropout_prob)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                medical_context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        batch_size, seq_length, hidden_size = hidden_states.size()
        
        # Standard multi-head attention
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)
        
        # Reshape for multi-head attention
        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_size)
        
        # Apply attention mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Medical attention enhancement
        if medical_context is not None:
            medical_query = self.medical_query(hidden_states)
            medical_key = self.medical_key(medical_context)
            
            medical_scores = torch.matmul(medical_query, medical_key.transpose(-1, -2))
            medical_scores = F.softmax(medical_scores, dim=-1)
            
            # Combine standard and medical attention
            attention_scores = attention_scores + 0.3 * medical_scores.unsqueeze(1)
        
        # Apply softmax to get attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)
        
        # Apply attention to value layer
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Project clinical context if available
        if medical_context is not None:
            clinical_context = self.clinical_context_proj(context_layer)
            context_layer = self.attention_fusion(torch.cat([context_layer, clinical_context], dim=-1))
        
        # Apply output dropout and layer norm
        context_layer = self.output_dropout(context_layer)
        context_layer = self.layer_norm(context_layer + hidden_states)
        
        return context_layer, attention_probs
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

class VibeyMedicalFeedForward(nn.Module):
    """Medical-enhanced feed-forward network"""
    
    def __init__(self, config: VibeyTransformerConfig):
        super(VibeyMedicalFeedForward, self).__init__()
        
        # Standard feed-forward layers
        self.dense_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        
        # Medical knowledge integration
        self.medical_projection = nn.Linear(config.hidden_size, config.medical_vocab_size)
        self.medical_dense = nn.Linear(config.medical_vocab_size, config.intermediate_size // 2)
        
        # Activation and normalization
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Standard feed-forward path
        intermediate = self.dense_1(hidden_states)
        intermediate = self.activation(intermediate)
        
        # Medical knowledge path
        medical_projection = self.medical_projection(hidden_states)
        medical_projection = F.relu(medical_projection)
        medical_intermediate = self.medical_dense(medical_projection)
        
        # Combine paths
        combined_intermediate = torch.cat([
            intermediate[:, :, :config.intermediate_size // 2],
            medical_intermediate
        ], dim=-1)
        
        # Final projection
        output = self.dense_2(combined_intermediate)
        output = self.dropout(output)
        
        # Residual connection and layer norm
        return self.layer_norm(output + hidden_states)

class VibeyTransformerLayer(nn.Module):
    """Complete Transformer layer with medical enhancements"""
    
    def __init__(self, config: VibeyTransformerConfig):
        super(VibeyTransformerLayer, self).__init__()
        
        self.attention = VibeyMedicalAttention(config)
        self.feed_forward = VibeyMedicalFeedForward(config)
        
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                medical_context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Self-attention with medical enhancement
        attention_output, attention_probs = self.attention(
            hidden_states, attention_mask, medical_context
        )
        
        # Feed-forward
        layer_output = self.feed_forward(attention_output)
        
        return layer_output, attention_probs

class VibeyMedicalReasoningModule(nn.Module):
    """Advanced medical reasoning and inference module"""
    
    def __init__(self, config: VibeyTransformerConfig):
        super(VibeyMedicalReasoningModule, self).__init__()
        
        self.config = config
        
        # Multi-step reasoning layers
        self.reasoning_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size, config.evidence_fusion_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout_prob),
                nn.Linear(config.evidence_fusion_dim, config.hidden_size)
            ) for _ in range(config.reasoning_depth)
        ])
        
        # Evidence fusion network
        self.evidence_fusion = nn.Sequential(
            nn.Linear(config.hidden_size * 3, config.evidence_fusion_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_prob),
            nn.Linear(config.evidence_fusion_dim, config.hidden_size)
        )
        
        # Diagnostic prediction heads
        self.disease_classifier = nn.Linear(config.hidden_size, config.num_disease_classes)
        self.symptom_classifier = nn.Linear(config.hidden_size, config.num_symptom_classes)
        self.medication_classifier = nn.Linear(config.hidden_size, config.num_medication_classes)
        self.procedure_classifier = nn.Linear(config.hidden_size, config.num_procedure_classes)
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(config.hidden_size if i == 0 else config.evidence_fusion_dim, 
                         config.evidence_fusion_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout_prob)
            ) for i in range(config.confidence_estimation_layers)],
            nn.Linear(config.evidence_fusion_dim, 1),
            nn.Sigmoid()
        )
        
        # Risk assessment module
        self.risk_assessor = nn.Sequential(
            nn.Linear(config.hidden_size, config.evidence_fusion_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_prob),
            nn.Linear(config.evidence_fusion_dim, config.evidence_fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_prob),
            nn.Linear(config.evidence_fusion_dim // 2, 6)  # 6 risk levels
        )
        
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        # Multi-step medical reasoning
        reasoning_state = hidden_states
        for reasoning_layer in self.reasoning_layers:
            residual = reasoning_state
            reasoning_state = reasoning_layer(reasoning_state)
            reasoning_state = reasoning_state + 0.3 * residual  # Residual connection
        
        # Pool different representations for evidence fusion
        pooled_hidden = torch.mean(hidden_states, dim=1)  # Average pooling
        max_pooled = torch.max(hidden_states, dim=1)[0]   # Max pooling
        attention_weighted = torch.sum(hidden_states * attention_weights.mean(dim=1).unsqueeze(-1), dim=1)
        
        # Fuse evidence from different pooling strategies
        fused_evidence = self.evidence_fusion(
            torch.cat([pooled_hidden, max_pooled, attention_weighted], dim=-1)
        )
        
        # Generate predictions
        predictions = {
            'disease_logits': self.disease_classifier(fused_evidence),
            'symptom_logits': self.symptom_classifier(fused_evidence),
            'medication_logits': self.medication_classifier(fused_evidence),
            'procedure_logits': self.procedure_classifier(fused_evidence),
            'confidence_scores': self.confidence_estimator(fused_evidence),
            'risk_assessment': self.risk_assessor(fused_evidence),
            'reasoning_embeddings': reasoning_state,
            'fused_evidence': fused_evidence
        }
        
        return predictions

class VibeyTransformerModel(nn.Module):
    """
    VibeyBot Advanced Transformer Model for Medical Language Understanding
    Combines state-of-the-art transformer architecture with medical domain expertise
    """
    
    def __init__(self, config: VibeyTransformerConfig):
        super(VibeyTransformerModel, self).__init__()
        
        self.config = config
        
        # Embedding layers
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = VibeyPositionalEncoding(
            config.hidden_size, config.max_position_embeddings
        )
        self.medical_embeddings = nn.Embedding(config.medical_vocab_size, config.hidden_size)
        
        # Embedding normalization and dropout
        self.embeddings_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.embeddings_dropout = nn.Dropout(config.dropout_prob)
        
        # Transformer layers
        self.encoder_layers = nn.ModuleList([
            VibeyTransformerLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Medical reasoning module
        self.medical_reasoning = VibeyMedicalReasoningModule(config)
        
        # Final layer normalization
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initialize model weights"""
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        
        self.apply(_init_weights)
    
    def get_extended_attention_mask(self, 
                                  attention_mask: torch.Tensor,
                                  input_shape: Tuple[int, ...]) -> torch.Tensor:
        """Create extended attention mask for transformer layers"""
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Wrong shape for attention_mask: {attention_mask.shape}")
        
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        return extended_attention_mask
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                medical_context_ids: Optional[torch.Tensor] = None,
                return_attention_weights: bool = False,
                return_all_layers: bool = False) -> Dict[str, torch.Tensor]:
        
        batch_size, seq_length = input_ids.size()
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=input_ids.device)
        
        # Get extended attention mask
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_ids.shape
        )
        
        # Embeddings
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(word_embeddings.transpose(0, 1)).transpose(0, 1)
        
        embeddings = word_embeddings + position_embeddings
        
        # Add medical context embeddings if provided
        medical_context = None
        if medical_context_ids is not None:
            medical_context = self.medical_embeddings(medical_context_ids)
        
        # Apply layer norm and dropout to embeddings
        embeddings = self.embeddings_layer_norm(embeddings)
        hidden_states = self.embeddings_dropout(embeddings)
        
        # Forward through transformer layers
        all_hidden_states = []
        all_attention_weights = []
        
        for i, layer_module in enumerate(self.encoder_layers):
            if return_all_layers:
                all_hidden_states.append(hidden_states)
            
            layer_outputs = layer_module(
                hidden_states, extended_attention_mask, medical_context
            )
            hidden_states, attention_weights = layer_outputs
            
            if return_attention_weights:
                all_attention_weights.append(attention_weights)
        
        # Final layer normalization
        hidden_states = self.final_layer_norm(hidden_states)
        
        if return_all_layers:
            all_hidden_states.append(hidden_states)
        
        # Medical reasoning and predictions
        avg_attention_weights = torch.stack(all_attention_weights).mean(dim=0) if all_attention_weights else None
        medical_predictions = self.medical_reasoning(hidden_states, avg_attention_weights)
        
        # Prepare outputs
        outputs = {
            'last_hidden_state': hidden_states,
            'medical_predictions': medical_predictions,
            'pooled_representation': torch.mean(hidden_states, dim=1)
        }
        
        if return_all_layers:
            outputs['all_hidden_states'] = all_hidden_states
        
        if return_attention_weights:
            outputs['attention_weights'] = all_attention_weights
        
        return outputs
    
    def get_medical_embeddings(self, medical_terms: List[str]) -> torch.Tensor:
        """Get embeddings for medical terms"""
        # This would typically use a medical term to ID mapping
        # For now, we'll create dummy IDs
        medical_ids = torch.tensor([hash(term) % self.config.medical_vocab_size for term in medical_terms])
        return self.medical_embeddings(medical_ids)
    
    def predict_medical_concepts(self, 
                                input_text: str, 
                                tokenizer,
                                top_k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """Predict medical concepts from input text"""
        # Tokenize input
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.forward(inputs['input_ids'], inputs['attention_mask'])
        
        predictions = outputs['medical_predictions']
        
        # Get top predictions for each category
        results = {}
        
        categories = ['disease', 'symptom', 'medication', 'procedure']
        for category in categories:
            logits = predictions[f'{category}_logits']
            probs = F.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, top_k, dim=-1)
            
            # Convert to readable format (would need actual vocabulary mapping)
            results[category] = [
                (f"{category}_{idx.item()}", prob.item()) 
                for idx, prob in zip(top_indices[0], top_probs[0])
            ]
        
        # Add confidence and risk scores
        results['confidence'] = predictions['confidence_scores'].item()
        results['risk_level'] = torch.argmax(predictions['risk_assessment'], dim=-1).item()
        
        return results
    
    def save_pretrained(self, save_directory: str):
        """Save model and configuration"""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save(self.state_dict(), save_directory / 'pytorch_model.bin')
        
        # Save configuration
        with open(save_directory / 'config.json', 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        # Save model info
        model_info = {
            'model_type': 'VibeyTransformerModel',
            'version': '4.2.1',
            'creation_timestamp': datetime.now().isoformat(),
            'parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
        
        with open(save_directory / 'model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"VibeyBot Transformer model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, model_directory: str):
        """Load model from directory"""
        model_directory = Path(model_directory)
        
        # Load configuration
        with open(model_directory / 'config.json', 'r') as f:
            config_dict = json.load(f)
        config = VibeyTransformerConfig(**config_dict)
        
        # Create model
        model = cls(config)
        
        # Load weights
        state_dict = torch.load(model_directory / 'pytorch_model.bin', map_location='cpu')
        model.load_state_dict(state_dict)
        
        logger.info(f"VibeyBot Transformer model loaded from {model_directory}")
        return model

class VibeyTransformerTrainer:
    """Training pipeline for VibeyBot Transformer model"""
    
    def __init__(self, model: VibeyTransformerModel, config: Dict = None):
        self.model = model
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training components
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.get('learning_rate', 2e-5),
            weight_decay=0.01
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=5,
            factor=0.5
        )
        
        # Loss functions
        self.classification_criterion = nn.CrossEntropyLoss()
        self.confidence_criterion = nn.MSELoss()
        self.risk_criterion = nn.CrossEntropyLoss()
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # Forward pass
        outputs = self.model(input_ids, attention_mask)
        predictions = outputs['medical_predictions']
        
        # Calculate losses
        losses = {}
        total_loss = 0
        
        # Disease classification loss
        if 'disease_labels' in batch:
            disease_labels = batch['disease_labels'].to(self.device)
            disease_loss = self.classification_criterion(
                predictions['disease_logits'], disease_labels
            )
            losses['disease_loss'] = disease_loss.item()
            total_loss += disease_loss
        
        # Symptom classification loss
        if 'symptom_labels' in batch:
            symptom_labels = batch['symptom_labels'].to(self.device)
            symptom_loss = self.classification_criterion(
                predictions['symptom_logits'], symptom_labels
            )
            losses['symptom_loss'] = symptom_loss.item()
            total_loss += symptom_loss
        
        # Confidence estimation loss
        if 'confidence_targets' in batch:
            confidence_targets = batch['confidence_targets'].to(self.device)
            confidence_loss = self.confidence_criterion(
                predictions['confidence_scores'], confidence_targets
            )
            losses['confidence_loss'] = confidence_loss.item()
            total_loss += 0.5 * confidence_loss
        
        # Risk assessment loss
        if 'risk_labels' in batch:
            risk_labels = batch['risk_labels'].to(self.device)
            risk_loss = self.risk_criterion(
                predictions['risk_assessment'], risk_labels
            )
            losses['risk_loss'] = risk_loss.item()
            total_loss += 0.3 * risk_loss
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        losses['total_loss'] = total_loss.item()
        return losses

def initialize_vibey_transformer(config: VibeyTransformerConfig = None) -> VibeyTransformerModel:
    """Initialize VibeyBot Transformer model"""
    if config is None:
        config = VibeyTransformerConfig()
    
    model = VibeyTransformerModel(config)
    
    logger.info("VibeyBot Transformer Model initialized")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Hidden size: {config.hidden_size}")
    logger.info(f"Number of layers: {config.num_hidden_layers}")
    logger.info(f"Number of attention heads: {config.num_attention_heads}")
    
    return model

if __name__ == "__main__":
    # Example usage and testing
    config = VibeyTransformerConfig(
        vocab_size=30000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12
    )
    
    # Initialize model
    model = initialize_vibey_transformer(config)
    
    # Create dummy input
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))
    
    # Forward pass
    outputs = model(input_ids, attention_mask, return_attention_weights=True)
    
    print("VibeyBot Transformer Model Test:")
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {outputs['last_hidden_state'].shape}")
    print(f"Medical predictions keys: {list(outputs['medical_predictions'].keys())}")
    print(f"Disease predictions shape: {outputs['medical_predictions']['disease_logits'].shape}")
    
    logger.info("VibeyBot Transformer model test completed successfully")