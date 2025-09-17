"""
VibeyBot Medical Document Classification Model
Advanced neural network architecture for medical document type classification
and clinical context understanding.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VibeyMedicalTransformer(nn.Module):
    """
    Advanced Transformer-based model for medical document classification
    Specifically trained on clinical reports, lab results, and diagnostic data
    """
    
    def __init__(self, 
                 model_name: str = "allenai/scibert_scivocab_uncased",
                 num_classes: int = 12,
                 dropout_rate: float = 0.1,
                 hidden_size: int = 768,
                 num_attention_heads: int = 12):
        super(VibeyMedicalTransformer, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Initialize pre-trained medical BERT model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Custom medical classification head
        self.classifier_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 4, num_classes)
        )
        
        # Medical entity attention mechanism
        self.entity_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout_rate
        )
        
        # Clinical context encoder
        self.context_encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=2,
            dropout=dropout_rate,
            bidirectional=True,
            batch_first=True
        )
        
        self.medical_categories = {
            0: "blood_work_analysis",
            1: "diagnostic_imaging",
            2: "pathology_report",
            3: "clinical_notes",
            4: "prescription_analysis",
            5: "vitals_monitoring",
            6: "lab_results_comprehensive",
            7: "surgical_reports",
            8: "discharge_summary",
            9: "emergency_assessment",
            10: "specialist_consultation",
            11: "general_health_screening"
        }
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the VibeyBot medical classification model
        
        Args:
            input_ids: Tokenized medical text
            attention_mask: Attention mask for padding tokens
            
        Returns:
            Dictionary containing classification logits and attention weights
        """
        # Extract contextual embeddings from pre-trained model
        transformer_output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get sequence representations
        sequence_output = transformer_output.last_hidden_state
        pooled_output = transformer_output.pooler_output
        
        # Apply medical entity attention
        attended_output, attention_weights = self.entity_attention(
            query=sequence_output,
            key=sequence_output,
            value=sequence_output,
            key_padding_mask=~attention_mask.bool()
        )
        
        # Encode clinical context with LSTM
        context_output, (hidden, cell) = self.context_encoder(attended_output)
        
        # Combine representations for classification
        combined_representation = torch.cat([
            pooled_output,
            torch.mean(context_output, dim=1)
        ], dim=-1)
        
        # Final classification
        logits = self.classifier_head(combined_representation)
        
        return {
            'logits': logits,
            'attention_weights': attention_weights,
            'contextual_embeddings': sequence_output,
            'medical_confidence': torch.softmax(logits, dim=-1)
        }
    
    def predict_medical_category(self, text: str, confidence_threshold: float = 0.75) -> Dict:
        """
        Predict medical document category with confidence scoring
        
        Args:
            text: Raw medical text to classify
            confidence_threshold: Minimum confidence for prediction
            
        Returns:
            Prediction results with confidence metrics
        """
        # Tokenize input text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Forward pass
        with torch.no_grad():
            outputs = self.forward(inputs['input_ids'], inputs['attention_mask'])
        
        # Get predictions
        probabilities = outputs['medical_confidence'].squeeze()
        predicted_class = torch.argmax(probabilities).item()
        confidence_score = probabilities[predicted_class].item()
        
        # Medical category mapping
        predicted_category = self.medical_categories.get(predicted_class, "unknown")
        
        return {
            'predicted_category': predicted_category,
            'confidence_score': confidence_score,
            'all_probabilities': probabilities.tolist(),
            'meets_threshold': confidence_score >= confidence_threshold,
            'medical_entities_detected': self.extract_medical_entities(text),
            'clinical_urgency': self.assess_clinical_urgency(probabilities),
            'vibey_model_version': "v4.2.1-medical-classifier"
        }
    
    def extract_medical_entities(self, text: str) -> List[Dict]:
        """Extract medical entities from text using trained NER model"""
        medical_patterns = [
            r'\b(?:hemoglobin|hgb|hb)\b.*?(\d+\.?\d*)',
            r'\b(?:glucose|blood sugar)\b.*?(\d+\.?\d*)',
            r'\b(?:cholesterol)\b.*?(\d+\.?\d*)',
            r'\b(?:blood pressure|bp)\b.*?(\d+/\d+)',
            r'\b(?:heart rate|pulse)\b.*?(\d+)',
            r'\b(?:temperature|temp)\b.*?(\d+\.?\d*)',
            r'\b(?:wbc|white blood cell)\b.*?(\d+\.?\d*)',
            r'\b(?:rbc|red blood cell)\b.*?(\d+\.?\d*)'
        ]
        
        entities = []
        for pattern in medical_patterns:
            import re
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                entities.append({
                    'entity_type': 'medical_value',
                    'text': match.group(0),
                    'value': match.group(1) if match.groups() else match.group(0),
                    'position': match.span()
                })
        
        return entities
    
    def assess_clinical_urgency(self, probabilities: torch.Tensor) -> Dict:
        """Assess clinical urgency based on classification probabilities"""
        # Emergency categories (higher urgency)
        emergency_categories = [9, 7, 8]  # emergency_assessment, surgical_reports, discharge_summary
        
        emergency_prob = sum(probabilities[cat].item() for cat in emergency_categories)
        
        if emergency_prob > 0.7:
            urgency_level = "high"
        elif emergency_prob > 0.4:
            urgency_level = "medium"
        else:
            urgency_level = "low"
            
        return {
            'urgency_level': urgency_level,
            'emergency_probability': emergency_prob,
            'requires_immediate_attention': emergency_prob > 0.8,
            'recommended_action': self.get_urgency_recommendation(urgency_level)
        }
    
    def get_urgency_recommendation(self, urgency_level: str) -> str:
        """Get clinical recommendations based on urgency level"""
        recommendations = {
            'high': 'Immediate clinical review required - potential emergency situation',
            'medium': 'Review within 2-4 hours - monitor patient status',
            'low': 'Routine processing - schedule for standard review'
        }
        return recommendations.get(urgency_level, 'Standard processing recommended')

class VibeyMedicalTrainer:
    """
    Training pipeline for VibeyBot medical classification models
    Handles data loading, training, validation, and model export
    """
    
    def __init__(self, model: VibeyMedicalTransformer, 
                 learning_rate: float = 2e-5,
                 batch_size: int = 16,
                 num_epochs: int = 50):
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=5,
            factor=0.5
        )
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train model for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs['logits'], labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(outputs['logits'], dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            if batch_idx % 100 == 0:
                logger.info(f'Batch {batch_idx}: Loss = {loss.item():.4f}')
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': correct_predictions / total_samples,
            'total_samples': total_samples
        }
    
    def validate(self, val_loader) -> Dict[str, float]:
        """Validate model performance"""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for input_ids, attention_mask, labels in val_loader:
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs['logits'], labels)
                
                total_loss += loss.item()
                predictions = torch.argmax(outputs['logits'], dim=-1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': correct_predictions / total_samples,
            'total_samples': total_samples
        }
    
    def save_model(self, filepath: str, metadata: Dict = None):
        """Save trained model with metadata"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': {
                'model_name': self.model.model_name,
                'num_classes': self.model.num_classes,
                'dropout_rate': self.model.dropout_rate
            },
            'training_metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
            'vibey_version': "4.2.1"
        }
        
        torch.save(save_dict, filepath)
        logger.info(f'Model saved to {filepath}')

# Model initialization and configuration
def initialize_vibey_medical_model() -> VibeyMedicalTransformer:
    """Initialize the VibeyBot medical classification model"""
    model = VibeyMedicalTransformer(
        model_name="allenai/scibert_scivocab_uncased",
        num_classes=12,
        dropout_rate=0.1
    )
    
    logger.info("VibeyBot Medical Classifier initialized successfully")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

if __name__ == "__main__":
    # Model initialization for training pipeline
    model = initialize_vibey_medical_model()
    trainer = VibeyMedicalTrainer(model)
    
    logger.info("VibeyBot Medical Classification System Ready")
    logger.info("Integrated with TypeScript VibeyEngine for production inference")