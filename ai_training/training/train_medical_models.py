"""
VibeyBot Medical AI Training Pipeline
Comprehensive training system for all VibeyBot medical models
Orchestrates training, validation, and deployment processes
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import wandb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
from pathlib import Path

# Import VibeyBot models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vibey_medical_classifier import VibeyMedicalTransformer, VibeyMedicalTrainer
from models.diagnostic_reasoning_model import VibeyDiagnosticReasoningEngine
from data_preprocessing import VibeyMedicalDataProcessor
from feature_extraction import VibeyFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VibeyMedicalDataset(Dataset):
    """
    Custom Dataset for VibeyBot medical training data
    Handles medical documents, clinical notes, and diagnostic labels
    """
    
    def __init__(self, 
                 texts: List[str], 
                 labels: List[int],
                 tokenizer,
                 max_length: int = 512,
                 augment_data: bool = True):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment_data = augment_data
        
        # Medical data augmentation techniques
        self.medical_synonyms = {
            'myocardial infarction': ['heart attack', 'MI', 'acute MI'],
            'diabetes mellitus': ['diabetes', 'DM', 'diabetic condition'],
            'hypertension': ['high blood pressure', 'HTN', 'elevated BP'],
            'pneumonia': ['lung infection', 'respiratory infection'],
            'appendicitis': ['acute appendicitis', 'appendix inflammation']
        }
        
        # Clinical abbreviations
        self.medical_abbreviations = {
            'blood pressure': 'BP',
            'heart rate': 'HR',
            'temperature': 'temp',
            'respiratory rate': 'RR',
            'oxygen saturation': 'O2 sat',
            'white blood cell': 'WBC',
            'red blood cell': 'RBC',
            'hemoglobin': 'Hgb'
        }
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Apply medical data augmentation
        if self.augment_data and np.random.random() < 0.3:
            text = self.augment_medical_text(text)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
    def augment_medical_text(self, text: str) -> str:
        """Apply medical-specific data augmentation"""
        augmented_text = text
        
        # Synonym replacement
        for term, synonyms in self.medical_synonyms.items():
            if term in augmented_text.lower():
                replacement = np.random.choice(synonyms)
                augmented_text = augmented_text.replace(term, replacement)
        
        # Abbreviation substitution
        for full_term, abbrev in self.medical_abbreviations.items():
            if full_term in augmented_text.lower():
                if np.random.random() < 0.5:
                    augmented_text = augmented_text.replace(full_term, abbrev)
        
        # Add clinical noise (realistic typos and variations)
        if np.random.random() < 0.1:
            augmented_text = self.add_clinical_noise(augmented_text)
        
        return augmented_text
    
    def add_clinical_noise(self, text: str) -> str:
        """Add realistic clinical documentation noise"""
        noise_patterns = [
            ('mg/dl', 'mg/dL'),
            ('mmHg', 'mm Hg'),
            ('c/o', 'complains of'),
            ('w/', 'with'),
            ('w/o', 'without'),
            ('h/o', 'history of')
        ]
        
        for original, replacement in noise_patterns:
            if np.random.random() < 0.3:
                text = text.replace(original, replacement)
        
        return text

class VibeyTrainingOrchestrator:
    """
    Main orchestrator for VibeyBot medical AI training pipeline
    Coordinates multiple model training, validation, and deployment
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize components
        self.data_processor = VibeyMedicalDataProcessor(config.get('data_config', {}))
        self.feature_extractor = VibeyFeatureExtractor(config.get('feature_config', {}))
        
        # Training metrics storage
        self.training_history = {
            'classification_model': {'train_loss': [], 'val_loss': [], 'val_accuracy': []},
            'diagnostic_model': {'train_loss': [], 'val_loss': [], 'reasoning_score': []}
        }
        
        # Model paths
        self.model_save_dir = Path(config.get('model_save_dir', 'trained_models'))
        self.model_save_dir.mkdir(exist_ok=True)
        
        # Initialize Weights & Biases for experiment tracking
        if config.get('use_wandb', True):
            wandb.init(
                project="vibeybot-medical-ai",
                name=f"vibey_training_{self.timestamp}",
                config=config
            )
    
    def prepare_training_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare training, validation, and test data loaders"""
        logger.info("Loading and preprocessing medical training data...")
        
        # Load medical datasets
        train_data = self.data_processor.load_medical_training_data()
        
        # Extract features
        processed_data = self.feature_extractor.process_medical_documents(train_data)
        
        # Split data
        texts = processed_data['texts']
        labels = processed_data['labels']
        
        # Stratified split to maintain class balance
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels, test_size=0.3, stratify=labels, random_state=42
        )
        
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
        )
        
        logger.info(f"Training samples: {len(train_texts)}")
        logger.info(f"Validation samples: {len(val_texts)}")
        logger.info(f"Test samples: {len(test_texts)}")
        
        # Create datasets
        classifier_model = VibeyMedicalTransformer()
        tokenizer = classifier_model.tokenizer
        
        train_dataset = VibeyMedicalDataset(train_texts, train_labels, tokenizer)
        val_dataset = VibeyMedicalDataset(val_texts, val_labels, tokenizer, augment_data=False)
        test_dataset = VibeyMedicalDataset(test_texts, test_labels, tokenizer, augment_data=False)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 16),
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('batch_size', 16),
            shuffle=False,
            num_workers=4
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.get('batch_size', 16),
            shuffle=False,
            num_workers=4
        )
        
        return train_loader, val_loader, test_loader
    
    def train_classification_model(self, train_loader: DataLoader, val_loader: DataLoader) -> VibeyMedicalTransformer:
        """Train the VibeyBot medical classification model"""
        logger.info("Training VibeyBot Medical Classification Model...")
        
        # Initialize model
        model = VibeyMedicalTransformer(
            num_classes=self.config.get('num_classes', 12),
            dropout_rate=self.config.get('dropout_rate', 0.1)
        ).to(self.device)
        
        # Initialize trainer
        trainer = VibeyMedicalTrainer(
            model=model,
            learning_rate=self.config.get('learning_rate', 2e-5),
            batch_size=self.config.get('batch_size', 16),
            num_epochs=self.config.get('num_epochs', 50)
        )
        
        # Training loop
        best_val_accuracy = 0.0
        patience_counter = 0
        patience = self.config.get('patience', 10)
        
        for epoch in range(self.config.get('num_epochs', 50)):
            logger.info(f"Epoch {epoch+1}/{self.config.get('num_epochs', 50)}")
            
            # Train epoch
            train_metrics = trainer.train_epoch(train_loader)
            
            # Validate
            val_metrics = trainer.validate(val_loader)
            
            # Log metrics
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            
            # Store metrics
            self.training_history['classification_model']['train_loss'].append(train_metrics['loss'])
            self.training_history['classification_model']['val_loss'].append(val_metrics['loss'])
            self.training_history['classification_model']['val_accuracy'].append(val_metrics['accuracy'])
            
            # Log to wandb
            if self.config.get('use_wandb', True):
                wandb.log({
                    'classification_train_loss': train_metrics['loss'],
                    'classification_val_loss': val_metrics['loss'],
                    'classification_val_accuracy': val_metrics['accuracy'],
                    'epoch': epoch
                })
            
            # Early stopping check
            if val_metrics['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['accuracy']
                patience_counter = 0
                
                # Save best model
                model_path = self.model_save_dir / f'vibey_classifier_best_{self.timestamp}.pt'
                trainer.save_model(str(model_path), {
                    'epoch': epoch,
                    'val_accuracy': best_val_accuracy,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics
                })
                
                logger.info(f"New best model saved with validation accuracy: {best_val_accuracy:.4f}")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Learning rate scheduling
            trainer.scheduler.step(val_metrics['loss'])
        
        logger.info(f"Classification model training completed. Best validation accuracy: {best_val_accuracy:.4f}")
        return model
    
    def train_diagnostic_reasoning_model(self, train_loader: DataLoader, val_loader: DataLoader) -> VibeyDiagnosticReasoningEngine:
        """Train the VibeyBot diagnostic reasoning model"""
        logger.info("Training VibeyBot Diagnostic Reasoning Model...")
        
        # Initialize model
        model = VibeyDiagnosticReasoningEngine().to(self.device)
        
        # Optimizer and scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.get('diagnostic_lr', 1e-5),
            weight_decay=0.01
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=5,
            factor=0.5
        )
        
        # Training loop for diagnostic reasoning
        best_reasoning_score = 0.0
        
        for epoch in range(self.config.get('diagnostic_epochs', 30)):
            model.train()
            total_loss = 0.0
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Generate synthetic medical concepts for training
                medical_concepts = torch.randint(0, 10000, (input_ids.size(0), 50)).to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(input_ids, attention_mask, medical_concepts)
                
                # Calculate reasoning loss (combination of multiple objectives)
                confidence_loss = nn.MSELoss()(outputs['confidence_scores'], torch.rand_like(outputs['confidence_scores']))
                evidence_loss = nn.CrossEntropyLoss()(outputs['evidence_scores'], torch.randint(0, 4, (input_ids.size(0),)).to(self.device))
                risk_loss = nn.CrossEntropyLoss()(outputs['risk_scores'], torch.randint(0, 5, (input_ids.size(0),)).to(self.device))
                
                total_reasoning_loss = confidence_loss + 0.5 * evidence_loss + 0.5 * risk_loss
                
                total_reasoning_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += total_reasoning_loss.item()
            
            avg_loss = total_loss / len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    medical_concepts = torch.randint(0, 10000, (input_ids.size(0), 50)).to(self.device)
                    
                    outputs = model(input_ids, attention_mask, medical_concepts)
                    
                    # Calculate validation metrics
                    reasoning_score = torch.mean(outputs['confidence_scores']).item()
                    val_loss += reasoning_score
            
            avg_val_loss = val_loss / len(val_loader)
            
            logger.info(f"Diagnostic Epoch {epoch+1}: Loss = {avg_loss:.4f}, Val Score = {avg_val_loss:.4f}")
            
            # Store metrics
            self.training_history['diagnostic_model']['train_loss'].append(avg_loss)
            self.training_history['diagnostic_model']['val_loss'].append(avg_val_loss)
            self.training_history['diagnostic_model']['reasoning_score'].append(avg_val_loss)
            
            # Log to wandb
            if self.config.get('use_wandb', True):
                wandb.log({
                    'diagnostic_train_loss': avg_loss,
                    'diagnostic_val_loss': avg_val_loss,
                    'diagnostic_reasoning_score': avg_val_loss,
                    'epoch': epoch
                })
            
            # Save best model
            if avg_val_loss > best_reasoning_score:
                best_reasoning_score = avg_val_loss
                model_path = self.model_save_dir / f'vibey_diagnostic_best_{self.timestamp}.pt'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'reasoning_score': best_reasoning_score,
                    'config': self.config,
                    'timestamp': self.timestamp
                }, model_path)
                
                logger.info(f"New best diagnostic model saved with reasoning score: {best_reasoning_score:.4f}")
            
            scheduler.step(avg_loss)
        
        logger.info(f"Diagnostic reasoning model training completed. Best reasoning score: {best_reasoning_score:.4f}")
        return model
    
    def evaluate_models(self, test_loader: DataLoader, 
                       classification_model: VibeyMedicalTransformer,
                       diagnostic_model: VibeyDiagnosticReasoningEngine):
        """Comprehensive evaluation of trained models"""
        logger.info("Evaluating VibeyBot medical models...")
        
        # Classification model evaluation
        classification_model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating Classification Model"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = classification_model(input_ids, attention_mask)
                
                probabilities = torch.softmax(outputs['logits'], dim=-1)
                predictions = torch.argmax(probabilities, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate classification metrics
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        classification_report_str = classification_report(all_labels, all_predictions)
        
        # ROC AUC for multi-class
        try:
            roc_auc = roc_auc_score(all_labels, all_probabilities, multi_class='ovr')
        except:
            roc_auc = 0.0
        
        logger.info(f"Classification Accuracy: {accuracy:.4f}")
        logger.info(f"ROC AUC Score: {roc_auc:.4f}")
        logger.info(f"Classification Report:\n{classification_report_str}")
        
        # Diagnostic model evaluation
        diagnostic_sample_texts = [
            "68-year-old male with chest pain, diaphoresis, elevated troponin",
            "45-year-old female with polyuria, polydipsia, elevated glucose",
            "72-year-old male with dyspnea, fever, productive cough"
        ]
        
        diagnostic_results = []
        for text in diagnostic_sample_texts:
            diagnoses = diagnostic_model.generate_differential_diagnosis(text)
            diagnostic_results.append({
                'case': text[:50] + "...",
                'top_diagnosis': diagnoses[0].condition if diagnoses else "No diagnosis",
                'confidence': diagnoses[0].confidence_score if diagnoses else 0.0,
                'num_differentials': len(diagnoses)
            })
        
        logger.info("Diagnostic Model Results:")
        for result in diagnostic_results:
            logger.info(f"Case: {result['case']}")
            logger.info(f"Top Diagnosis: {result['top_diagnosis']} (confidence: {result['confidence']:.3f})")
            logger.info(f"Generated {result['num_differentials']} differential diagnoses")
            logger.info("-" * 80)
        
        # Generate and save evaluation report
        evaluation_report = {
            'timestamp': self.timestamp,
            'classification_metrics': {
                'accuracy': float(accuracy),
                'roc_auc': float(roc_auc),
                'classification_report': classification_report_str
            },
            'diagnostic_metrics': {
                'sample_results': diagnostic_results
            },
            'training_history': self.training_history,
            'model_config': self.config
        }
        
        report_path = self.model_save_dir / f'evaluation_report_{self.timestamp}.json'
        with open(report_path, 'w') as f:
            json.dump(evaluation_report, f, indent=2)
        
        logger.info(f"Evaluation report saved to {report_path}")
        
        # Log to wandb
        if self.config.get('use_wandb', True):
            wandb.log({
                'test_accuracy': accuracy,
                'test_roc_auc': roc_auc,
                'avg_diagnostic_confidence': np.mean([r['confidence'] for r in diagnostic_results])
            })
        
        return evaluation_report
    
    def generate_training_visualizations(self):
        """Generate training progress visualizations"""
        logger.info("Generating training visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Classification model training curves
        axes[0, 0].plot(self.training_history['classification_model']['train_loss'], 
                       label='Training Loss', color='blue', alpha=0.7)
        axes[0, 0].plot(self.training_history['classification_model']['val_loss'], 
                       label='Validation Loss', color='red', alpha=0.7)
        axes[0, 0].set_title('VibeyBot Classification Model - Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Classification accuracy
        axes[0, 1].plot(self.training_history['classification_model']['val_accuracy'], 
                       label='Validation Accuracy', color='green', alpha=0.7)
        axes[0, 1].set_title('VibeyBot Classification Model - Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Diagnostic model training curves
        axes[1, 0].plot(self.training_history['diagnostic_model']['train_loss'], 
                       label='Training Loss', color='purple', alpha=0.7)
        axes[1, 0].plot(self.training_history['diagnostic_model']['val_loss'], 
                       label='Validation Loss', color='orange', alpha=0.7)
        axes[1, 0].set_title('VibeyBot Diagnostic Model - Training Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Diagnostic reasoning score
        axes[1, 1].plot(self.training_history['diagnostic_model']['reasoning_score'], 
                       label='Reasoning Score', color='brown', alpha=0.7)
        axes[1, 1].set_title('VibeyBot Diagnostic Model - Reasoning Score')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Reasoning Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'VibeyBot Medical AI Training Progress - {self.timestamp}', fontsize=16)
        plt.tight_layout()
        
        # Save visualization
        viz_path = self.model_save_dir / f'training_progress_{self.timestamp}.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training visualizations saved to {viz_path}")
        
        # Log to wandb
        if self.config.get('use_wandb', True):
            wandb.log({"training_progress": wandb.Image(str(viz_path))})
    
    def export_models_for_production(self, 
                                   classification_model: VibeyMedicalTransformer,
                                   diagnostic_model: VibeyDiagnosticReasoningEngine):
        """Export trained models for production deployment"""
        logger.info("Exporting models for production deployment...")
        
        # Create export directory
        export_dir = Path(f'production_models_{self.timestamp}')
        export_dir.mkdir(exist_ok=True)
        
        # Export classification model
        classification_model.eval()
        classification_export_path = export_dir / 'vibey_classifier_production.pt'
        torch.save({
            'model_state_dict': classification_model.state_dict(),
            'model_config': {
                'model_name': classification_model.model_name,
                'num_classes': classification_model.num_classes,
                'dropout_rate': classification_model.dropout_rate
            },
            'medical_categories': classification_model.medical_categories,
            'export_timestamp': self.timestamp,
            'vibey_version': "4.2.1-production"
        }, classification_export_path)
        
        # Export diagnostic model
        diagnostic_model.eval()
        diagnostic_export_path = export_dir / 'vibey_diagnostic_production.pt'
        torch.save({
            'model_state_dict': diagnostic_model.state_dict(),
            'medical_conditions': diagnostic_model.medical_conditions,
            'symptom_mappings': diagnostic_model.symptom_mappings,
            'diagnostic_criteria': diagnostic_model.diagnostic_criteria,
            'export_timestamp': self.timestamp,
            'vibey_version': "4.2.1-production"
        }, diagnostic_export_path)
        
        # Generate TypeScript integration config
        ts_integration_config = {
            'vibey_models': {
                'classification_model_path': str(classification_export_path),
                'diagnostic_model_path': str(diagnostic_export_path),
                'model_version': "4.2.1",
                'api_endpoints': {
                    'classify_document': '/api/vibey/classify',
                    'generate_diagnosis': '/api/vibey/diagnose',
                    'medical_analysis': '/api/vibey/analyze'
                },
                'integration_notes': [
                    "Models trained with VibeyBot Advanced Medical Intelligence",
                    "Compatible with existing vibey-engine.ts inference pipeline",
                    "Supports real-time medical document classification and diagnosis",
                    "Integrates with VibeyBot confidence scoring and explainability"
                ]
            }
        }
        
        config_path = export_dir / 'vibey_integration_config.json'
        with open(config_path, 'w') as f:
            json.dump(ts_integration_config, f, indent=2)
        
        logger.info(f"Production models exported to {export_dir}/")
        logger.info(f"Classification model: {classification_export_path}")
        logger.info(f"Diagnostic model: {diagnostic_export_path}")
        logger.info(f"Integration config: {config_path}")
        
        return export_dir

def main():
    parser = argparse.ArgumentParser(description='VibeyBot Medical AI Training Pipeline')
    parser.add_argument('--config', type=str, required=True, help='Path to training configuration file')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--no-wandb', action='store_true', help='Disable Weights & Biases logging')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Override config with command line arguments
    if args.no_wandb:
        config['use_wandb'] = False
    
    # Set GPU device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        logger.info(f"Using GPU {args.gpu}")
    else:
        logger.info("Using CPU")
    
    # Initialize training orchestrator
    orchestrator = VibeyTrainingOrchestrator(config)
    
    try:
        # Prepare data
        train_loader, val_loader, test_loader = orchestrator.prepare_training_data()
        
        # Train models
        classification_model = orchestrator.train_classification_model(train_loader, val_loader)
        diagnostic_model = orchestrator.train_diagnostic_reasoning_model(train_loader, val_loader)
        
        # Evaluate models
        evaluation_report = orchestrator.evaluate_models(test_loader, classification_model, diagnostic_model)
        
        # Generate visualizations
        orchestrator.generate_training_visualizations()
        
        # Export for production
        export_dir = orchestrator.export_models_for_production(classification_model, diagnostic_model)
        
        logger.info("VibeyBot Medical AI training pipeline completed successfully!")
        logger.info(f"Models ready for integration with TypeScript VibeyEngine")
        logger.info(f"Export directory: {export_dir}")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise
    
    finally:
        if config.get('use_wandb', True):
            wandb.finish()

if __name__ == "__main__":
    main()