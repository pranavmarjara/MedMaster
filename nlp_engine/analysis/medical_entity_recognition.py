"""
VibeyBot Advanced Medical Named Entity Recognition System
Sophisticated medical NER for clinical text understanding and analysis
Integrated with VibeyBot intelligence for comprehensive medical entity extraction
"""
import os
import sys
import re
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from datetime import datetime
import numpy as np
from dataclasses import dataclass
from enum import Enum

# NLP and machine learning libraries
import spacy
from spacy import displacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import scispacy
import medspacy
from medspacy.ner import TargetRule
from medspacy.context import ConTextRule
from medspacy.section_detection import Sectionizer
from medspacy.postprocess import PostprocessingRule

# Transformers for advanced NER
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, 
    TokenClassificationPipeline, pipeline
)
import torch
import torch.nn as nn

# Medical knowledge bases
import umls_client
from pymetamap import MetaMap

# Data processing
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalEntityType(Enum):
    """Medical entity types for VibeyBot NER system"""
    MEDICATION = "MEDICATION"
    DOSAGE = "DOSAGE"
    FREQUENCY = "FREQUENCY"
    ROUTE = "ROUTE"
    DISEASE = "DISEASE"
    SYMPTOM = "SYMPTOM"
    ANATOMY = "ANATOMY"
    PROCEDURE = "PROCEDURE"
    LAB_TEST = "LAB_TEST"
    LAB_VALUE = "LAB_VALUE"
    VITAL_SIGN = "VITAL_SIGN"
    ALLERGY = "ALLERGY"
    FAMILY_HISTORY = "FAMILY_HISTORY"
    SOCIAL_HISTORY = "SOCIAL_HISTORY"
    TEMPORAL = "TEMPORAL"
    SEVERITY = "SEVERITY"
    CLINICAL_MODIFIER = "CLINICAL_MODIFIER"
    MEDICAL_DEVICE = "MEDICAL_DEVICE"
    TREATMENT = "TREATMENT"
    DIAGNOSIS = "DIAGNOSIS"

@dataclass
class MedicalEntity:
    """Medical entity data structure"""
    text: str
    label: MedicalEntityType
    start: int
    end: int
    confidence: float
    normalized_form: str
    cui: Optional[str] = None  # UMLS Concept Unique Identifier
    semantic_type: Optional[str] = None
    attributes: Optional[Dict[str, Any]] = None
    context_modifiers: Optional[List[str]] = None

@dataclass
class NERResult:
    """NER processing result"""
    text: str
    entities: List[MedicalEntity]
    processing_time: float
    model_version: str
    confidence_score: float
    sections_detected: List[Dict[str, Any]]

class VibeyMedicalNERSystem:
    """
    Advanced Medical Named Entity Recognition System for VibeyBot
    Combines multiple NER models and medical knowledge bases
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize NLP models
        self._initialize_nlp_models()
        
        # Initialize medical knowledge bases
        self._initialize_knowledge_bases()
        
        # Initialize medical pattern matching
        self._initialize_medical_patterns()
        
        # Initialize entity normalization
        self._initialize_entity_normalization()
        
        logger.info("VibeyBot Medical NER System initialized successfully")
    
    def _initialize_nlp_models(self):
        """Initialize various NLP models for medical NER"""
        # Primary medical model
        try:
            self.medical_nlp = spacy.load("en_core_sci_md")
            logger.info("Loaded SciSpaCy medical model")
        except OSError:
            self.medical_nlp = spacy.load("en_core_web_sm")
            logger.warning("SciSpaCy not available, using standard spaCy model")
        
        # Add medSpaCy components
        self.medical_nlp.add_pipe("medspacy_pyrush")
        self.medical_nlp.add_pipe("medspacy_target_matcher")
        self.medical_nlp.add_pipe("medspacy_context")
        self.medical_nlp.add_pipe("medspacy_sectionizer")
        
        # BioBERT for advanced medical NER
        try:
            self.biobert_tokenizer = AutoTokenizer.from_pretrained(
                "dmis-lab/biobert-base-cased-v1.1"
            )
            self.biobert_model = AutoModelForTokenClassification.from_pretrained(
                "dmis-lab/biobert-base-cased-v1.1"
            )
            self.biobert_pipeline = TokenClassificationPipeline(
                model=self.biobert_model,
                tokenizer=self.biobert_tokenizer,
                aggregation_strategy="simple"
            )
            logger.info("Loaded BioBERT for medical NER")
        except Exception as e:
            logger.warning(f"Could not load BioBERT: {e}")
            self.biobert_pipeline = None
        
        # Clinical BERT
        try:
            self.clinical_bert_pipeline = pipeline(
                "ner",
                model="emilyalsentzer/Bio_ClinicalBERT",
                tokenizer="emilyalsentzer/Bio_ClinicalBERT",
                aggregation_strategy="simple"
            )
            logger.info("Loaded Clinical BERT")
        except Exception as e:
            logger.warning(f"Could not load Clinical BERT: {e}")
            self.clinical_bert_pipeline = None
    
    def _initialize_knowledge_bases(self):
        """Initialize medical knowledge bases and ontologies"""
        # UMLS integration
        try:
            self.umls = umls_client.UmlsClient()
            logger.info("UMLS client initialized")
        except Exception as e:
            logger.warning(f"UMLS not available: {e}")
            self.umls = None
        
        # Medical vocabularies
        self.medical_vocabularies = {
            'medications': self._load_medication_vocabulary(),
            'diseases': self._load_disease_vocabulary(),
            'procedures': self._load_procedure_vocabulary(),
            'anatomy': self._load_anatomy_vocabulary(),
            'lab_tests': self._load_lab_test_vocabulary()
        }
        
        logger.info("Medical vocabularies loaded")
    
    def _initialize_medical_patterns(self):
        """Initialize medical pattern matching rules"""
        self.medical_patterns = {
            'medication_patterns': [
                r'\b(?:mg|mcg|g|ml|tablets?|caps?|units?)\b',
                r'\b\d+\s*(?:mg|mcg|g|ml|tablets?|caps?|units?)\b',
                r'\b(?:po|iv|im|sc|sl|pr|topical|inhaled)\b',
                r'\b(?:bid|tid|qid|qd|q\d+h|prn|hs|ac|pc)\b'
            ],
            'vital_signs_patterns': [
                r'\b(?:bp|blood\s+pressure)\s*:?\s*\d+/\d+',
                r'\b(?:hr|heart\s+rate|pulse)\s*:?\s*\d+',
                r'\b(?:temp|temperature)\s*:?\s*\d+\.?\d*',
                r'\b(?:rr|resp|respiratory\s+rate)\s*:?\s*\d+',
                r'\b(?:o2\s*sat|spo2|oxygen\s+saturation)\s*:?\s*\d+%?'
            ],
            'lab_value_patterns': [
                r'\b(?:glucose|blood\s+sugar)\s*:?\s*\d+\.?\d*\s*(?:mg/dl)?',
                r'\b(?:hemoglobin|hgb|hb)\s*:?\s*\d+\.?\d*\s*(?:g/dl)?',
                r'\b(?:wbc|white\s+blood\s+cells?)\s*:?\s*\d+\.?\d*',
                r'\b(?:creatinine)\s*:?\s*\d+\.?\d*\s*(?:mg/dl)?',
                r'\b(?:cholesterol)\s*:?\s*\d+\.?\d*\s*(?:mg/dl)?'
            ],
            'temporal_patterns': [
                r'\b(?:yesterday|today|tomorrow)\b',
                r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
                r'\b(?:last|this|next)\s+(?:week|month|year)\b',
                r'\b\d+\s+(?:days?|weeks?|months?|years?)\s+ago\b',
                r'\b(?:morning|afternoon|evening|night)\b'
            ],
            'severity_patterns': [
                r'\b(?:mild|moderate|severe|critical|life-threatening)\b',
                r'\b(?:1-10|mild-severe)\s+(?:pain|symptoms?)\b',
                r'\b(?:stable|improving|worsening|deteriorating)\b'
            ]
        }
        
        # Compile regex patterns
        self.compiled_patterns = {}
        for category, patterns in self.medical_patterns.items():
            self.compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    def _initialize_entity_normalization(self):
        """Initialize entity normalization mappings"""
        self.normalization_mappings = {
            'medications': {
                'asa': 'aspirin',
                'hctz': 'hydrochlorothiazide',
                'lisinopril': 'lisinopril',
                'metformin': 'metformin',
                'advil': 'ibuprofen',
                'tylenol': 'acetaminophen'
            },
            'diseases': {
                'dm': 'diabetes mellitus',
                'htn': 'hypertension',
                'mi': 'myocardial infarction',
                'copd': 'chronic obstructive pulmonary disease',
                'uti': 'urinary tract infection'
            },
            'lab_tests': {
                'cbc': 'complete blood count',
                'bmp': 'basic metabolic panel',
                'cmp': 'comprehensive metabolic panel',
                'pt/inr': 'prothrombin time/international normalized ratio',
                'ptt': 'partial thromboplastin time'
            }
        }
    
    def _load_medication_vocabulary(self) -> Set[str]:
        """Load comprehensive medication vocabulary"""
        medications = {
            # Common medications
            'aspirin', 'acetaminophen', 'ibuprofen', 'metformin', 'lisinopril',
            'amlodipine', 'atorvastatin', 'simvastatin', 'omeprazole', 'levothyroxine',
            'warfarin', 'clopidogrel', 'furosemide', 'insulin', 'prednisone',
            'albuterol', 'losartan', 'hydrochlorothiazide', 'gabapentin', 'tramadol',
            
            # Antibiotics
            'amoxicillin', 'azithromycin', 'ciprofloxacin', 'doxycycline', 'penicillin',
            'cephalexin', 'clindamycin', 'levofloxacin', 'trimethoprim', 'sulfamethoxazole',
            
            # Cardiac medications
            'metoprolol', 'carvedilol', 'diltiazem', 'verapamil', 'nifedipine',
            'enalapril', 'ramipril', 'candesartan', 'valsartan', 'spironolactone',
            
            # Psychiatric medications
            'sertraline', 'fluoxetine', 'paroxetine', 'citalopram', 'escitalopram',
            'venlafaxine', 'duloxetine', 'bupropion', 'mirtazapine', 'trazodone',
            
            # Pain medications
            'morphine', 'oxycodone', 'hydrocodone', 'codeine', 'fentanyl',
            'naproxen', 'celecoxib', 'diclofenac', 'meloxicam', 'ketorolac'
        }
        return medications
    
    def _load_disease_vocabulary(self) -> Set[str]:
        """Load comprehensive disease vocabulary"""
        diseases = {
            # Cardiovascular
            'myocardial infarction', 'angina', 'heart failure', 'arrhythmia',
            'atrial fibrillation', 'hypertension', 'hypotension', 'cardiomyopathy',
            
            # Respiratory
            'pneumonia', 'asthma', 'copd', 'bronchitis', 'pneumothorax',
            'pulmonary embolism', 'respiratory failure', 'lung cancer',
            
            # Endocrine
            'diabetes mellitus', 'hypothyroidism', 'hyperthyroidism', 'diabetes insipidus',
            'addison disease', 'cushing syndrome', 'metabolic syndrome',
            
            # Gastrointestinal
            'appendicitis', 'cholecystitis', 'pancreatitis', 'gastritis',
            'peptic ulcer', 'inflammatory bowel disease', 'cirrhosis',
            
            # Neurological
            'stroke', 'seizure', 'epilepsy', 'migraine', 'parkinson disease',
            'alzheimer disease', 'multiple sclerosis', 'neuropathy',
            
            # Infectious
            'sepsis', 'pneumonia', 'meningitis', 'endocarditis', 'cellulitis',
            'urinary tract infection', 'gastroenteritis', 'tuberculosis'
        }
        return diseases
    
    def _load_procedure_vocabulary(self) -> Set[str]:
        """Load comprehensive procedure vocabulary"""
        procedures = {
            # Cardiac procedures
            'echocardiogram', 'electrocardiogram', 'cardiac catheterization',
            'angioplasty', 'coronary artery bypass', 'pacemaker insertion',
            
            # Imaging
            'chest x-ray', 'ct scan', 'mri', 'ultrasound', 'pet scan',
            'mammogram', 'bone scan', 'nuclear stress test',
            
            # Laboratory
            'blood culture', 'urine culture', 'complete blood count',
            'basic metabolic panel', 'liver function tests', 'lipid panel',
            
            # Surgical
            'appendectomy', 'cholecystectomy', 'colonoscopy', 'endoscopy',
            'bronchoscopy', 'biopsy', 'thoracentesis', 'paracentesis'
        }
        return procedures
    
    def _load_anatomy_vocabulary(self) -> Set[str]:
        """Load comprehensive anatomy vocabulary"""
        anatomy = {
            # Cardiovascular
            'heart', 'aorta', 'pulmonary artery', 'coronary artery', 'ventricle',
            'atrium', 'mitral valve', 'aortic valve', 'tricuspid valve',
            
            # Respiratory
            'lung', 'bronchi', 'trachea', 'pleura', 'diaphragm', 'alveoli',
            
            # Gastrointestinal
            'stomach', 'liver', 'pancreas', 'gallbladder', 'intestine',
            'esophagus', 'duodenum', 'colon', 'rectum', 'appendix',
            
            # Neurological
            'brain', 'spinal cord', 'cerebellum', 'brainstem', 'cerebrum',
            'meninges', 'cranial nerves', 'peripheral nerves',
            
            # Musculoskeletal
            'bone', 'muscle', 'tendon', 'ligament', 'cartilage', 'joint'
        }
        return anatomy
    
    def _load_lab_test_vocabulary(self) -> Set[str]:
        """Load comprehensive lab test vocabulary"""
        lab_tests = {
            'complete blood count', 'basic metabolic panel', 'comprehensive metabolic panel',
            'liver function tests', 'lipid panel', 'thyroid function tests',
            'hemoglobin a1c', 'prothrombin time', 'partial thromboplastin time',
            'erythrocyte sedimentation rate', 'c-reactive protein', 'troponin',
            'brain natriuretic peptide', 'urinalysis', 'urine culture',
            'blood culture', 'arterial blood gas', 'vitamin d', 'vitamin b12'
        }
        return lab_tests
    
    def extract_medical_entities(self, text: str) -> NERResult:
        """
        Extract medical entities using multiple approaches
        
        Args:
            text: Input medical text
            
        Returns:
            NERResult containing all extracted entities
        """
        start_time = datetime.now()
        
        # Initialize result containers
        all_entities = []
        sections_detected = []
        
        # Method 1: SpaCy-based extraction
        spacy_entities = self._extract_with_spacy(text)
        all_entities.extend(spacy_entities)
        
        # Method 2: BioBERT extraction
        if self.biobert_pipeline:
            biobert_entities = self._extract_with_biobert(text)
            all_entities.extend(biobert_entities)
        
        # Method 3: Clinical BERT extraction
        if self.clinical_bert_pipeline:
            clinical_bert_entities = self._extract_with_clinical_bert(text)
            all_entities.extend(clinical_bert_entities)
        
        # Method 4: Pattern-based extraction
        pattern_entities = self._extract_with_patterns(text)
        all_entities.extend(pattern_entities)
        
        # Method 5: Dictionary-based extraction
        dictionary_entities = self._extract_with_dictionaries(text)
        all_entities.extend(dictionary_entities)
        
        # Resolve entity conflicts and merge overlapping entities
        resolved_entities = self._resolve_entity_conflicts(all_entities)
        
        # Normalize entities
        normalized_entities = self._normalize_entities(resolved_entities)
        
        # Add UMLS codes if available
        if self.umls:
            normalized_entities = self._add_umls_codes(normalized_entities)
        
        # Detect document sections
        sections_detected = self._detect_document_sections(text)
        
        # Calculate processing time and confidence
        processing_time = (datetime.now() - start_time).total_seconds()
        confidence_score = self._calculate_overall_confidence(normalized_entities)
        
        return NERResult(
            text=text,
            entities=normalized_entities,
            processing_time=processing_time,
            model_version="VibeyBot-NER-4.2.1",
            confidence_score=confidence_score,
            sections_detected=sections_detected
        )
    
    def _extract_with_spacy(self, text: str) -> List[MedicalEntity]:
        """Extract entities using spaCy medical model"""
        doc = self.medical_nlp(text)
        entities = []
        
        for ent in doc.ents:
            # Map spaCy labels to VibeyBot entity types
            entity_type = self._map_spacy_label_to_vibey_type(ent.label_)
            
            if entity_type:
                medical_entity = MedicalEntity(
                    text=ent.text,
                    label=entity_type,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.85,  # spaCy default confidence
                    normalized_form=ent.text.lower().strip(),
                    attributes={'spacy_label': ent.label_}
                )
                entities.append(medical_entity)
        
        return entities
    
    def _extract_with_biobert(self, text: str) -> List[MedicalEntity]:
        """Extract entities using BioBERT"""
        if not self.biobert_pipeline:
            return []
        
        try:
            results = self.biobert_pipeline(text)
            entities = []
            
            for result in results:
                entity_type = self._map_biobert_label_to_vibey_type(result['entity_group'])
                
                if entity_type:
                    medical_entity = MedicalEntity(
                        text=result['word'],
                        label=entity_type,
                        start=result['start'],
                        end=result['end'],
                        confidence=result['score'],
                        normalized_form=result['word'].lower().strip(),
                        attributes={'biobert_label': result['entity_group']}
                    )
                    entities.append(medical_entity)
            
            return entities
        except Exception as e:
            logger.warning(f"BioBERT extraction failed: {e}")
            return []
    
    def _extract_with_clinical_bert(self, text: str) -> List[MedicalEntity]:
        """Extract entities using Clinical BERT"""
        if not self.clinical_bert_pipeline:
            return []
        
        try:
            results = self.clinical_bert_pipeline(text)
            entities = []
            
            for result in results:
                entity_type = self._map_clinical_bert_label_to_vibey_type(result['entity_group'])
                
                if entity_type:
                    medical_entity = MedicalEntity(
                        text=result['word'],
                        label=entity_type,
                        start=result['start'],
                        end=result['end'],
                        confidence=result['score'],
                        normalized_form=result['word'].lower().strip(),
                        attributes={'clinical_bert_label': result['entity_group']}
                    )
                    entities.append(medical_entity)
            
            return entities
        except Exception as e:
            logger.warning(f"Clinical BERT extraction failed: {e}")
            return []
    
    def _extract_with_patterns(self, text: str) -> List[MedicalEntity]:
        """Extract entities using pattern matching"""
        entities = []
        
        for category, patterns in self.compiled_patterns.items():
            entity_type = self._map_pattern_category_to_vibey_type(category)
            
            if entity_type:
                for pattern in patterns:
                    for match in pattern.finditer(text):
                        medical_entity = MedicalEntity(
                            text=match.group(0),
                            label=entity_type,
                            start=match.start(),
                            end=match.end(),
                            confidence=0.75,  # Pattern-based confidence
                            normalized_form=match.group(0).lower().strip(),
                            attributes={'pattern_category': category}
                        )
                        entities.append(medical_entity)
        
        return entities
    
    def _extract_with_dictionaries(self, text: str) -> List[MedicalEntity]:
        """Extract entities using medical dictionaries"""
        entities = []
        text_lower = text.lower()
        
        for category, vocabulary in self.medical_vocabularies.items():
            entity_type = self._map_vocabulary_category_to_vibey_type(category)
            
            if entity_type:
                for term in vocabulary:
                    term_lower = term.lower()
                    start = 0
                    
                    while True:
                        pos = text_lower.find(term_lower, start)
                        if pos == -1:
                            break
                        
                        # Check word boundaries
                        if self._is_valid_word_boundary(text_lower, pos, pos + len(term_lower)):
                            medical_entity = MedicalEntity(
                                text=text[pos:pos + len(term_lower)],
                                label=entity_type,
                                start=pos,
                                end=pos + len(term_lower),
                                confidence=0.70,  # Dictionary-based confidence
                                normalized_form=term_lower,
                                attributes={'dictionary_category': category}
                            )
                            entities.append(medical_entity)
                        
                        start = pos + 1
        
        return entities
    
    def _map_spacy_label_to_vibey_type(self, label: str) -> Optional[MedicalEntityType]:
        """Map spaCy entity labels to VibeyBot entity types"""
        mapping = {
            'CHEMICAL': MedicalEntityType.MEDICATION,
            'DISEASE': MedicalEntityType.DISEASE,
            'SYMPTOM': MedicalEntityType.SYMPTOM,
            'ANATOMY': MedicalEntityType.ANATOMY,
            'PROCEDURE': MedicalEntityType.PROCEDURE,
            'LAB': MedicalEntityType.LAB_TEST,
            'MEDICATION': MedicalEntityType.MEDICATION,
            'DOSAGE': MedicalEntityType.DOSAGE
        }
        return mapping.get(label)
    
    def _map_biobert_label_to_vibey_type(self, label: str) -> Optional[MedicalEntityType]:
        """Map BioBERT entity labels to VibeyBot entity types"""
        mapping = {
            'CHEMICAL': MedicalEntityType.MEDICATION,
            'DISEASE': MedicalEntityType.DISEASE,
            'GENE': MedicalEntityType.ANATOMY,
            'PROTEIN': MedicalEntityType.LAB_TEST,
            'CELL': MedicalEntityType.ANATOMY,
            'TISSUE': MedicalEntityType.ANATOMY
        }
        return mapping.get(label)
    
    def _map_clinical_bert_label_to_vibey_type(self, label: str) -> Optional[MedicalEntityType]:
        """Map Clinical BERT entity labels to VibeyBot entity types"""
        mapping = {
            'PROBLEM': MedicalEntityType.DISEASE,
            'TREATMENT': MedicalEntityType.TREATMENT,
            'TEST': MedicalEntityType.LAB_TEST,
            'DRUG': MedicalEntityType.MEDICATION,
            'DOSAGE': MedicalEntityType.DOSAGE,
            'FREQUENCY': MedicalEntityType.FREQUENCY,
            'ROUTE': MedicalEntityType.ROUTE
        }
        return mapping.get(label)
    
    def _map_pattern_category_to_vibey_type(self, category: str) -> Optional[MedicalEntityType]:
        """Map pattern categories to VibeyBot entity types"""
        mapping = {
            'medication_patterns': MedicalEntityType.MEDICATION,
            'vital_signs_patterns': MedicalEntityType.VITAL_SIGN,
            'lab_value_patterns': MedicalEntityType.LAB_VALUE,
            'temporal_patterns': MedicalEntityType.TEMPORAL,
            'severity_patterns': MedicalEntityType.SEVERITY
        }
        return mapping.get(category)
    
    def _map_vocabulary_category_to_vibey_type(self, category: str) -> Optional[MedicalEntityType]:
        """Map vocabulary categories to VibeyBot entity types"""
        mapping = {
            'medications': MedicalEntityType.MEDICATION,
            'diseases': MedicalEntityType.DISEASE,
            'procedures': MedicalEntityType.PROCEDURE,
            'anatomy': MedicalEntityType.ANATOMY,
            'lab_tests': MedicalEntityType.LAB_TEST
        }
        return mapping.get(category)
    
    def _is_valid_word_boundary(self, text: str, start: int, end: int) -> bool:
        """Check if entity boundaries align with word boundaries"""
        if start > 0 and text[start - 1].isalnum():
            return False
        if end < len(text) and text[end].isalnum():
            return False
        return True
    
    def _resolve_entity_conflicts(self, entities: List[MedicalEntity]) -> List[MedicalEntity]:
        """Resolve overlapping and conflicting entities"""
        # Sort entities by start position
        sorted_entities = sorted(entities, key=lambda x: (x.start, x.end))
        
        resolved_entities = []
        current_entity = None
        
        for entity in sorted_entities:
            if current_entity is None:
                current_entity = entity
            elif entity.start >= current_entity.end:
                # No overlap, add current and move to next
                resolved_entities.append(current_entity)
                current_entity = entity
            else:
                # Overlapping entities - choose the one with higher confidence
                if entity.confidence > current_entity.confidence:
                    current_entity = entity
                # If same confidence, prefer longer entity
                elif (entity.confidence == current_entity.confidence and 
                      (entity.end - entity.start) > (current_entity.end - current_entity.start)):
                    current_entity = entity
        
        if current_entity:
            resolved_entities.append(current_entity)
        
        return resolved_entities
    
    def _normalize_entities(self, entities: List[MedicalEntity]) -> List[MedicalEntity]:
        """Normalize entity text using predefined mappings"""
        normalized_entities = []
        
        for entity in entities:
            normalized_form = entity.normalized_form
            
            # Apply normalization mappings
            for category, mappings in self.normalization_mappings.items():
                if normalized_form in mappings:
                    normalized_form = mappings[normalized_form]
                    break
            
            # Create new entity with normalized form
            normalized_entity = MedicalEntity(
                text=entity.text,
                label=entity.label,
                start=entity.start,
                end=entity.end,
                confidence=entity.confidence,
                normalized_form=normalized_form,
                cui=entity.cui,
                semantic_type=entity.semantic_type,
                attributes=entity.attributes,
                context_modifiers=entity.context_modifiers
            )
            
            normalized_entities.append(normalized_entity)
        
        return normalized_entities
    
    def _add_umls_codes(self, entities: List[MedicalEntity]) -> List[MedicalEntity]:
        """Add UMLS concept codes to entities"""
        if not self.umls:
            return entities
        
        enriched_entities = []
        
        for entity in entities:
            try:
                # Search UMLS for the normalized entity
                concepts = self.umls.search(entity.normalized_form, 
                                          search_type='words', 
                                          include='cui,preferredName')
                
                if concepts:
                    best_concept = concepts[0]  # Take the first (best) match
                    entity.cui = best_concept.get('cui')
                    entity.semantic_type = best_concept.get('semanticType')
            except Exception as e:
                logger.debug(f"UMLS lookup failed for '{entity.normalized_form}': {e}")
            
            enriched_entities.append(entity)
        
        return enriched_entities
    
    def _detect_document_sections(self, text: str) -> List[Dict[str, Any]]:
        """Detect document sections in medical text"""
        sections = []
        
        # Common medical document section headers
        section_patterns = {
            'chief_complaint': r'(?:chief\s+complaint|cc):\s*(.+?)(?=\n\n|\n[A-Z]|$)',
            'history_present_illness': r'(?:history\s+of\s+present\s+illness|hpi):\s*(.+?)(?=\n\n|\n[A-Z]|$)',
            'past_medical_history': r'(?:past\s+medical\s+history|pmh):\s*(.+?)(?=\n\n|\n[A-Z]|$)',
            'medications': r'(?:medications|meds):\s*(.+?)(?=\n\n|\n[A-Z]|$)',
            'allergies': r'(?:allergies|nkda):\s*(.+?)(?=\n\n|\n[A-Z]|$)',
            'physical_exam': r'(?:physical\s+exam|pe):\s*(.+?)(?=\n\n|\n[A-Z]|$)',
            'assessment_plan': r'(?:assessment\s+and\s+plan|a&p|assessment|plan):\s*(.+?)(?=\n\n|\n[A-Z]|$)'
        }
        
        for section_name, pattern in section_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                sections.append({
                    'section_name': section_name,
                    'content': match.group(1).strip(),
                    'start': match.start(),
                    'end': match.end()
                })
        
        return sections
    
    def _calculate_overall_confidence(self, entities: List[MedicalEntity]) -> float:
        """Calculate overall confidence score for the NER results"""
        if not entities:
            return 0.0
        
        total_confidence = sum(entity.confidence for entity in entities)
        return total_confidence / len(entities)

def process_medical_text_for_vibey(text: str, config: Dict = None) -> NERResult:
    """
    Main function to process medical text for VibeyBot NER
    
    Args:
        text: Input medical text
        config: Processing configuration
        
    Returns:
        NER processing results
    """
    ner_system = VibeyMedicalNERSystem(config)
    return ner_system.extract_medical_entities(text)

if __name__ == "__main__":
    # Example usage
    sample_medical_text = """
    Chief Complaint: 68-year-old male with chest pain
    
    History of Present Illness:
    Patient presents with severe chest pain that started 2 hours ago while mowing the lawn.
    Pain is described as crushing, 8/10 severity, radiating to left arm.
    Associated with diaphoresis and nausea.
    
    Past Medical History:
    Diabetes mellitus type 2, hypertension, hyperlipidemia
    
    Medications:
    Metformin 1000mg twice daily
    Lisinopril 10mg daily
    Atorvastatin 40mg daily
    
    Physical Exam:
    BP: 160/95, HR: 110, Temp: 98.6F, RR: 20, O2 Sat: 96%
    
    Lab Results:
    Troponin I: 2.1 ng/mL (elevated)
    Glucose: 180 mg/dL
    Creatinine: 1.2 mg/dL
    """
    
    # Process the medical text
    result = process_medical_text_for_vibey(sample_medical_text)
    
    print(f"VibeyBot Medical NER Results:")
    print(f"Processing Time: {result.processing_time:.3f} seconds")
    print(f"Overall Confidence: {result.confidence_score:.3f}")
    print(f"Entities Found: {len(result.entities)}")
    print(f"Sections Detected: {len(result.sections_detected)}")
    
    print("\nExtracted Entities:")
    for entity in result.entities[:10]:  # Show first 10 entities
        print(f"  {entity.text} ({entity.label.value}) - Confidence: {entity.confidence:.3f}")
    
    print("\nDocument Sections:")
    for section in result.sections_detected:
        print(f"  {section['section_name']}: {section['content'][:50]}...")