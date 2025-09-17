"""
VibeyBot Medical Data Preprocessing Pipeline
Advanced data preprocessing for medical AI training datasets
Handles medical document cleaning, normalization, and feature extraction
"""
import os
import sys
import re
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass
from collections import Counter, defaultdict
import multiprocessing as mp
from functools import partial

# NLP and text processing
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import textstat
import re

# Machine learning and data processing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Medical text processing
import medspacy
from medspacy.ner import TargetRule
from medspacy.context import ConTextRule

# Data validation
import pandas_profiling
from great_expectations import DataContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

@dataclass
class PreprocessingConfig:
    """Configuration for VibeyBot data preprocessing"""
    remove_phi: bool = True
    normalize_medical_terms: bool = True
    extract_structured_data: bool = True
    min_text_length: int = 50
    max_text_length: int = 10000
    remove_duplicates: bool = True
    handle_missing_values: bool = True
    standardize_dates: bool = True
    normalize_lab_values: bool = True
    extract_medications: bool = True
    extract_diagnoses: bool = True
    quality_score_threshold: float = 0.6

class VibeyMedicalDataProcessor:
    """
    Advanced medical data preprocessing pipeline for VibeyBot AI training
    Handles various medical data types and ensures HIPAA compliance
    """
    
    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or PreprocessingConfig()
        
        # Initialize NLP components
        self._initialize_nlp_components()
        
        # Initialize medical vocabularies and mappings
        self._initialize_medical_mappings()
        
        # Initialize data validation components
        self._initialize_data_validation()
        
        # PHI detection patterns
        self._initialize_phi_patterns()
        
        # Statistics tracking
        self.processing_stats = {
            'documents_processed': 0,
            'phi_instances_removed': 0,
            'duplicates_removed': 0,
            'quality_rejections': 0,
            'processing_time': 0.0
        }
        
        logger.info("VibeyBot Medical Data Processor initialized")
    
    def _initialize_nlp_components(self):
        """Initialize NLP models and components"""
        try:
            self.nlp = spacy.load("en_core_sci_md")
            logger.info("Loaded SciSpaCy medical model")
        except OSError:
            self.nlp = spacy.load("en_core_web_sm")
            logger.warning("Using standard spaCy model")
        
        # Initialize medical NLP components
        self.nlp.add_pipe("medspacy_pyrush")
        self.nlp.add_pipe("medspacy_target_matcher")
        self.nlp.add_pipe("medspacy_context")
        
        # Text processing components
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Medical stop words (additional medical-specific stop words)
        self.medical_stop_words = {
            'patient', 'history', 'examination', 'assessment', 'plan', 'follow',
            'continue', 'return', 'clinic', 'hospital', 'medical', 'doctor',
            'physician', 'nurse', 'provider', 'visit', 'appointment'
        }
        self.stop_words.update(self.medical_stop_words)
    
    def _initialize_medical_mappings(self):
        """Initialize medical term normalization mappings"""
        self.medical_abbreviations = {
            # Vitals and measurements
            'bp': 'blood pressure', 'hr': 'heart rate', 'rr': 'respiratory rate',
            'temp': 'temperature', 'wt': 'weight', 'ht': 'height', 'bmi': 'body mass index',
            
            # Medical conditions
            'dm': 'diabetes mellitus', 'htn': 'hypertension', 'mi': 'myocardial infarction',
            'copd': 'chronic obstructive pulmonary disease', 'uti': 'urinary tract infection',
            'ckd': 'chronic kidney disease', 'chf': 'congestive heart failure',
            
            # Laboratory tests
            'cbc': 'complete blood count', 'bmp': 'basic metabolic panel',
            'cmp': 'comprehensive metabolic panel', 'lft': 'liver function tests',
            'tft': 'thyroid function tests', 'pt': 'prothrombin time',
            'ptt': 'partial thromboplastin time', 'inr': 'international normalized ratio',
            
            # Medications
            'asa': 'aspirin', 'hctz': 'hydrochlorothiazide', 'ace': 'angiotensin converting enzyme',
            'arb': 'angiotensin receptor blocker', 'nsaid': 'nonsteroidal anti-inflammatory drug',
            
            # Frequency and dosing
            'bid': 'twice daily', 'tid': 'three times daily', 'qid': 'four times daily',
            'qd': 'once daily', 'qhs': 'at bedtime', 'prn': 'as needed',
            'ac': 'before meals', 'pc': 'after meals'
        }
        
        # Unit standardization
        self.unit_standardization = {
            'mg/dl': 'mg/dL', 'mg/deciliter': 'mg/dL', 'milligram/deciliter': 'mg/dL',
            'g/dl': 'g/dL', 'g/deciliter': 'g/dL', 'gram/deciliter': 'g/dL',
            'mmol/l': 'mmol/L', 'millimole/liter': 'mmol/L',
            'miu/l': 'mIU/L', 'microinternational unit/liter': 'mIU/L',
            'ng/ml': 'ng/mL', 'nanogram/milliliter': 'ng/mL',
            'pg/ml': 'pg/mL', 'picogram/milliliter': 'pg/mL',
            'u/l': 'U/L', 'unit/liter': 'U/L',
            'iu/l': 'IU/L', 'international unit/liter': 'IU/L'
        }
        
        # Medical value ranges (for normalization and outlier detection)
        self.normal_ranges = {
            'glucose': {'min': 70, 'max': 100, 'unit': 'mg/dL'},
            'hemoglobin': {'min': 12, 'max': 16, 'unit': 'g/dL'},
            'hematocrit': {'min': 36, 'max': 46, 'unit': '%'},
            'wbc': {'min': 4.5, 'max': 11.0, 'unit': 'K/uL'},
            'platelet': {'min': 150, 'max': 450, 'unit': 'K/uL'},
            'creatinine': {'min': 0.6, 'max': 1.2, 'unit': 'mg/dL'},
            'bun': {'min': 7, 'max': 20, 'unit': 'mg/dL'},
            'total_cholesterol': {'min': 0, 'max': 200, 'unit': 'mg/dL'},
            'hdl_cholesterol': {'min': 40, 'max': 60, 'unit': 'mg/dL'},
            'ldl_cholesterol': {'min': 0, 'max': 100, 'unit': 'mg/dL'},
            'triglycerides': {'min': 0, 'max': 150, 'unit': 'mg/dL'},
            'hba1c': {'min': 4.0, 'max': 5.6, 'unit': '%'},
            'tsh': {'min': 0.4, 'max': 4.0, 'unit': 'mIU/L'}
        }
    
    def _initialize_data_validation(self):
        """Initialize data validation components"""
        # Quality metrics thresholds
        self.quality_thresholds = {
            'min_readability_score': 30,
            'max_readability_score': 100,
            'min_unique_words': 10,
            'max_repetition_ratio': 0.3,
            'min_sentence_variety': 0.2
        }
        
        # Data completeness requirements
        self.required_fields = {
            'patient_demographics': ['age', 'gender'],
            'clinical_note': ['text_content', 'note_type', 'date'],
            'lab_result': ['test_name', 'value', 'unit', 'date'],
            'medication': ['name', 'dosage', 'frequency']
        }
    
    def _initialize_phi_patterns(self):
        """Initialize PHI (Protected Health Information) detection patterns"""
        # Comprehensive PHI detection patterns
        self.phi_patterns = {
            'ssn': [
                r'\b\d{3}-\d{2}-\d{4}\b',
                r'\b\d{3}\s\d{2}\s\d{4}\b',
                r'\b\d{9}\b'
            ],
            'phone': [
                r'\b\d{3}-\d{3}-\d{4}\b',
                r'\b\(\d{3}\)\s?\d{3}-\d{4}\b',
                r'\b\d{3}\.\d{3}\.\d{4}\b'
            ],
            'email': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            'dates': [
                r'\b\d{1,2}/\d{1,2}/\d{4}\b',
                r'\b\d{1,2}-\d{1,2}-\d{4}\b',
                r'\b\d{4}/\d{1,2}/\d{1,2}\b',
                r'\b\d{4}-\d{1,2}-\d{1,2}\b'
            ],
            'medical_record_numbers': [
                r'\b(?:mrn|medical record|record number)[\s:]*(\d{6,})\b',
                r'\b(?:patient id|pat id)[\s:]*(\d{6,})\b'
            ],
            'names': [
                # Common name patterns in medical records
                r'\b(?:mr|mrs|ms|dr|doctor|patient)\s+[A-Z][a-z]+\b',
                r'\b[A-Z][a-z]+,\s+[A-Z][a-z]+\b'  # Last, First format
            ],
            'addresses': [
                r'\b\d+\s+[A-Za-z\s]+(?:street|st|avenue|ave|road|rd|drive|dr|lane|ln|boulevard|blvd)\b',
                r'\b[A-Za-z\s]+,\s+[A-Z]{2}\s+\d{5}(?:-\d{4})?\b'  # City, State ZIP
            ]
        }
        
        # Compile regex patterns for efficiency
        self.compiled_phi_patterns = {}
        for category, patterns in self.phi_patterns.items():
            self.compiled_phi_patterns[category] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    def remove_phi(self, text: str) -> Tuple[str, int]:
        """Remove PHI from medical text while preserving clinical content"""
        cleaned_text = text
        phi_removed_count = 0
        
        # Apply PHI removal patterns
        for category, patterns in self.compiled_phi_patterns.items():
            for pattern in patterns:
                matches = list(pattern.finditer(cleaned_text))
                for match in matches:
                    # Replace with appropriate placeholder
                    if category == 'ssn':
                        replacement = '[SSN]'
                    elif category == 'phone':
                        replacement = '[PHONE]'
                    elif category == 'email':
                        replacement = '[EMAIL]'
                    elif category == 'dates':
                        replacement = '[DATE]'
                    elif category == 'medical_record_numbers':
                        replacement = '[MRN]'
                    elif category == 'names':
                        replacement = '[NAME]'
                    elif category == 'addresses':
                        replacement = '[ADDRESS]'
                    else:
                        replacement = '[REDACTED]'
                    
                    cleaned_text = cleaned_text.replace(match.group(), replacement)
                    phi_removed_count += 1
        
        return cleaned_text, phi_removed_count
    
    def normalize_medical_text(self, text: str) -> str:
        """Normalize medical text using medical vocabulary mappings"""
        normalized_text = text.lower()
        
        # Expand medical abbreviations
        for abbrev, expansion in self.medical_abbreviations.items():
            # Use word boundaries to avoid partial replacements
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            normalized_text = re.sub(pattern, expansion, normalized_text, flags=re.IGNORECASE)
        
        # Standardize units
        for variant, standard in self.unit_standardization.items():
            pattern = r'\b' + re.escape(variant) + r'\b'
            normalized_text = re.sub(pattern, standard, normalized_text, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        normalized_text = re.sub(r'\s+', ' ', normalized_text).strip()
        
        return normalized_text
    
    def extract_structured_data(self, text: str) -> Dict[str, Any]:
        """Extract structured medical data from text"""
        structured_data = {
            'medications': [],
            'lab_values': [],
            'vital_signs': [],
            'diagnoses': [],
            'procedures': [],
            'symptoms': []
        }
        
        # Extract medications
        medication_patterns = [
            r'(\w+(?:\s+\w+)*)\s+(\d+\.?\d*)\s*(mg|mcg|g|ml|units?)\s+(daily|bid|tid|qid|prn)',
            r'(?:medication|med|drug)[\s:]*(\w+(?:\s+\w+)*)',
            r'(?:prescribed|taking|on)[\s:]*(\w+(?:\s+\w+)*)'
        ]
        
        for pattern in medication_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 4:  # Full medication info
                    structured_data['medications'].append({
                        'name': match.group(1),
                        'dosage': match.group(2),
                        'unit': match.group(3),
                        'frequency': match.group(4)
                    })
                else:  # Just medication name
                    structured_data['medications'].append({
                        'name': match.group(1),
                        'dosage': None,
                        'unit': None,
                        'frequency': None
                    })
        
        # Extract lab values
        lab_patterns = [
            r'(?:glucose|blood sugar)[\s:]*(\d+\.?\d*)\s*(?:mg/dl)?',
            r'(?:hemoglobin|hgb|hb)[\s:]*(\d+\.?\d*)\s*(?:g/dl)?',
            r'(?:creatinine)[\s:]*(\d+\.?\d*)\s*(?:mg/dl)?',
            r'(?:cholesterol)[\s:]*(\d+\.?\d*)\s*(?:mg/dl)?',
            r'(?:wbc|white blood cell)[\s:]*(\d+\.?\d*)'
        ]
        
        for pattern in lab_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                test_name = self._extract_test_name_from_pattern(pattern)
                structured_data['lab_values'].append({
                    'test': test_name,
                    'value': float(match.group(1)),
                    'unit': self._get_default_unit(test_name)
                })
        
        # Extract vital signs
        vital_patterns = [
            r'(?:bp|blood pressure)[\s:]*(\d+)/(\d+)',
            r'(?:hr|heart rate|pulse)[\s:]*(\d+)',
            r'(?:temp|temperature)[\s:]*(\d+\.?\d*)',
            r'(?:rr|respiratory rate)[\s:]*(\d+)',
            r'(?:o2 sat|spo2)[\s:]*(\d+)%?'
        ]
        
        for pattern in vital_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                vital_name = self._extract_vital_name_from_pattern(pattern)
                if 'blood pressure' in vital_name.lower() and len(match.groups()) >= 2:
                    structured_data['vital_signs'].append({
                        'type': 'blood_pressure',
                        'systolic': int(match.group(1)),
                        'diastolic': int(match.group(2))
                    })
                else:
                    structured_data['vital_signs'].append({
                        'type': vital_name,
                        'value': float(match.group(1))
                    })
        
        return structured_data
    
    def _extract_test_name_from_pattern(self, pattern: str) -> str:
        """Extract test name from regex pattern"""
        pattern_to_name = {
            'glucose': 'glucose',
            'blood sugar': 'glucose',
            'hemoglobin': 'hemoglobin',
            'hgb': 'hemoglobin',
            'hb': 'hemoglobin',
            'creatinine': 'creatinine',
            'cholesterol': 'cholesterol',
            'wbc': 'white_blood_cell_count',
            'white blood cell': 'white_blood_cell_count'
        }
        
        for key, value in pattern_to_name.items():
            if key in pattern.lower():
                return value
        
        return 'unknown'
    
    def _extract_vital_name_from_pattern(self, pattern: str) -> str:
        """Extract vital sign name from regex pattern"""
        pattern_to_name = {
            'bp': 'blood_pressure',
            'blood pressure': 'blood_pressure',
            'hr': 'heart_rate',
            'heart rate': 'heart_rate',
            'pulse': 'heart_rate',
            'temp': 'temperature',
            'temperature': 'temperature',
            'rr': 'respiratory_rate',
            'respiratory rate': 'respiratory_rate',
            'o2 sat': 'oxygen_saturation',
            'spo2': 'oxygen_saturation'
        }
        
        for key, value in pattern_to_name.items():
            if key in pattern.lower():
                return value
        
        return 'unknown'
    
    def _get_default_unit(self, test_name: str) -> str:
        """Get default unit for lab test"""
        default_units = {
            'glucose': 'mg/dL',
            'hemoglobin': 'g/dL',
            'creatinine': 'mg/dL',
            'cholesterol': 'mg/dL',
            'white_blood_cell_count': 'K/uL'
        }
        
        return default_units.get(test_name, '')
    
    def calculate_text_quality_score(self, text: str) -> float:
        """Calculate quality score for medical text"""
        if not text or len(text) < 10:
            return 0.0
        
        scores = []
        
        # Length score (optimal range 100-2000 characters)
        length_score = min(1.0, len(text) / 1000) if len(text) <= 1000 else max(0.5, 2000 / len(text))
        scores.append(length_score)
        
        # Readability score
        try:
            flesch_score = textstat.flesch_reading_ease(text)
            readability_score = flesch_score / 100.0
            readability_score = max(0.0, min(1.0, readability_score))
        except:
            readability_score = 0.5
        scores.append(readability_score)
        
        # Vocabulary diversity (unique words / total words)
        words = text.lower().split()
        if len(words) > 0:
            unique_words = len(set(words))
            diversity_score = unique_words / len(words)
        else:
            diversity_score = 0.0
        scores.append(diversity_score)
        
        # Medical content score (presence of medical terms)
        medical_terms = [
            'patient', 'diagnosis', 'treatment', 'medication', 'symptom', 'condition',
            'examination', 'test', 'result', 'history', 'assessment', 'plan'
        ]
        medical_term_count = sum(1 for term in medical_terms if term in text.lower())
        medical_score = min(1.0, medical_term_count / 5)  # Normalize to 5 terms
        scores.append(medical_score)
        
        # Sentence structure score
        sentences = sent_tokenize(text)
        if len(sentences) > 0:
            avg_sentence_length = len(words) / len(sentences)
            # Optimal sentence length between 10-25 words
            if 10 <= avg_sentence_length <= 25:
                structure_score = 1.0
            elif avg_sentence_length < 10:
                structure_score = avg_sentence_length / 10
            else:
                structure_score = 25 / avg_sentence_length
        else:
            structure_score = 0.0
        scores.append(structure_score)
        
        return np.mean(scores)
    
    def clean_medical_text(self, text: str) -> str:
        """Clean and standardize medical text"""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize line breaks
        cleaned = re.sub(r'\s+', ' ', text).strip()
        
        # Remove non-printable characters
        cleaned = re.sub(r'[^\x20-\x7E]', ' ', cleaned)
        
        # Standardize medical abbreviations and units
        if self.config.normalize_medical_terms:
            cleaned = self.normalize_medical_text(cleaned)
        
        # Remove PHI if configured
        if self.config.remove_phi:
            cleaned, phi_count = self.remove_phi(cleaned)
            self.processing_stats['phi_instances_removed'] += phi_count
        
        return cleaned
    
    def preprocess_medical_document(self, document: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Preprocess a single medical document"""
        try:
            # Extract text content
            text_content = document.get('content', document.get('text', ''))
            
            if not text_content or len(text_content) < self.config.min_text_length:
                self.processing_stats['quality_rejections'] += 1
                return None
            
            # Clean the text
            cleaned_text = self.clean_medical_text(text_content)
            
            if len(cleaned_text) > self.config.max_text_length:
                cleaned_text = cleaned_text[:self.config.max_text_length]
            
            # Calculate quality score
            quality_score = self.calculate_text_quality_score(cleaned_text)
            
            if quality_score < self.config.quality_score_threshold:
                self.processing_stats['quality_rejections'] += 1
                return None
            
            # Extract structured data if configured
            structured_data = {}
            if self.config.extract_structured_data:
                structured_data = self.extract_structured_data(cleaned_text)
            
            # Create processed document
            processed_doc = {
                'document_id': document.get('document_id', f"doc_{datetime.now().timestamp()}"),
                'original_text': text_content,
                'cleaned_text': cleaned_text,
                'quality_score': quality_score,
                'structured_data': structured_data,
                'metadata': {
                    'original_length': len(text_content),
                    'processed_length': len(cleaned_text),
                    'processing_timestamp': datetime.now().isoformat(),
                    'vibey_processor_version': '4.2.1'
                }
            }
            
            # Preserve original metadata
            if 'metadata' in document:
                processed_doc['metadata'].update(document['metadata'])
            
            # Preserve other fields
            for key, value in document.items():
                if key not in ['content', 'text']:
                    processed_doc[key] = value
            
            self.processing_stats['documents_processed'] += 1
            
            return processed_doc
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return None
    
    def preprocess_medical_dataset(self, documents: List[Dict[str, Any]], 
                                 parallel_processing: bool = True) -> List[Dict[str, Any]]:
        """Preprocess a dataset of medical documents"""
        start_time = datetime.now()
        
        logger.info(f"Starting preprocessing of {len(documents)} medical documents")
        
        if parallel_processing and len(documents) > 100:
            # Use multiprocessing for large datasets
            num_processes = min(mp.cpu_count(), 8)
            with mp.Pool(processes=num_processes) as pool:
                processed_docs = pool.map(self.preprocess_medical_document, documents)
        else:
            # Sequential processing
            processed_docs = []
            for i, doc in enumerate(documents):
                if i % 100 == 0:
                    logger.info(f"Processed {i}/{len(documents)} documents")
                processed_doc = self.preprocess_medical_document(doc)
                processed_docs.append(processed_doc)
        
        # Filter out None results
        valid_docs = [doc for doc in processed_docs if doc is not None]
        
        # Remove duplicates if configured
        if self.config.remove_duplicates:
            valid_docs = self._remove_duplicate_documents(valid_docs)
        
        # Update processing statistics
        end_time = datetime.now()
        self.processing_stats['processing_time'] = (end_time - start_time).total_seconds()
        
        logger.info(f"Preprocessing completed:")
        logger.info(f"  Original documents: {len(documents)}")
        logger.info(f"  Valid processed documents: {len(valid_docs)}")
        logger.info(f"  Quality rejections: {self.processing_stats['quality_rejections']}")
        logger.info(f"  Duplicates removed: {self.processing_stats['duplicates_removed']}")
        logger.info(f"  PHI instances removed: {self.processing_stats['phi_instances_removed']}")
        logger.info(f"  Processing time: {self.processing_stats['processing_time']:.2f} seconds")
        
        return valid_docs
    
    def _remove_duplicate_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate documents based on text similarity"""
        if len(documents) <= 1:
            return documents
        
        logger.info("Removing duplicate documents...")
        
        # Extract text content for similarity comparison
        texts = [doc['cleaned_text'] for doc in documents]
        
        # Vectorize texts using TF-IDF
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
        except ValueError:
            # Handle case where all documents are empty or very similar
            return documents
        
        # Calculate pairwise similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Find duplicates (similarity > 0.9)
        duplicates = set()
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix)):
                if similarity_matrix[i][j] > 0.9:
                    duplicates.add(j)  # Mark the second document as duplicate
        
        # Filter out duplicates
        unique_docs = [doc for i, doc in enumerate(documents) if i not in duplicates]
        
        self.processing_stats['duplicates_removed'] = len(duplicates)
        
        logger.info(f"Removed {len(duplicates)} duplicate documents")
        
        return unique_docs
    
    def generate_feature_vectors(self, documents: List[Dict[str, Any]], 
                                method: str = 'tfidf') -> Tuple[np.ndarray, Any]:
        """Generate feature vectors from processed documents"""
        texts = [doc['cleaned_text'] for doc in documents]
        
        if method == 'tfidf':
            vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            feature_matrix = vectorizer.fit_transform(texts)
            
        elif method == 'count':
            vectorizer = CountVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
            feature_matrix = vectorizer.fit_transform(texts)
            
        elif method == 'lda':
            # First create TF-IDF features
            tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                min_df=2,
                max_df=0.95
            )
            tfidf_features = tfidf_vectorizer.fit_transform(texts)
            
            # Then apply LDA
            lda = LatentDirichletAllocation(
                n_components=50,
                random_state=42,
                max_iter=100
            )
            feature_matrix = lda.fit_transform(tfidf_features)
            vectorizer = {'tfidf': tfidf_vectorizer, 'lda': lda}
            
        else:
            raise ValueError(f"Unknown feature extraction method: {method}")
        
        return feature_matrix, vectorizer
    
    def create_training_dataset(self, processed_documents: List[Dict[str, Any]], 
                              labels: Optional[List[str]] = None,
                              test_size: float = 0.2) -> Dict[str, Any]:
        """Create training dataset from processed documents"""
        
        logger.info(f"Creating training dataset from {len(processed_documents)} documents")
        
        # Generate features
        feature_matrix, vectorizer = self.generate_feature_vectors(processed_documents)
        
        # Prepare text data
        texts = [doc['cleaned_text'] for doc in processed_documents]
        metadata = [doc['metadata'] for doc in processed_documents]
        
        # Split data
        if labels is not None:
            X_train, X_test, y_train, y_test, texts_train, texts_test, meta_train, meta_test = train_test_split(
                feature_matrix, labels, texts, metadata, 
                test_size=test_size, random_state=42, stratify=labels
            )
        else:
            X_train, X_test, texts_train, texts_test, meta_train, meta_test = train_test_split(
                feature_matrix, texts, metadata, 
                test_size=test_size, random_state=42
            )
            y_train = y_test = None
        
        training_dataset = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'texts_train': texts_train,
            'texts_test': texts_test,
            'metadata_train': meta_train,
            'metadata_test': meta_test,
            'vectorizer': vectorizer,
            'feature_names': vectorizer.get_feature_names_out() if hasattr(vectorizer, 'get_feature_names_out') else None,
            'dataset_stats': {
                'total_documents': len(processed_documents),
                'training_documents': len(texts_train),
                'test_documents': len(texts_test),
                'feature_dimensions': feature_matrix.shape[1],
                'creation_timestamp': datetime.now().isoformat(),
                'vibey_version': '4.2.1'
            }
        }
        
        logger.info(f"Training dataset created:")
        logger.info(f"  Training samples: {len(texts_train)}")
        logger.info(f"  Test samples: {len(texts_test)}")
        logger.info(f"  Feature dimensions: {feature_matrix.shape[1]}")
        
        return training_dataset
    
    def save_processed_dataset(self, dataset: Dict[str, Any], output_path: str):
        """Save processed dataset to disk"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save main dataset
        with open(output_path / 'vibey_processed_dataset.pickle', 'wb') as f:
            pickle.dump(dataset, f)
        
        # Save statistics and metadata
        stats = {
            'processing_stats': self.processing_stats,
            'dataset_stats': dataset['dataset_stats'],
            'config': self.config.__dict__
        }
        
        with open(output_path / 'processing_stats.json', 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"Processed dataset saved to {output_path}")
    
    def load_processed_dataset(self, input_path: str) -> Dict[str, Any]:
        """Load processed dataset from disk"""
        input_path = Path(input_path)
        
        with open(input_path / 'vibey_processed_dataset.pickle', 'rb') as f:
            dataset = pickle.load(f)
        
        logger.info(f"Loaded processed dataset from {input_path}")
        logger.info(f"  Training samples: {dataset['dataset_stats']['training_documents']}")
        logger.info(f"  Test samples: {dataset['dataset_stats']['test_documents']}")
        
        return dataset

def preprocess_vibey_medical_data(documents: List[Dict[str, Any]], 
                                config: PreprocessingConfig = None) -> Dict[str, Any]:
    """
    Main function to preprocess medical data for VibeyBot training
    
    Args:
        documents: List of medical documents to preprocess
        config: Preprocessing configuration
        
    Returns:
        Processed training dataset ready for VibeyBot AI training
    """
    processor = VibeyMedicalDataProcessor(config)
    
    # Preprocess documents
    processed_docs = processor.preprocess_medical_dataset(documents)
    
    # Create training dataset
    training_dataset = processor.create_training_dataset(processed_docs)
    
    logger.info("VibeyBot medical data preprocessing completed")
    logger.info(f"Dataset ready for AI training with {training_dataset['dataset_stats']['training_documents']} samples")
    
    return training_dataset

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='VibeyBot Medical Data Preprocessing')
    parser.add_argument('--input', required=True, help='Input file with medical documents (JSON)')
    parser.add_argument('--output', required=True, help='Output directory for processed dataset')
    parser.add_argument('--config', help='Configuration file (JSON)')
    
    args = parser.parse_args()
    
    # Load input data
    with open(args.input, 'r') as f:
        documents = json.load(f)
    
    # Load configuration if provided
    config = PreprocessingConfig()
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = PreprocessingConfig(**config_dict)
    
    # Process data
    processor = VibeyMedicalDataProcessor(config)
    training_dataset = preprocess_vibey_medical_data(documents, config)
    
    # Save processed dataset
    processor.save_processed_dataset(training_dataset, args.output)
    
    print(f"VibeyBot medical data preprocessing completed!")
    print(f"Processed dataset saved to: {args.output}")
    print(f"Ready for VibeyBot AI training pipeline")