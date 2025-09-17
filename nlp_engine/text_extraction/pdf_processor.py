"""
VibeyBot Advanced PDF Medical Document Processing
Sophisticated PDF text extraction and analysis for medical documents
Integrates with the main VibeyBot intelligence system for document processing
"""
import os
import sys
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime
import json
import tempfile
import subprocess

# PDF processing libraries
import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
from pdfminer.high_level import extract_text as pdfminer_extract
from pdfminer.layout import LAParams
import tabula
import camelot

# OCR and image processing
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import cv2

# Medical text processing
import spacy
from spacy import displacy
import medspacy
from medspacy.ner import TargetRule
from medspacy.context import ConTextRule

# Machine learning libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VibeyMedicalPDFProcessor:
    """
    Advanced PDF processor specifically designed for medical documents
    Handles various medical document types with specialized extraction techniques
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.temp_dir = Path(tempfile.gettempdir()) / "vibey_pdf_processing"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Initialize medical NLP model
        self.nlp = self._initialize_medical_nlp()
        
        # Medical document patterns
        self.medical_patterns = {
            'lab_values': [
                r'(?:hemoglobin|hgb|hb)[\s:]*(\d+\.?\d*)\s*(?:g/dl|g/dL|gm/dl)',
                r'(?:glucose|blood\s+sugar)[\s:]*(\d+\.?\d*)\s*(?:mg/dl|mg/dL)',
                r'(?:cholesterol)[\s:]*(\d+\.?\d*)\s*(?:mg/dl|mg/dL)',
                r'(?:wbc|white\s+blood\s+cell)[\s:]*(\d+\.?\d*)\s*(?:/cmm|K/uL)',
                r'(?:rbc|red\s+blood\s+cell)[\s:]*(\d+\.?\d*)\s*(?:M/uL|million/cmm)',
                r'(?:platelet)[\s:]*(\d+\.?\d*)\s*(?:K/uL|thou/cmm)',
                r'(?:creatinine)[\s:]*(\d+\.?\d*)\s*(?:mg/dl|mg/dL)',
                r'(?:bun)[\s:]*(\d+\.?\d*)\s*(?:mg/dl|mg/dL)',
                r'(?:troponin)[\s:]*(\d+\.?\d*)\s*(?:ng/ml|ng/mL)'
            ],
            'vital_signs': [
                r'(?:bp|blood\s+pressure)[\s:]*(\d+)/(\d+)\s*(?:mmhg|mm\s+hg)?',
                r'(?:hr|heart\s+rate|pulse)[\s:]*(\d+)\s*(?:bpm)?',
                r'(?:temp|temperature)[\s:]*(\d+\.?\d*)\s*(?:f|°f|c|°c)?',
                r'(?:rr|respiratory\s+rate)[\s:]*(\d+)\s*(?:rpm|/min)?',
                r'(?:o2\s+sat|oxygen\s+saturation|spo2)[\s:]*(\d+)\s*%?'
            ],
            'medications': [
                r'(?:aspirin|asa)[\s:]*(\d+\.?\d*)\s*(?:mg)',
                r'(?:metformin)[\s:]*(\d+\.?\d*)\s*(?:mg)',
                r'(?:lisinopril)[\s:]*(\d+\.?\d*)\s*(?:mg)',
                r'(?:amlodipine)[\s:]*(\d+\.?\d*)\s*(?:mg)',
                r'(?:atorvastatin|lipitor)[\s:]*(\d+\.?\d*)\s*(?:mg)'
            ],
            'medical_conditions': [
                r'(?:diabetes|dm|diabetic)',
                r'(?:hypertension|htn|high\s+blood\s+pressure)',
                r'(?:myocardial\s+infarction|mi|heart\s+attack)',
                r'(?:pneumonia|lung\s+infection)',
                r'(?:appendicitis)',
                r'(?:copd|chronic\s+obstructive\s+pulmonary)',
                r'(?:ckd|chronic\s+kidney\s+disease)'
            ]
        }
        
        # Document type classifiers
        self.document_classifiers = {
            'lab_report': ['laboratory', 'lab report', 'blood work', 'chemistry', 'hematology'],
            'radiology': ['x-ray', 'ct scan', 'mri', 'ultrasound', 'imaging', 'radiology'],
            'pathology': ['pathology', 'biopsy', 'histology', 'cytology', 'specimen'],
            'clinical_notes': ['progress note', 'consultation', 'history and physical', 'discharge'],
            'prescription': ['prescription', 'medication', 'pharmacy', 'rx', 'drug'],
            'insurance': ['insurance', 'claim', 'authorization', 'coverage', 'benefit']
        }
        
    def _initialize_medical_nlp(self) -> spacy.language.Language:
        """Initialize medical NLP model with specialized components"""
        try:
            # Try to load medical model
            nlp = spacy.load("en_core_sci_md")
            logger.info("Loaded SciSpaCy medical model")
        except OSError:
            try:
                # Fallback to standard model
                nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded standard spaCy model (consider installing SciSpaCy for better medical NER)")
            except OSError:
                # Create blank model if no models available
                nlp = spacy.blank("en")
                logger.warning("No spaCy models found, using blank model")
        
        # Add medical NER rules
        if not nlp.has_pipe("entity_ruler"):
            ruler = nlp.add_pipe("entity_ruler", before="ner" if nlp.has_pipe("ner") else None)
            
            # Medical entity patterns
            medical_patterns = [
                {"label": "MEDICATION", "pattern": [{"LOWER": {"IN": ["aspirin", "metformin", "lisinopril", "amlodipine"]}}]},
                {"label": "LAB_TEST", "pattern": [{"LOWER": {"IN": ["hemoglobin", "glucose", "cholesterol", "creatinine"]}}]},
                {"label": "VITAL_SIGN", "pattern": [{"LOWER": "blood"}, {"LOWER": "pressure"}]},
                {"label": "VITAL_SIGN", "pattern": [{"LOWER": "heart"}, {"LOWER": "rate"}]},
                {"label": "MEDICAL_CONDITION", "pattern": [{"LOWER": {"IN": ["diabetes", "hypertension", "pneumonia"]}}]}
            ]
            
            ruler.add_patterns(medical_patterns)
        
        return nlp
    
    def extract_text_multi_method(self, pdf_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract text using multiple methods and combine results
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Processing medical PDF: {pdf_path.name}")
        
        extraction_results = {
            'file_name': pdf_path.name,
            'file_size': pdf_path.stat().st_size,
            'processing_timestamp': datetime.now().isoformat(),
            'extraction_methods': {},
            'combined_text': '',
            'document_type': '',
            'medical_entities': [],
            'structured_data': {},
            'quality_score': 0.0
        }
        
        # Method 1: PyPDF2
        try:
            pypdf2_text = self._extract_with_pypdf2(pdf_path)
            extraction_results['extraction_methods']['pypdf2'] = {
                'text': pypdf2_text,
                'length': len(pypdf2_text),
                'success': len(pypdf2_text) > 50
            }
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {str(e)}")
            extraction_results['extraction_methods']['pypdf2'] = {'success': False, 'error': str(e)}
        
        # Method 2: pdfplumber
        try:
            pdfplumber_text = self._extract_with_pdfplumber(pdf_path)
            extraction_results['extraction_methods']['pdfplumber'] = {
                'text': pdfplumber_text,
                'length': len(pdfplumber_text),
                'success': len(pdfplumber_text) > 50
            }
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {str(e)}")
            extraction_results['extraction_methods']['pdfplumber'] = {'success': False, 'error': str(e)}
        
        # Method 3: PyMuPDF (fitz)
        try:
            pymupdf_text = self._extract_with_pymupdf(pdf_path)
            extraction_results['extraction_methods']['pymupdf'] = {
                'text': pymupdf_text,
                'length': len(pymupdf_text),
                'success': len(pymupdf_text) > 50
            }
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {str(e)}")
            extraction_results['extraction_methods']['pymupdf'] = {'success': False, 'error': str(e)}
        
        # Method 4: pdfminer
        try:
            pdfminer_text = self._extract_with_pdfminer(pdf_path)
            extraction_results['extraction_methods']['pdfminer'] = {
                'text': pdfminer_text,
                'length': len(pdfminer_text),
                'success': len(pdfminer_text) > 50
            }
        except Exception as e:
            logger.warning(f"pdfminer extraction failed: {str(e)}")
            extraction_results['extraction_methods']['pdfminer'] = {'success': False, 'error': str(e)}
        
        # Method 5: OCR fallback for image-based PDFs
        try:
            ocr_text = self._extract_with_ocr(pdf_path)
            extraction_results['extraction_methods']['ocr'] = {
                'text': ocr_text,
                'length': len(ocr_text),
                'success': len(ocr_text) > 50
            }
        except Exception as e:
            logger.warning(f"OCR extraction failed: {str(e)}")
            extraction_results['extraction_methods']['ocr'] = {'success': False, 'error': str(e)}
        
        # Combine and select best extraction
        best_text = self._select_best_extraction(extraction_results['extraction_methods'])
        extraction_results['combined_text'] = best_text
        
        # Classify document type
        extraction_results['document_type'] = self._classify_medical_document(best_text)
        
        # Extract medical entities
        extraction_results['medical_entities'] = self._extract_medical_entities(best_text)
        
        # Extract structured medical data
        extraction_results['structured_data'] = self._extract_structured_medical_data(best_text)
        
        # Calculate quality score
        extraction_results['quality_score'] = self._calculate_quality_score(extraction_results)
        
        logger.info(f"PDF processing completed. Document type: {extraction_results['document_type']}, Quality: {extraction_results['quality_score']:.2f}")
        
        return extraction_results
    
    def _extract_with_pypdf2(self, pdf_path: Path) -> str:
        """Extract text using PyPDF2"""
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _extract_with_pdfplumber(self, pdf_path: Path) -> str:
        """Extract text using pdfplumber with table detection"""
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Extract text
                page_text = page.extract_text() or ""
                text += page_text + "\n"
                
                # Extract tables
                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        if row:
                            row_text = " | ".join([cell or "" for cell in row])
                            text += row_text + "\n"
        return text
    
    def _extract_with_pymupdf(self, pdf_path: Path) -> str:
        """Extract text using PyMuPDF"""
        text = ""
        pdf_document = fitz.open(str(pdf_path))
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text += page.get_text() + "\n"
        
        pdf_document.close()
        return text
    
    def _extract_with_pdfminer(self, pdf_path: Path) -> str:
        """Extract text using pdfminer"""
        laparams = LAParams(
            line_margin=0.5,
            word_margin=0.1,
            char_margin=2.0,
            boxes_flow=0.5,
            detect_vertical=True
        )
        
        return pdfminer_extract(str(pdf_path), laparams=laparams)
    
    def _extract_with_ocr(self, pdf_path: Path) -> str:
        """Extract text using OCR for image-based PDFs"""
        text = ""
        
        # Convert PDF pages to images
        pdf_document = fitz.open(str(pdf_path))
        
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            
            # Convert page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))  # 2x zoom for better OCR
            img_data = pix.tobytes("png")
            
            # Save temporary image
            temp_img_path = self.temp_dir / f"page_{page_num}.png"
            with open(temp_img_path, "wb") as f:
                f.write(img_data)
            
            # Preprocess image for better OCR
            processed_img = self._preprocess_image_for_ocr(temp_img_path)
            
            # Perform OCR
            try:
                page_text = pytesseract.image_to_string(processed_img, config='--psm 1')
                text += page_text + "\n"
            except Exception as e:
                logger.warning(f"OCR failed for page {page_num}: {str(e)}")
            
            # Clean up temporary file
            temp_img_path.unlink(missing_ok=True)
        
        pdf_document.close()
        return text
    
    def _preprocess_image_for_ocr(self, image_path: Path) -> Image.Image:
        """Preprocess image for better OCR accuracy"""
        # Open image
        img = Image.open(image_path)
        
        # Convert to grayscale
        img = img.convert('L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(2.0)
        
        # Apply median filter to reduce noise
        img = img.filter(ImageFilter.MedianFilter(size=3))
        
        return img
    
    def _select_best_extraction(self, extraction_methods: Dict) -> str:
        """Select the best text extraction result"""
        valid_extractions = []
        
        for method, result in extraction_methods.items():
            if result.get('success', False) and result.get('length', 0) > 50:
                valid_extractions.append({
                    'method': method,
                    'text': result['text'],
                    'length': result['length'],
                    'medical_score': self._calculate_medical_content_score(result['text'])
                })
        
        if not valid_extractions:
            return "No text could be extracted from PDF"
        
        # Sort by medical content score and length
        valid_extractions.sort(key=lambda x: (x['medical_score'], x['length']), reverse=True)
        
        best_extraction = valid_extractions[0]
        logger.info(f"Selected {best_extraction['method']} as best extraction method")
        
        return best_extraction['text']
    
    def _calculate_medical_content_score(self, text: str) -> float:
        """Calculate how much medical content is in the text"""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        medical_terms = [
            'patient', 'medical', 'diagnosis', 'treatment', 'symptom', 'lab', 'test',
            'blood', 'pressure', 'heart', 'rate', 'temperature', 'glucose', 'hemoglobin',
            'doctor', 'physician', 'hospital', 'clinic', 'medication', 'prescription',
            'mg/dl', 'mmhg', 'bpm', 'normal', 'abnormal', 'elevated', 'low', 'high'
        ]
        
        found_terms = sum(1 for term in medical_terms if term in text_lower)
        return found_terms / len(medical_terms)
    
    def _classify_medical_document(self, text: str) -> str:
        """Classify the type of medical document"""
        if not text:
            return 'unknown'
        
        text_lower = text.lower()
        
        # Score each document type
        type_scores = {}
        for doc_type, keywords in self.document_classifiers.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            type_scores[doc_type] = score
        
        # Return type with highest score
        if type_scores:
            best_type = max(type_scores.items(), key=lambda x: x[1])
            if best_type[1] > 0:
                return best_type[0]
        
        return 'clinical_notes'  # Default classification
    
    def _extract_medical_entities(self, text: str) -> List[Dict]:
        """Extract medical entities using NLP"""
        if not text:
            return []
        
        # Process with spaCy
        doc = self.nlp(text)
        entities = []
        
        # Extract named entities
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': 1.0  # spaCy doesn't provide confidence scores
            })
        
        # Extract pattern-based entities
        pattern_entities = self._extract_pattern_entities(text)
        entities.extend(pattern_entities)
        
        return entities
    
    def _extract_pattern_entities(self, text: str) -> List[Dict]:
        """Extract medical entities using regex patterns"""
        entities = []
        
        for category, patterns in self.medical_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entities.append({
                        'text': match.group(0),
                        'label': category.upper(),
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.8,
                        'value': match.groups() if match.groups() else match.group(0)
                    })
        
        return entities
    
    def _extract_structured_medical_data(self, text: str) -> Dict:
        """Extract structured medical data from text"""
        structured_data = {
            'lab_values': {},
            'vital_signs': {},
            'medications': [],
            'conditions': [],
            'demographics': {}
        }
        
        if not text:
            return structured_data
        
        # Extract lab values
        for pattern in self.medical_patterns['lab_values']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                test_name = self._extract_lab_name(match.group(0))
                value = match.groups()[0] if match.groups() else None
                if test_name and value:
                    structured_data['lab_values'][test_name] = {
                        'value': float(value) if value.replace('.', '').isdigit() else value,
                        'unit': self._extract_unit(match.group(0)),
                        'text': match.group(0)
                    }
        
        # Extract vital signs
        for pattern in self.medical_patterns['vital_signs']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                vital_name = self._extract_vital_name(match.group(0))
                if vital_name:
                    if 'blood pressure' in vital_name.lower() and len(match.groups()) >= 2:
                        structured_data['vital_signs'][vital_name] = {
                            'systolic': int(match.groups()[0]),
                            'diastolic': int(match.groups()[1]),
                            'text': match.group(0)
                        }
                    else:
                        value = match.groups()[0] if match.groups() else None
                        if value:
                            structured_data['vital_signs'][vital_name] = {
                                'value': float(value) if value.replace('.', '').isdigit() else value,
                                'text': match.group(0)
                            }
        
        # Extract medications
        for pattern in self.medical_patterns['medications']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                medication = {
                    'name': self._extract_medication_name(match.group(0)),
                    'dosage': match.groups()[0] if match.groups() else None,
                    'unit': self._extract_unit(match.group(0)),
                    'text': match.group(0)
                }
                structured_data['medications'].append(medication)
        
        # Extract conditions
        for pattern in self.medical_patterns['medical_conditions']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                condition = {
                    'name': match.group(0),
                    'text': match.group(0)
                }
                structured_data['conditions'].append(condition)
        
        return structured_data
    
    def _extract_lab_name(self, text: str) -> str:
        """Extract laboratory test name from text"""
        lab_mapping = {
            'hemoglobin': 'Hemoglobin',
            'hgb': 'Hemoglobin',
            'hb': 'Hemoglobin',
            'glucose': 'Glucose',
            'blood sugar': 'Glucose',
            'cholesterol': 'Cholesterol',
            'wbc': 'White Blood Cell Count',
            'white blood cell': 'White Blood Cell Count',
            'rbc': 'Red Blood Cell Count',
            'red blood cell': 'Red Blood Cell Count',
            'platelet': 'Platelet Count',
            'creatinine': 'Creatinine',
            'bun': 'Blood Urea Nitrogen',
            'troponin': 'Troponin'
        }
        
        text_lower = text.lower()
        for key, value in lab_mapping.items():
            if key in text_lower:
                return value
        
        return text.strip()
    
    def _extract_vital_name(self, text: str) -> str:
        """Extract vital sign name from text"""
        vital_mapping = {
            'bp': 'Blood Pressure',
            'blood pressure': 'Blood Pressure',
            'hr': 'Heart Rate',
            'heart rate': 'Heart Rate',
            'pulse': 'Heart Rate',
            'temp': 'Temperature',
            'temperature': 'Temperature',
            'rr': 'Respiratory Rate',
            'respiratory rate': 'Respiratory Rate',
            'o2 sat': 'Oxygen Saturation',
            'oxygen saturation': 'Oxygen Saturation',
            'spo2': 'Oxygen Saturation'
        }
        
        text_lower = text.lower()
        for key, value in vital_mapping.items():
            if key in text_lower:
                return value
        
        return text.strip()
    
    def _extract_medication_name(self, text: str) -> str:
        """Extract medication name from text"""
        medication_mapping = {
            'aspirin': 'Aspirin',
            'asa': 'Aspirin',
            'metformin': 'Metformin',
            'lisinopril': 'Lisinopril',
            'amlodipine': 'Amlodipine',
            'atorvastatin': 'Atorvastatin',
            'lipitor': 'Atorvastatin'
        }
        
        text_lower = text.lower()
        for key, value in medication_mapping.items():
            if key in text_lower:
                return value
        
        return text.strip()
    
    def _extract_unit(self, text: str) -> str:
        """Extract unit from text"""
        units = ['mg/dl', 'mg/dL', 'g/dl', 'g/dL', '/cmm', 'K/uL', 'M/uL', 'ng/ml', 'ng/mL', 
                'mmhg', 'mm hg', 'bpm', '°f', '°c', 'f', 'c', '%', 'mg']
        
        text_lower = text.lower()
        for unit in units:
            if unit.lower() in text_lower:
                return unit
        
        return ''
    
    def _calculate_quality_score(self, extraction_results: Dict) -> float:
        """Calculate overall quality score for the extraction"""
        scores = []
        
        # Text length score
        text_length = len(extraction_results.get('combined_text', ''))
        length_score = min(text_length / 1000, 1.0)  # Normalize to 1.0 for 1000+ chars
        scores.append(length_score)
        
        # Medical content score
        medical_score = self._calculate_medical_content_score(extraction_results.get('combined_text', ''))
        scores.append(medical_score)
        
        # Structured data score
        structured_data = extraction_results.get('structured_data', {})
        structured_score = 0.0
        if structured_data.get('lab_values'):
            structured_score += 0.3
        if structured_data.get('vital_signs'):
            structured_score += 0.3
        if structured_data.get('medications'):
            structured_score += 0.2
        if structured_data.get('conditions'):
            structured_score += 0.2
        scores.append(structured_score)
        
        # Entity extraction score
        entities = extraction_results.get('medical_entities', [])
        entity_score = min(len(entities) / 20, 1.0)  # Normalize to 1.0 for 20+ entities
        scores.append(entity_score)
        
        return np.mean(scores)

def process_medical_pdf_for_vibey(pdf_path: Union[str, Path], 
                                config: Dict = None) -> Dict[str, Any]:
    """
    Main function to process medical PDFs for VibeyBot system
    
    Args:
        pdf_path: Path to PDF file
        config: Processing configuration
        
    Returns:
        Processed medical document data
    """
    processor = VibeyMedicalPDFProcessor(config)
    
    try:
        # Extract text and medical data
        extraction_results = processor.extract_text_multi_method(pdf_path)
        
        # Format for VibeyBot integration
        vibey_formatted_data = {
            'document_id': f"pdf_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'source_file': extraction_results['file_name'],
            'document_type': extraction_results['document_type'],
            'extracted_text': extraction_results['combined_text'],
            'medical_entities': extraction_results['medical_entities'],
            'structured_data': extraction_results['structured_data'],
            'quality_metrics': {
                'extraction_quality': extraction_results['quality_score'],
                'text_length': len(extraction_results['combined_text']),
                'entity_count': len(extraction_results['medical_entities']),
                'processing_method': 'vibey_advanced_pdf_processor'
            },
            'vibey_metadata': {
                'processed_timestamp': extraction_results['processing_timestamp'],
                'processor_version': '4.2.1',
                'compatible_with_vibey_engine': True,
                'ready_for_analysis': extraction_results['quality_score'] > 0.5
            }
        }
        
        logger.info(f"PDF processed for VibeyBot: {pdf_path}")
        logger.info(f"Quality score: {extraction_results['quality_score']:.3f}")
        logger.info(f"Document type: {extraction_results['document_type']}")
        
        return vibey_formatted_data
        
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
        return {
            'error': str(e),
            'source_file': str(pdf_path),
            'processing_failed': True,
            'vibey_metadata': {
                'processed_timestamp': datetime.now().isoformat(),
                'processor_version': '4.2.1',
                'compatible_with_vibey_engine': False,
                'ready_for_analysis': False
            }
        }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='VibeyBot Medical PDF Processor')
    parser.add_argument('pdf_path', help='Path to PDF file to process')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--output', help='Output file for processed data')
    
    args = parser.parse_args()
    
    # Load config if provided
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Process PDF
    result = process_medical_pdf_for_vibey(args.pdf_path, config)
    
    # Save output if specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
    else:
        print(json.dumps(result, indent=2))