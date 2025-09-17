"""
VibeyBot Medical Synthetic Data Generator
Advanced synthetic medical data generation for AI training and testing
Generates realistic medical documents, patient records, and clinical scenarios
"""
import os
import sys
import random
import json
import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

# Data generation libraries
from faker import Faker
from faker.providers import BaseProvider
import pandas as pd

# Medical data libraries
import medspacy
from medspacy.ner import TargetRule

# Statistical libraries
from scipy import stats
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Faker with medical seed data
fake = Faker(['en_US'])
fake.seed_instance(42)  # For reproducible generation

class PatientGender(Enum):
    """Patient gender enumeration"""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"

class DocumentType(Enum):
    """Medical document type enumeration"""
    LAB_REPORT = "lab_report"
    RADIOLOGY_REPORT = "radiology_report"
    PATHOLOGY_REPORT = "pathology_report"
    CLINICAL_NOTES = "clinical_notes"
    DISCHARGE_SUMMARY = "discharge_summary"
    PROCEDURE_NOTE = "procedure_note"
    CONSULTATION_NOTE = "consultation_note"
    PROGRESS_NOTE = "progress_note"
    EMERGENCY_NOTE = "emergency_note"
    PRESCRIPTION = "prescription"

class Severity(Enum):
    """Medical condition severity levels"""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"

@dataclass
class SyntheticPatient:
    """Synthetic patient data structure"""
    patient_id: str
    first_name: str
    last_name: str
    date_of_birth: str
    gender: PatientGender
    age: int
    height_cm: float
    weight_kg: float
    bmi: float
    blood_type: str
    allergies: List[str]
    medications: List[Dict[str, Any]]
    medical_history: List[Dict[str, Any]]
    family_history: List[str]
    social_history: Dict[str, Any]
    insurance: Dict[str, str]
    emergency_contact: Dict[str, str]
    primary_care_physician: str

@dataclass
class SyntheticLabResult:
    """Synthetic lab result data structure"""
    test_name: str
    value: float
    unit: str
    reference_range: str
    abnormal_flag: Optional[str]
    collection_date: str
    result_date: str

@dataclass
class SyntheticMedicalDocument:
    """Synthetic medical document data structure"""
    document_id: str
    document_type: DocumentType
    patient_id: str
    creation_date: str
    provider_name: str
    facility: str
    content: str
    structured_data: Dict[str, Any]
    metadata: Dict[str, Any]

class MedicalDataProvider(BaseProvider):
    """Custom Faker provider for medical data"""
    
    def __init__(self, generator):
        super().__init__(generator)
        
        # Medical terminology databases
        self.medications = [
            'Metformin', 'Lisinopril', 'Atorvastatin', 'Levothyroxine', 'Amlodipine',
            'Omeprazole', 'Simvastatin', 'Losartan', 'Gabapentin', 'Hydrochlorothiazide',
            'Sertraline', 'Ibuprofen', 'Furosemide', 'Clopidogrel', 'Montelukast',
            'Escitalopram', 'Rosuvastatin', 'Tramadol', 'Trazodone', 'Duloxetine',
            'Pregabalin', 'Warfarin', 'Insulin', 'Prednisone', 'Aspirin'
        ]
        
        self.medical_conditions = [
            'Hypertension', 'Diabetes Mellitus Type 2', 'Hyperlipidemia', 'Asthma',
            'Depression', 'Anxiety', 'Osteoarthritis', 'COPD', 'Heart Disease',
            'Hypothyroidism', 'Chronic Kidney Disease', 'Atrial Fibrillation',
            'Sleep Apnea', 'Gastroesophageal Reflux', 'Migraine', 'Obesity',
            'Chronic Pain', 'Bipolar Disorder', 'Rheumatoid Arthritis', 'Cancer History'
        ]
        
        self.allergies = [
            'Penicillin', 'Sulfa drugs', 'Latex', 'Peanuts', 'Shellfish', 'Iodine',
            'Codeine', 'Aspirin', 'NSAIDs', 'Eggs', 'Milk', 'Soy', 'Tree nuts',
            'Fish', 'Wheat', 'Contrast dye', 'Adhesive tape', 'No known allergies'
        ]
        
        self.symptoms = [
            'Chest pain', 'Shortness of breath', 'Fatigue', 'Headache', 'Nausea',
            'Dizziness', 'Abdominal pain', 'Back pain', 'Joint pain', 'Cough',
            'Fever', 'Weight loss', 'Weight gain', 'Insomnia', 'Rash',
            'Swelling', 'Palpitations', 'Weakness', 'Numbness', 'Confusion'
        ]
        
        self.lab_tests = [
            'Complete Blood Count', 'Basic Metabolic Panel', 'Comprehensive Metabolic Panel',
            'Lipid Panel', 'Liver Function Tests', 'Thyroid Function Tests',
            'Hemoglobin A1C', 'Prothrombin Time', 'Urinalysis', 'Vitamin D',
            'Vitamin B12', 'Folate', 'Iron Studies', 'PSA', 'Troponin I',
            'BNP', 'C-Reactive Protein', 'ESR', 'Rheumatoid Factor', 'ANA'
        ]
        
        self.procedures = [
            'Echocardiogram', 'Electrocardiogram', 'Chest X-ray', 'CT Scan',
            'MRI', 'Ultrasound', 'Colonoscopy', 'Endoscopy', 'Stress Test',
            'Holter Monitor', 'Pulmonary Function Test', 'Bone Density Scan',
            'Mammogram', 'Pap Smear', 'Biopsy', 'Blood Pressure Monitoring',
            'Glucose Tolerance Test', 'Sleep Study', 'Cardiac Catheterization'
        ]
        
        self.specialties = [
            'Internal Medicine', 'Cardiology', 'Endocrinology', 'Pulmonology',
            'Gastroenterology', 'Neurology', 'Psychiatry', 'Dermatology',
            'Orthopedics', 'Urology', 'Gynecology', 'Ophthalmology',
            'Otolaryngology', 'Emergency Medicine', 'Family Medicine',
            'Geriatrics', 'Pediatrics', 'Oncology', 'Nephrology', 'Rheumatology'
        ]
    
    def medication_name(self) -> str:
        return self.random_element(self.medications)
    
    def medical_condition(self) -> str:
        return self.random_element(self.medical_conditions)
    
    def allergy(self) -> str:
        return self.random_element(self.allergies)
    
    def symptom(self) -> str:
        return self.random_element(self.symptoms)
    
    def lab_test(self) -> str:
        return self.random_element(self.lab_tests)
    
    def medical_procedure(self) -> str:
        return self.random_element(self.procedures)
    
    def medical_specialty(self) -> str:
        return self.random_element(self.specialties)
    
    def blood_type(self) -> str:
        blood_types = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
        weights = [0.34, 0.06, 0.09, 0.02, 0.03, 0.01, 0.38, 0.07]  # Approximate US distribution
        return np.random.choice(blood_types, p=weights)
    
    def vital_sign_value(self, vital_type: str) -> float:
        """Generate realistic vital sign values"""
        normal_ranges = {
            'systolic_bp': (110, 140, 15),     # mean, max, std
            'diastolic_bp': (70, 90, 10),
            'heart_rate': (60, 100, 12),
            'respiratory_rate': (12, 20, 3),
            'temperature_f': (97.0, 99.5, 0.8),
            'oxygen_saturation': (95, 100, 2),
            'weight_kg': (50, 120, 25),
            'height_cm': (150, 190, 15),
            'bmi': (18.5, 30, 5)
        }
        
        if vital_type in normal_ranges:
            min_val, max_val, std = normal_ranges[vital_type]
            # Use truncated normal distribution
            value = np.random.normal(loc=(min_val + max_val) / 2, scale=std)
            return max(min_val, min(max_val, value))
        
        return 0.0

# Add the custom provider to Faker
fake.add_provider(MedicalDataProvider)

class VibeyMedicalSyntheticDataGenerator:
    """
    Advanced synthetic medical data generator for VibeyBot AI training
    Generates realistic medical documents and patient data
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.output_dir = Path(self.config.get('output_dir', 'synthetic_data'))
        self.output_dir.mkdir(exist_ok=True)
        
        # Lab test reference ranges
        self.lab_reference_ranges = {
            'Hemoglobin': {'male': (13.8, 17.2), 'female': (12.1, 15.1), 'unit': 'g/dL'},
            'Hematocrit': {'male': (40.7, 50.3), 'female': (36.1, 44.3), 'unit': '%'},
            'WBC Count': {'normal': (4.8, 10.8), 'unit': 'K/uL'},
            'Platelet Count': {'normal': (150, 450), 'unit': 'K/uL'},
            'Glucose': {'normal': (70, 100), 'prediabetic': (100, 125), 'diabetic': (126, 300), 'unit': 'mg/dL'},
            'Creatinine': {'male': (0.74, 1.35), 'female': (0.59, 1.04), 'unit': 'mg/dL'},
            'BUN': {'normal': (7, 20), 'unit': 'mg/dL'},
            'Total Cholesterol': {'normal': (0, 200), 'borderline': (200, 239), 'high': (240, 400), 'unit': 'mg/dL'},
            'LDL Cholesterol': {'normal': (0, 100), 'borderline': (100, 129), 'high': (130, 300), 'unit': 'mg/dL'},
            'HDL Cholesterol': {'male': (40, 60), 'female': (50, 60), 'unit': 'mg/dL'},
            'Triglycerides': {'normal': (0, 150), 'borderline': (150, 199), 'high': (200, 500), 'unit': 'mg/dL'},
            'HbA1c': {'normal': (4.0, 5.6), 'prediabetic': (5.7, 6.4), 'diabetic': (6.5, 12.0), 'unit': '%'},
            'TSH': {'normal': (0.4, 4.0), 'unit': 'mIU/L'},
            'ALT': {'male': (7, 56), 'female': (7, 56), 'unit': 'U/L'},
            'AST': {'male': (10, 40), 'female': (10, 40), 'unit': 'U/L'},
            'Troponin I': {'normal': (0.0, 0.04), 'elevated': (0.04, 10.0), 'unit': 'ng/mL'}
        }
        
        logger.info("VibeyBot Synthetic Medical Data Generator initialized")
    
    def generate_synthetic_patient(self, patient_conditions: List[str] = None) -> SyntheticPatient:
        """Generate a synthetic patient with realistic medical profile"""
        
        # Basic demographics
        gender = fake.random_element([PatientGender.MALE, PatientGender.FEMALE, PatientGender.OTHER])
        age = fake.random_int(min=18, max=85)
        
        # Calculate realistic height, weight, and BMI with some correlation
        if gender == PatientGender.MALE:
            height_cm = np.random.normal(175, 8)
            base_weight = (height_cm - 100) * 0.9
        else:
            height_cm = np.random.normal(162, 7)
            base_weight = (height_cm - 100) * 0.85
        
        # Add age-related weight variation
        age_factor = 1 + (age - 40) * 0.005  # Weight tends to increase with age
        weight_variation = np.random.normal(0, 10)
        weight_kg = max(40, base_weight * age_factor + weight_variation)
        
        bmi = weight_kg / ((height_cm / 100) ** 2)
        
        # Generate medical conditions based on age and demographics
        if patient_conditions is None:
            patient_conditions = self._generate_realistic_conditions(age, gender, bmi)
        
        # Generate medications based on conditions
        medications = self._generate_medications_for_conditions(patient_conditions)
        
        # Generate allergies
        num_allergies = np.random.choice([0, 1, 2, 3], p=[0.6, 0.25, 0.12, 0.03])
        allergies = [fake.allergy() for _ in range(num_allergies)]
        
        # Generate family history
        family_conditions = fake.random_elements(
            elements=['Heart Disease', 'Diabetes', 'Hypertension', 'Cancer', 'Stroke', 'Alzheimer\'s'],
            length=fake.random_int(0, 3),
            unique=True
        )
        
        # Social history
        social_history = {
            'smoking_status': fake.random_element(['Never', 'Former', 'Current', 'Unknown']),
            'alcohol_use': fake.random_element(['None', 'Occasional', 'Moderate', 'Heavy']),
            'exercise_frequency': fake.random_element(['Sedentary', 'Light', 'Moderate', 'Active']),
            'occupation': fake.job(),
            'marital_status': fake.random_element(['Single', 'Married', 'Divorced', 'Widowed'])
        }
        
        # Generate patient
        patient = SyntheticPatient(
            patient_id=f"PAT{fake.random_int(100000, 999999)}",
            first_name=fake.first_name_male() if gender == PatientGender.MALE else fake.first_name_female(),
            last_name=fake.last_name(),
            date_of_birth=fake.date_of_birth(minimum_age=age, maximum_age=age).isoformat(),
            gender=gender,
            age=age,
            height_cm=round(height_cm, 1),
            weight_kg=round(weight_kg, 1),
            bmi=round(bmi, 1),
            blood_type=fake.blood_type(),
            allergies=allergies,
            medications=medications,
            medical_history=[{'condition': condition, 'onset_date': fake.date_between(start_date='-10y').isoformat()} 
                           for condition in patient_conditions],
            family_history=family_conditions,
            social_history=social_history,
            insurance={
                'provider': fake.company(),
                'policy_number': fake.uuid4()[:8].upper(),
                'group_number': fake.random_int(1000, 9999)
            },
            emergency_contact={
                'name': fake.name(),
                'relationship': fake.random_element(['Spouse', 'Child', 'Parent', 'Sibling', 'Friend']),
                'phone': fake.phone_number()
            },
            primary_care_physician=f"Dr. {fake.name()}"
        )
        
        return patient
    
    def _generate_realistic_conditions(self, age: int, gender: PatientGender, bmi: float) -> List[str]:
        """Generate realistic medical conditions based on demographics"""
        conditions = []
        
        # Age-related condition probabilities
        age_conditions = {
            18: {'Asthma': 0.08, 'Depression': 0.06, 'Anxiety': 0.05},
            30: {'Hypertension': 0.05, 'Diabetes Mellitus Type 2': 0.02, 'Depression': 0.08},
            40: {'Hypertension': 0.15, 'Diabetes Mellitus Type 2': 0.05, 'Hyperlipidemia': 0.10},
            50: {'Hypertension': 0.25, 'Diabetes Mellitus Type 2': 0.10, 'Hyperlipidemia': 0.20, 'Heart Disease': 0.05},
            60: {'Hypertension': 0.35, 'Diabetes Mellitus Type 2': 0.15, 'Hyperlipidemia': 0.30, 'Heart Disease': 0.10, 'COPD': 0.05},
            70: {'Hypertension': 0.45, 'Diabetes Mellitus Type 2': 0.20, 'Heart Disease': 0.15, 'Osteoarthritis': 0.20, 'Chronic Kidney Disease': 0.08}
        }
        
        # BMI-related conditions
        if bmi > 30:  # Obese
            conditions.extend(['Obesity'])
            # Higher probability of diabetes and hypertension
        elif bmi > 25:  # Overweight
            # Moderate increase in metabolic conditions
            pass
        
        # Gender-specific conditions
        if gender == PatientGender.FEMALE:
            if age > 50:
                if np.random.random() < 0.15:
                    conditions.append('Osteoporosis')
            if age > 40:
                if np.random.random() < 0.10:
                    conditions.append('Hypothyroidism')
        
        # Sample conditions based on age
        age_bracket = min(70, max(18, (age // 10) * 10))
        if age_bracket in age_conditions:
            for condition, probability in age_conditions[age_bracket].items():
                if np.random.random() < probability:
                    conditions.append(condition)
        
        return list(set(conditions))  # Remove duplicates
    
    def _generate_medications_for_conditions(self, conditions: List[str]) -> List[Dict[str, Any]]:
        """Generate realistic medications based on medical conditions"""
        medications = []
        
        condition_medication_mapping = {
            'Hypertension': [
                {'name': 'Lisinopril', 'dosage': '10mg', 'frequency': 'daily'},
                {'name': 'Amlodipine', 'dosage': '5mg', 'frequency': 'daily'},
                {'name': 'Hydrochlorothiazide', 'dosage': '25mg', 'frequency': 'daily'}
            ],
            'Diabetes Mellitus Type 2': [
                {'name': 'Metformin', 'dosage': '1000mg', 'frequency': 'twice daily'},
                {'name': 'Insulin', 'dosage': '20 units', 'frequency': 'with meals'}
            ],
            'Hyperlipidemia': [
                {'name': 'Atorvastatin', 'dosage': '40mg', 'frequency': 'daily'},
                {'name': 'Simvastatin', 'dosage': '20mg', 'frequency': 'daily'}
            ],
            'Depression': [
                {'name': 'Sertraline', 'dosage': '50mg', 'frequency': 'daily'},
                {'name': 'Escitalopram', 'dosage': '10mg', 'frequency': 'daily'}
            ],
            'Anxiety': [
                {'name': 'Lorazepam', 'dosage': '0.5mg', 'frequency': 'as needed'},
                {'name': 'Alprazolam', 'dosage': '0.25mg', 'frequency': 'twice daily'}
            ],
            'Asthma': [
                {'name': 'Albuterol', 'dosage': '2 puffs', 'frequency': 'as needed'},
                {'name': 'Montelukast', 'dosage': '10mg', 'frequency': 'daily'}
            ],
            'Heart Disease': [
                {'name': 'Clopidogrel', 'dosage': '75mg', 'frequency': 'daily'},
                {'name': 'Metoprolol', 'dosage': '50mg', 'frequency': 'twice daily'}
            ]
        }
        
        for condition in conditions:
            if condition in condition_medication_mapping:
                condition_meds = condition_medication_mapping[condition]
                # Randomly select 1-2 medications for each condition
                selected_meds = fake.random_elements(
                    elements=condition_meds, 
                    length=fake.random_int(1, min(2, len(condition_meds))), 
                    unique=True
                )
                medications.extend(selected_meds)
        
        return medications
    
    def generate_lab_results(self, patient: SyntheticPatient, num_results: int = None) -> List[SyntheticLabResult]:
        """Generate realistic lab results for a patient"""
        if num_results is None:
            num_results = fake.random_int(5, 15)
        
        lab_results = []
        selected_tests = fake.random_elements(
            elements=list(self.lab_reference_ranges.keys()),
            length=num_results,
            unique=True
        )
        
        for test_name in selected_tests:
            test_info = self.lab_reference_ranges[test_name]
            
            # Determine if result should be abnormal based on patient conditions
            abnormal_probability = self._get_abnormal_probability(test_name, patient)
            is_abnormal = np.random.random() < abnormal_probability
            
            # Generate value based on normal/abnormal status
            value, reference_range, abnormal_flag = self._generate_lab_value(
                test_name, test_info, patient, is_abnormal
            )
            
            # Generate dates
            collection_date = fake.date_between(start_date='-30d', end_date='today')
            result_date = collection_date + timedelta(days=fake.random_int(0, 3))
            
            lab_result = SyntheticLabResult(
                test_name=test_name,
                value=round(value, 2),
                unit=test_info['unit'],
                reference_range=reference_range,
                abnormal_flag=abnormal_flag,
                collection_date=collection_date.isoformat(),
                result_date=result_date.isoformat()
            )
            
            lab_results.append(lab_result)
        
        return lab_results
    
    def _get_abnormal_probability(self, test_name: str, patient: SyntheticPatient) -> float:
        """Determine probability of abnormal result based on patient conditions"""
        base_probability = 0.05  # Base 5% chance of abnormal result
        
        condition_effects = {
            'Diabetes Mellitus Type 2': {
                'Glucose': 0.7, 'HbA1c': 0.8, 'Triglycerides': 0.3, 'Creatinine': 0.2
            },
            'Hypertension': {
                'Creatinine': 0.2, 'BUN': 0.15
            },
            'Hyperlipidemia': {
                'Total Cholesterol': 0.6, 'LDL Cholesterol': 0.7, 'Triglycerides': 0.4
            },
            'Heart Disease': {
                'Troponin I': 0.3, 'Total Cholesterol': 0.3, 'LDL Cholesterol': 0.4
            },
            'Chronic Kidney Disease': {
                'Creatinine': 0.8, 'BUN': 0.7
            }
        }
        
        patient_conditions = [item['condition'] for item in patient.medical_history]
        
        for condition in patient_conditions:
            if condition in condition_effects:
                if test_name in condition_effects[condition]:
                    return condition_effects[condition][test_name]
        
        return base_probability
    
    def _generate_lab_value(self, test_name: str, test_info: Dict, 
                          patient: SyntheticPatient, is_abnormal: bool) -> Tuple[float, str, Optional[str]]:
        """Generate a lab value with appropriate reference range"""
        
        # Determine reference range based on gender if applicable
        if 'male' in test_info and 'female' in test_info:
            if patient.gender == PatientGender.MALE:
                normal_range = test_info['male']
            else:
                normal_range = test_info['female']
            reference_range = f"{normal_range[0]}-{normal_range[1]} {test_info['unit']}"
        elif 'normal' in test_info:
            normal_range = test_info['normal']
            reference_range = f"{normal_range[0]}-{normal_range[1]} {test_info['unit']}"
        else:
            normal_range = (0, 100)  # Default fallback
            reference_range = f"{normal_range[0]}-{normal_range[1]} {test_info['unit']}"
        
        if is_abnormal:
            # Generate abnormal value
            if np.random.random() < 0.5:  # 50% chance high, 50% chance low
                # High abnormal
                value = np.random.uniform(normal_range[1] * 1.1, normal_range[1] * 2.0)
                abnormal_flag = "H"
            else:
                # Low abnormal
                value = np.random.uniform(max(0, normal_range[0] * 0.3), normal_range[0] * 0.9)
                abnormal_flag = "L"
        else:
            # Generate normal value
            value = np.random.uniform(normal_range[0], normal_range[1])
            abnormal_flag = None
        
        return value, reference_range, abnormal_flag
    
    def generate_clinical_note(self, patient: SyntheticPatient, 
                             note_type: DocumentType = DocumentType.PROGRESS_NOTE) -> SyntheticMedicalDocument:
        """Generate a realistic clinical note"""
        
        # Generate vital signs
        vitals = {
            'blood_pressure': f"{fake.vital_sign_value('systolic_bp'):.0f}/{fake.vital_sign_value('diastolic_bp'):.0f}",
            'heart_rate': f"{fake.vital_sign_value('heart_rate'):.0f}",
            'respiratory_rate': f"{fake.vital_sign_value('respiratory_rate'):.0f}",
            'temperature': f"{fake.vital_sign_value('temperature_f'):.1f}°F",
            'oxygen_saturation': f"{fake.vital_sign_value('oxygen_saturation'):.0f}%",
            'weight': f"{patient.weight_kg:.1f} kg",
            'height': f"{patient.height_cm:.0f} cm",
            'bmi': f"{patient.bmi:.1f}"
        }
        
        # Generate chief complaint and symptoms
        chief_complaint = self._generate_chief_complaint(patient)
        current_symptoms = fake.random_elements(elements=fake.symptom(), length=fake.random_int(1, 4))
        
        # Generate assessment and plan
        assessment = self._generate_assessment(patient)
        plan = self._generate_treatment_plan(patient)
        
        # Create clinical note content
        content = self._format_clinical_note(
            patient, note_type, chief_complaint, current_symptoms,
            vitals, assessment, plan
        )
        
        # Create document
        document = SyntheticMedicalDocument(
            document_id=f"DOC{fake.random_int(100000, 999999)}",
            document_type=note_type,
            patient_id=patient.patient_id,
            creation_date=fake.date_between(start_date='-7d', end_date='today').isoformat(),
            provider_name=f"Dr. {fake.name()}",
            facility=f"{fake.city()} Medical Center",
            content=content,
            structured_data={
                'vitals': vitals,
                'chief_complaint': chief_complaint,
                'symptoms': current_symptoms,
                'assessment': assessment,
                'plan': plan
            },
            metadata={
                'generated_by': 'VibeyBot Synthetic Data Generator',
                'version': '4.2.1',
                'generation_timestamp': datetime.now().isoformat()
            }
        )
        
        return document
    
    def _generate_chief_complaint(self, patient: SyntheticPatient) -> str:
        """Generate a chief complaint based on patient conditions"""
        condition_complaints = {
            'Diabetes Mellitus Type 2': ['Increased urination and thirst', 'Fatigue and blurred vision', 'Slow healing wounds'],
            'Hypertension': ['Headache', 'Dizziness', 'Routine blood pressure check'],
            'Heart Disease': ['Chest pain', 'Shortness of breath', 'Palpitations'],
            'Asthma': ['Difficulty breathing', 'Wheezing and cough', 'Chest tightness'],
            'Depression': ['Feeling sad and hopeless', 'Loss of interest in activities', 'Sleep problems'],
            'Anxiety': ['Feeling anxious and worried', 'Panic attacks', 'Difficulty concentrating']
        }
        
        patient_conditions = [item['condition'] for item in patient.medical_history]
        
        # If patient has conditions, use related complaint 70% of the time
        if patient_conditions and np.random.random() < 0.7:
            condition = np.random.choice(patient_conditions)
            if condition in condition_complaints:
                return np.random.choice(condition_complaints[condition])
        
        # Otherwise, use general complaints
        general_complaints = [
            'Routine follow-up', 'Annual physical exam', 'Medication refill',
            'Fatigue', 'Headache', 'Back pain', 'Joint pain', 'Skin rash',
            'Cold symptoms', 'Stomach upset', 'Sleep problems'
        ]
        
        return np.random.choice(general_complaints)
    
    def _generate_assessment(self, patient: SyntheticPatient) -> List[str]:
        """Generate medical assessment based on patient profile"""
        assessments = []
        
        # Include existing conditions
        for condition_info in patient.medical_history:
            condition = condition_info['condition']
            stability = np.random.choice(['stable', 'improving', 'worsening', 'controlled'], p=[0.6, 0.2, 0.1, 0.1])
            assessments.append(f"{condition}, {stability}")
        
        # Add potential new findings
        if np.random.random() < 0.3:  # 30% chance of new finding
            new_findings = [
                'Hypertensive blood pressure', 'Elevated glucose', 'Abnormal heart sounds',
                'Respiratory congestion', 'Skin lesion', 'Joint swelling',
                'Lymphadenopathy', 'Abdominal tenderness'
            ]
            assessments.append(np.random.choice(new_findings))
        
        return assessments if assessments else ['No acute distress']
    
    def _generate_treatment_plan(self, patient: SyntheticPatient) -> List[str]:
        """Generate treatment plan based on assessment"""
        plans = []
        
        # Medication management
        if patient.medications:
            plans.append('Continue current medications as prescribed')
            if np.random.random() < 0.2:  # 20% chance of medication adjustment
                med = np.random.choice(patient.medications)
                plans.append(f"Adjust {med['name']} dosage - follow up in 2 weeks")
        
        # Follow-up care
        follow_up_options = [
            'Return in 3 months for routine follow-up',
            'Schedule follow-up in 2 weeks',
            'Return PRN for worsening symptoms',
            'Annual labs and physical exam',
            'Follow up with specialist as needed'
        ]
        plans.append(np.random.choice(follow_up_options))
        
        # Lifestyle recommendations
        lifestyle_recommendations = [
            'Continue regular exercise and healthy diet',
            'Monitor blood pressure at home',
            'Limit sodium intake',
            'Weight management counseling',
            'Smoking cessation counseling',
            'Stress management techniques'
        ]
        if np.random.random() < 0.4:  # 40% chance of lifestyle recommendation
            plans.append(np.random.choice(lifestyle_recommendations))
        
        # Preventive care
        if patient.age > 50 and np.random.random() < 0.3:
            preventive_care = [
                'Mammogram screening',
                'Colonoscopy screening',
                'Bone density scan',
                'Cardiovascular risk assessment'
            ]
            plans.append(np.random.choice(preventive_care))
        
        return plans
    
    def _format_clinical_note(self, patient: SyntheticPatient, note_type: DocumentType,
                            chief_complaint: str, symptoms: List[str], vitals: Dict,
                            assessment: List[str], plan: List[str]) -> str:
        """Format the clinical note content"""
        
        content = f"""
PATIENT: {patient.last_name}, {patient.first_name}
DOB: {patient.date_of_birth}
MRN: {patient.patient_id}
DATE: {fake.date_between(start_date='-7d', end_date='today').strftime('%m/%d/%Y')}

CHIEF COMPLAINT:
{chief_complaint}

HISTORY OF PRESENT ILLNESS:
Patient is a {patient.age}-year-old {patient.gender.value} who presents with {chief_complaint.lower()}. 
Associated symptoms include {', '.join(symptoms) if symptoms else 'none reported'}.

PAST MEDICAL HISTORY:
{', '.join([item['condition'] for item in patient.medical_history]) if patient.medical_history else 'No significant past medical history'}

MEDICATIONS:
{chr(10).join([f"- {med['name']} {med['dosage']} {med['frequency']}" for med in patient.medications]) if patient.medications else 'None'}

ALLERGIES:
{', '.join(patient.allergies) if patient.allergies else 'NKDA'}

SOCIAL HISTORY:
Smoking: {patient.social_history['smoking_status']}
Alcohol: {patient.social_history['alcohol_use']}
Exercise: {patient.social_history['exercise_frequency']}

VITAL SIGNS:
BP: {vitals['blood_pressure']} mmHg
HR: {vitals['heart_rate']} bpm  
RR: {vitals['respiratory_rate']} /min
Temp: {vitals['temperature']}
O2 Sat: {vitals['oxygen_saturation']}
Weight: {vitals['weight']}
Height: {vitals['height']}
BMI: {vitals['bmi']}

PHYSICAL EXAMINATION:
General: Patient appears comfortable, no acute distress
HEENT: Normocephalic, atraumatic
Cardiovascular: Regular rate and rhythm, no murmurs
Pulmonary: Clear to auscultation bilaterally
Abdomen: Soft, non-tender, non-distended
Extremities: No edema, pulses intact
Neurological: Alert and oriented x3

ASSESSMENT AND PLAN:
{chr(10).join([f"{i+1}. {item}" for i, item in enumerate(assessment)])}

PLAN:
{chr(10).join([f"- {item}" for item in plan])}

Provider: {fake.name()}, MD
Dictated: {datetime.now().strftime('%m/%d/%Y %H:%M')}
"""
        
        return content.strip()
    
    def generate_batch_synthetic_data(self, 
                                    num_patients: int = 100,
                                    documents_per_patient: int = 3,
                                    include_lab_results: bool = True) -> Dict[str, List]:
        """Generate a complete batch of synthetic medical data"""
        
        logger.info(f"Generating {num_patients} synthetic patients with {documents_per_patient} documents each")
        
        all_patients = []
        all_documents = []
        all_lab_results = []
        
        for i in range(num_patients):
            if i % 10 == 0:
                logger.info(f"Generated {i}/{num_patients} patients...")
            
            # Generate patient
            patient = self.generate_synthetic_patient()
            all_patients.append(patient)
            
            # Generate documents for this patient
            document_types = [DocumentType.PROGRESS_NOTE, DocumentType.CLINICAL_NOTES, DocumentType.LAB_REPORT]
            for j in range(documents_per_patient):
                doc_type = np.random.choice(document_types)
                document = self.generate_clinical_note(patient, doc_type)
                all_documents.append(document)
            
            # Generate lab results
            if include_lab_results:
                lab_results = self.generate_lab_results(patient)
                all_lab_results.extend(lab_results)
        
        logger.info(f"Generated {len(all_patients)} patients, {len(all_documents)} documents, {len(all_lab_results)} lab results")
        
        return {
            'patients': all_patients,
            'documents': all_documents,
            'lab_results': all_lab_results
        }
    
    def save_synthetic_data(self, data: Dict[str, List], output_format: str = 'json'):
        """Save synthetic data to files"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_format.lower() == 'json':
            # Save as JSON files
            patients_file = self.output_dir / f'synthetic_patients_{timestamp}.json'
            documents_file = self.output_dir / f'synthetic_documents_{timestamp}.json'
            lab_results_file = self.output_dir / f'synthetic_lab_results_{timestamp}.json'
            
            with open(patients_file, 'w') as f:
                json.dump([asdict(patient) for patient in data['patients']], f, indent=2, default=str)
            
            with open(documents_file, 'w') as f:
                json.dump([asdict(doc) for doc in data['documents']], f, indent=2, default=str)
            
            with open(lab_results_file, 'w') as f:
                json.dump([asdict(result) for result in data['lab_results']], f, indent=2, default=str)
        
        elif output_format.lower() == 'csv':
            # Save as CSV files
            patients_df = pd.DataFrame([asdict(patient) for patient in data['patients']])
            documents_df = pd.DataFrame([asdict(doc) for doc in data['documents']])
            lab_results_df = pd.DataFrame([asdict(result) for result in data['lab_results']])
            
            patients_df.to_csv(self.output_dir / f'synthetic_patients_{timestamp}.csv', index=False)
            documents_df.to_csv(self.output_dir / f'synthetic_documents_{timestamp}.csv', index=False)
            lab_results_df.to_csv(self.output_dir / f'synthetic_lab_results_{timestamp}.csv', index=False)
        
        logger.info(f"Synthetic data saved to {self.output_dir} in {output_format} format")

def generate_vibey_training_data(num_patients: int = 1000, 
                               config: Dict = None) -> Dict[str, List]:
    """
    Main function to generate VibeyBot training data
    
    Args:
        num_patients: Number of synthetic patients to generate
        config: Generation configuration
        
    Returns:
        Dictionary containing all generated data
    """
    generator = VibeyMedicalSyntheticDataGenerator(config)
    
    # Generate synthetic data
    synthetic_data = generator.generate_batch_synthetic_data(
        num_patients=num_patients,
        documents_per_patient=3,
        include_lab_results=True
    )
    
    # Save the data
    generator.save_synthetic_data(synthetic_data, output_format='json')
    generator.save_synthetic_data(synthetic_data, output_format='csv')
    
    logger.info(f"VibeyBot synthetic training data generation completed")
    logger.info(f"Generated data for {len(synthetic_data['patients'])} patients")
    logger.info(f"Compatible with VibeyBot medical AI training pipeline")
    
    return synthetic_data

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='VibeyBot Synthetic Medical Data Generator')
    parser.add_argument('--patients', type=int, default=100, help='Number of patients to generate')
    parser.add_argument('--output-dir', type=str, default='synthetic_data', help='Output directory')
    parser.add_argument('--format', choices=['json', 'csv', 'both'], default='both', help='Output format')
    
    args = parser.parse_args()
    
    config = {
        'output_dir': args.output_dir
    }
    
    # Generate synthetic data
    data = generate_vibey_training_data(args.patients, config)
    
    print(f"Generated synthetic medical data for VibeyBot AI training:")
    print(f"- {len(data['patients'])} patients")
    print(f"- {len(data['documents'])} medical documents") 
    print(f"- {len(data['lab_results'])} lab results")
    print(f"- Output saved to: {args.output_dir}")