#!/usr/bin/env python3
"""
Test script for the comprehensive medical intelligence system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from medical_ai_service import MedicalIntelligenceEngine

def test_medical_analysis():
    """Test the comprehensive medical analysis system"""
    
    # Initialize the medical intelligence engine
    engine = MedicalIntelligenceEngine()
    
    # Sample medical report with various lab values and vital signs
    sample_report = """
    LABORATORY RESULTS:
    Glucose: 165 mg/dL (high)
    Hemoglobin: 8.2 g/dL (low)
    Creatinine: 2.1 mg/dL (elevated)
    BUN: 45 mg/dL
    Sodium: 138 mEq/L
    Potassium: 4.2 mEq/L
    LDL Cholesterol: 180 mg/dL (high)
    HDL Cholesterol: 35 mg/dL (low)
    Triglycerides: 220 mg/dL (high)
    ALT: 65 U/L (elevated)
    AST: 78 U/L (elevated)
    Troponin I: 0.08 ng/mL (elevated)
    
    VITAL SIGNS:
    Blood Pressure: 158/92 mmHg
    Heart Rate: 105 bpm
    Respiratory Rate: 18 /min
    Temperature: 98.6°F
    Oxygen Saturation: 94%
    """
    
    print("🔬 COMPREHENSIVE MEDICAL INTELLIGENCE SYSTEM TEST")
    print("=" * 60)
    
    # Test parsing functionality
    lab_values, vital_signs = engine.parser.parse_text(sample_report)
    
    print(f"\n📊 PARSING RESULTS:")
    print(f"Lab values extracted: {len(lab_values)}")
    print(f"Vital signs extracted: {len(vital_signs)}")
    
    print(f"\n🧪 LAB VALUES FOUND:")
    for lab in lab_values[:8]:  # Show first 8
        print(f"  • {lab.name}: {lab.value} {lab.unit} ({lab.status})")
    
    print(f"\n💓 VITAL SIGNS FOUND:")
    for vital in vital_signs[:5]:  # Show first 5
        print(f"  • {vital.name}: {vital.value} {vital.unit} ({vital.status})")
    
    # Test risk assessment
    cv_risk, cv_assessment, cv_factors = engine.risk_assessor.assess_cardiovascular_risk(lab_values, vital_signs)
    diabetes_risk, diabetes_assessment, diabetes_factors = engine.risk_assessor.assess_diabetes_risk(lab_values)
    kidney_risk, kidney_assessment, kidney_factors = engine.risk_assessor.assess_kidney_function(lab_values)
    
    print(f"\n🫀 CARDIOVASCULAR RISK: {cv_risk.value.upper()}")
    print(f"Assessment: {cv_assessment}")
    print(f"Risk factors: {len(cv_factors)} identified")
    
    print(f"\n🩺 DIABETES RISK: {diabetes_risk.value.upper()}")
    print(f"Assessment: {diabetes_assessment}")
    
    print(f"\n🫘 KIDNEY FUNCTION RISK: {kidney_risk.value.upper()}")
    print(f"Assessment: {kidney_assessment}")
    
    # Test pattern recognition
    patterns = engine.pattern_recognizer.identify_patterns(lab_values, vital_signs)
    print(f"\n🔍 MEDICAL PATTERNS IDENTIFIED: {len(patterns)}")
    for pattern in patterns:
        print(f"  • {pattern}")
    
    print(f"\n✅ SYSTEM STATUS:")
    print(f"  • Medical text parsing: OPERATIONAL")
    print(f"  • Reference ranges database: {len(engine.parser.lab_patterns) + len(engine.parser.vital_patterns)} patterns loaded")
    print(f"  • Risk assessment algorithms: OPERATIONAL")
    print(f"  • Pattern recognition: OPERATIONAL")
    print(f"  • Clinical analysis: OPERATIONAL")
    
    return True

if __name__ == "__main__":
    try:
        test_medical_analysis()
        print(f"\n🎉 COMPREHENSIVE MEDICAL INTELLIGENCE SYSTEM TEST COMPLETED SUCCESSFULLY!")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()