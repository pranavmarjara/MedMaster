#!/usr/bin/env python3
"""
VibeyBot AI Service Launcher
Starts the medical AI analysis service on port 8000
"""

import uvicorn
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    print("🚀 Starting VibeyBot Medical AI Service...")
    print("📊 Loading medical analysis models...")
    print("🏥 Service will be available at http://localhost:8000")
    
    uvicorn.run(
        "medical_ai_service:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )