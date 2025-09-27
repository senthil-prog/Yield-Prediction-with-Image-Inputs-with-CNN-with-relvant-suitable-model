#!/usr/bin/env python3
"""
Quick execution script for Crop Yield Prediction
Run this file to execute the complete pipeline
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

def main():
    """Main execution function"""
    print("🚀 CROP YIELD PREDICTION - QUICK START")
    print("=" * 45)
    
    try:
        # Import and run the pipeline
        from yield_predictor import run_complete_pipeline
        
        print("⏳ Starting complete pipeline...")
        print("📊 This will take approximately 15-25 minutes...")
        print("🔥 Make sure you have TensorFlow and other dependencies installed!")
        
        # Run the complete pipeline
        predictor, results = run_complete_pipeline()
        
        print(f"\n✅ SUCCESS! Check the 'results' folder for outputs.")
        
        return predictor, results
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        return None, None
        
    except Exception as e:
        print(f"❌ Execution Error: {e}")
        print("💡 Check the console output and log files for details")
        return None, None

if __name__ == "__main__":
    predictor, results = main()