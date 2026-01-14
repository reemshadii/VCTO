"""
Personal Color Analysis API - Production Version
Includes ALL optimizations from the roadmap
"""

import cv2
import numpy as np
import time
import logging
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import all enhanced modules
from vision_engine_v2 import FeatureExtractorV2
from color_logic_v2 import PersonalColorAnalystV2
from validation_utils import (
    detect_environmental_issues_v2,
    calculate_overall_confidence
)

# Setup logging
logging.basicConfig(
    filename='analysis_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = FastAPI(
    title="Personal Color Analysis API",
    description="AI-powered seasonal color analysis with computer vision",
    version="2.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analyzers
print("🚀 Initializing Personal Color Analysis System V2.0...")
print("   ✓ Step 1.2: Validation Metrics - ENABLED")
print("   ✓ Step 2.1: 9-Zone Skin Sampling - ENABLED")
print("   ✓ Step 2.2: Multi-Zone Hair Detection - ENABLED")
print("   ✓ Step 2.3: Enhanced Eye Color - ENABLED")
print("   ✓ Step 3.1: Temperature Scoring V2 - ENABLED")
print("   ✓ Step 3.2: Contrast Calculation V2 - ENABLED")
print("   ✓ Step 3.3: Fuzzy Decision Tree - ENABLED")
print("   ✓ Step 4.1: Environmental Validation - ENABLED")
print("   ✓ Step 4.2: Confidence Scoring - ENABLED")

extractor = FeatureExtractorV2()
analyst = PersonalColorAnalystV2()

# Analytics tracking
analytics = {
    'total_analyses': 0,
    'average_confidence': 0,
    'common_seasons': {},
    'validation_issues_count': 0
}


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "service": "Personal Color Analysis API",
        "version": "2.0",
        "status": "operational",
        "features": [
            "✓ Enhanced 9-zone skin sampling",
            "✓ Multi-zone hair detection with outlier rejection",
            "✓ Improved eye color filtering",
            "✓ Multi-dimensional temperature scoring",
            "✓ Enhanced contrast calculation",
            "✓ Fuzzy season boundaries",
            "✓ Confidence tracking",
            "✓ Environmental validation",
            "✓ Real-time quality checks"
        ],
        "endpoints": {
            "/analyze": "POST - Analyze image for color season",
            "/health": "GET - System health check",
            "/stats": "GET - System statistics"
        }
    }


@app.get("/health")
async def health_check():
    """System health check"""
    return {
        "status": "healthy",
        "version": "2.0",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "feature_extractor": "FeatureExtractorV2",
            "color_analyst": "PersonalColorAnalystV2",
            "validation": "Enhanced"
        },
        "total_analyses": analytics['total_analyses']
    }


@app.get("/stats")
async def get_statistics():
    """Get system statistics"""
    return {
        "total_analyses": analytics['total_analyses'],
        "average_confidence": analytics['average_confidence'],
        "most_common_seasons": dict(sorted(
            analytics['common_seasons'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]),
        "validation_issues_encountered": analytics['validation_issues_count']
    }


@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    is_dyed: bool = Form(False)
):
    """
    Analyze uploaded image for personal color season
    
    IMPLEMENTED IMPROVEMENTS:
    - Step 2.1: Enhanced skin sampling (9 zones)
    - Step 2.2: Multi-zone hair detection
    - Step 2.3: Improved eye color extraction
    - Step 3.1: Multi-dimensional temperature scoring
    - Step 3.2: Enhanced contrast calculation
    - Step 3.3: Fuzzy decision boundaries
    - Step 4.1: Environmental validation
    - Step 4.2: Confidence scoring
    
    Parameters:
    - file: Image file (JPG, PNG)
    - is_dyed: Boolean indicating if hair is dyed
    
    Returns:
    - subseason_key: 12-season classification
    - undertone: Warm/Cool/Neutral
    - contrast: High/Medium/Low
    - confidence: Overall confidence assessment
    - validation_note: Quality warnings
    - details: Detailed analysis breakdown
    """
    
    start_time = time.time()
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file. Please upload a valid JPG or PNG."
            )
        
        logging.info(f"Analysis started - File: {file.filename}, Dyed: {is_dyed}")
        
        # ==================================================================
        # STEP 2: ENHANCED FEATURE EXTRACTION
        # ==================================================================
        
        try:
            features = extractor.extract_features_from_memory(img, is_dyed=is_dyed)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Face detection failed: {str(e)}. Please ensure image shows a clear front-facing face."
            )
        
        skin_lab = features['skin_lab']
        hair_lab = features['hair_lab']
        eye_lab = features['eye_lab']
        skin_samples = features['skin_samples']
        
        # ==================================================================
        # STEP 4.1: ENVIRONMENTAL VALIDATION
        # ==================================================================
        
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        validation_issues = detect_environmental_issues_v2(
            skin_samples,
            image_rgb,
            skin_lab
        )
        
        # Track validation issues
        if validation_issues:
            analytics['validation_issues_count'] += 1
        
        # ==================================================================
        # STEP 3: ENHANCED COLOR ANALYSIS
        # ==================================================================
        
        results = analyst.analyze_from_lab(skin_lab, hair_lab, eye_lab)
        
        # ==================================================================
        # STEP 4.2: CONFIDENCE CALCULATION
        # ==================================================================
        
        temp_confidence = results['Details']['temperature']['confidence']
        contrast_confidence = results['Details']['contrast']['confidence']
        
        confidence_data = calculate_overall_confidence(
            temp_confidence,
            contrast_confidence,
            validation_issues
        )
        
        # ==================================================================
        # FORMAT RESPONSE
        # ==================================================================
        
        # Format validation message
        if validation_issues:
            critical_issues = [msg for sev, msg in validation_issues if sev == "CRITICAL"]
            warning_issues = [msg for sev, msg in validation_issues if sev == "WARNING"]
            info_issues = [msg for sev, msg in validation_issues if sev == "INFO"]
            
            validation_msg = ""
            if critical_issues:
                validation_msg = "⚠️ CRITICAL: " + " | ".join(critical_issues)
            elif warning_issues:
                validation_msg = "⚠️ WARNING: " + " | ".join(warning_issues)
            elif info_issues:
                validation_msg = "ℹ️ INFO: " + " | ".join(info_issues)
        else:
            validation_msg = "✅ No issues detected - excellent image quality"
        
        # Update analytics
        analytics['total_analyses'] += 1
        
        # Update average confidence
        prev_avg = analytics['average_confidence']
        total = analytics['total_analyses']
        analytics['average_confidence'] = (prev_avg * (total - 1) + confidence_data['score']) / total
        
        # Track season frequencies
        season_key = results['Sub_Season']
        analytics['common_seasons'][season_key] = analytics['common_seasons'].get(season_key, 0) + 1
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Log successful analysis
        logging.info(f"""
        Analysis completed:
          Result: {season_key}
          Confidence: {confidence_data['level']} ({confidence_data['score']}/100)
          Temperature: {results['Temperature_Category']} (score: {results['Temperature_Score']})
          Contrast: {results['Contrast_Category']} ({results['Contrast_Value']:.1f})
          Issues: {len(validation_issues)}
          Processing time: {processing_time:.2f}s
        """)
        
        # Build comprehensive response
        response = {
            "status": "success",
            "subseason_key": season_key,
            "undertone": results['Temperature_Category'],
            "contrast": results['Contrast_Category'],
            "confidence": {
                "level": confidence_data['level'],
                "score": confidence_data['score'],
                "recommendation": confidence_data['recommendation']
            },
            "validation_note": validation_msg,
            "validation_issues": [
                {"severity": sev, "message": msg}
                for sev, msg in validation_issues
            ],
            "details": {
                "temperature_score": results['Temperature_Score'],
                "temperature_factors": [
                    f"{factor[0]}: {factor[1]:+d} ({factor[2]} confidence)"
                    for factor in results['Details']['temperature']['factors']
                ],
                "contrast_value": results['Contrast_Value'],
                "contrast_breakdown": {
                    "luminance_contrast": results['Details']['contrast']['l_contrast'],
                    "chroma_contrast": results['Details']['contrast']['chroma_contrast']
                },
                "season_scores": results['Details']['season_scores'],
                "color_values": {
                    "skin": {
                        "L": round(float(skin_lab[0]), 2),
                        "a": round(float(skin_lab[1]), 2),
                        "b": round(float(skin_lab[2]), 2)
                    },
                    "hair": {
                        "L": round(float(hair_lab[0]), 2),
                        "a": round(float(hair_lab[1]), 2),
                        "b": round(float(hair_lab[2]), 2)
                    },
                    "eye": {
                        "L": round(float(eye_lab[0]), 2),
                        "a": round(float(eye_lab[1]), 2),
                        "b": round(float(eye_lab[2]), 2)
                    }
                },
                "num_skin_samples": len(skin_samples),
                "processing_time_seconds": round(processing_time, 3)
            },
            "metadata": {
                "version": "2.0",
                "timestamp": datetime.now().isoformat(),
                "features_used": [
                    "9-zone skin sampling",
                    "Multi-zone hair detection",
                    "Enhanced eye color filtering",
                    "Multi-dimensional temperature scoring",
                    "Chroma-weighted contrast",
                    "Fuzzy season boundaries",
                    "Environmental validation"
                ]
            }
        }
        
        return JSONResponse(content=response)
    
    except HTTPException:
        raise
    
    except Exception as e:
        logging.error(f"Analysis error: {str(e)}", exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "detail": "Analysis failed. Please ensure the image shows a clear front-facing face with good lighting."
            }
        )


@app.post("/feedback")
async def submit_feedback(
    analysis_id: str = Form(...),
    predicted_season: str = Form(...),
    actual_season: str = Form(None),
    user_rating: int = Form(None),
    comments: str = Form(None)
):
    """
    Collect user feedback for continuous improvement
    Step 5.1: Systematic Testing Protocol
    """
    
    feedback_entry = {
        'timestamp': datetime.now().isoformat(),
        'analysis_id': analysis_id,
        'predicted': predicted_season,
        'user_correction': actual_season,
        'rating': user_rating,
        'comments': comments
    }
    
    # Log feedback
    logging.info(f"Feedback received: {feedback_entry}")
    
    # Save to feedback file
    import json
    try:
        with open('feedback_log.json', 'a') as f:
            json.dump(feedback_entry, f)
            f.write('\n')
    except Exception as e:
        logging.error(f"Failed to save feedback: {e}")
    
    return {
        "status": "success",
        "message": "Thank you for your feedback! This helps us improve."
    }


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("🎨 Personal Color Analysis API V2.0")
    print("="*70)
    print("\nImplemented Improvements:")
    print("  ✓ Step 1.2: Validation Metrics")
    print("  ✓ Step 2.1: Enhanced Skin Sampling (9 zones)")
    print("  ✓ Step 2.2: Multi-Zone Hair Detection")
    print("  ✓ Step 2.3: Eye Color Enhancement")
    print("  ✓ Step 3.1: Temperature Scoring V2")
    print("  ✓ Step 3.2: Contrast Calculation V2")
    print("  ✓ Step 3.3: Fuzzy Decision Tree")
    print("  ✓ Step 4.1: Environmental Validation")
    print("  ✓ Step 4.2: Confidence Scoring")
    print("  ✓ Step 5.1: Testing & Feedback System")
    print("\nStarting server on http://localhost:8000")
    print("API documentation: http://localhost:8000/docs")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
