import cv2
import numpy as np
import time
import logging
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from vision_engine_v2 import FeatureExtractorV2
from color_logic_v2 import PersonalColorAnalystV2
from validation_utils import detect_environmental_issues_v2, calculate_overall_confidence

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

@app.options("/analyze")
async def options_analyze():
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

extractor = FeatureExtractorV2()
analyst = PersonalColorAnalystV2()

analytics = {
    'total_analyses': 0,
    'average_confidence': 0,
    'common_seasons': {},
    'validation_issues_count': 0
}

@app.get("/")
async def root():
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
            "/stats": "GET - System statistics",
            "/generate-tryon": "POST - Generate virtual try-on with color variants"
        }
    }

@app.get("/health")
async def health_check():
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
    start_time = time.time()
    
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file. Please upload a valid JPG or PNG."
            )
        
        logging.info(f"Analysis started - File: {file.filename}, Dyed: {is_dyed}")
        
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
        
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        validation_issues = detect_environmental_issues_v2(skin_samples, image_rgb, skin_lab)
        
        if validation_issues:
            analytics['validation_issues_count'] += 1
        
        results = analyst.analyze_from_lab(skin_lab, hair_lab, eye_lab)
        
        temp_confidence = results['Details']['temperature']['confidence']
        contrast_confidence = results['Details']['contrast']['confidence']
        
        confidence_data = calculate_overall_confidence(
            temp_confidence,
            contrast_confidence,
            validation_issues
        )
        
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
        
        analytics['total_analyses'] += 1
        prev_avg = analytics['average_confidence']
        total = analytics['total_analyses']
        analytics['average_confidence'] = (prev_avg * (total - 1) + confidence_data['score']) / total
        
        season_key = results['Sub_Season']
        analytics['common_seasons'][season_key] = analytics['common_seasons'].get(season_key, 0) + 1
        
        processing_time = time.time() - start_time
        
        logging.info(f"""
        Analysis completed:
          Result: {season_key}
          Confidence: {confidence_data['level']} ({confidence_data['score']}/100)
          Temperature: {results['Temperature_Category']} (score: {results['Temperature_Score']})
          Contrast: {results['Contrast_Category']} ({results['Contrast_Value']:.1f})
          Issues: {len(validation_issues)}
          Processing time: {processing_time:.2f}s
        """)
        
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

# ============================================================================
# VIRTUAL TRY-ON ENDPOINT - INTEGRATE YOUR MODEL HERE
# ============================================================================
@app.post("/generate-tryon")
async def generate_tryon(
    user_image: UploadFile = File(...),
    clothing_image: UploadFile = File(...),
    colors: str = Form(...)
):
    """
    Generate virtual try-on with color variants
    
    Parameters:
    - user_image: User's photo (from Step 1)
    - clothing_image: Clothing item (from Step 3)
    - colors: Comma-separated hex colors (e.g., "#6495ED,#915F6D,#FF00FF")
    
    Returns:
    - generated_images: List of base64-encoded images, one for each color
    
    YOUR MODEL INTEGRATION:
    1. Read both images:
       user_img = cv2.imdecode(np.frombuffer(await user_image.read(), np.uint8), cv2.IMREAD_COLOR)
       clothing_img = cv2.imdecode(np.frombuffer(await clothing_image.read(), np.uint8), cv2.IMREAD_COLOR)
    
    2. Parse colors:
       color_list = colors.split(',')  # ["#6495ED", "#915F6D", "#FF00FF"]
    
    3. For each color:
       - Recolor the clothing_img to that hex color
       - Run your virtual try-on model: model(user_img, recolored_clothing_img)
       - Get output image
    
    4. Return all generated images as base64:
       results = []
       for output_img in generated_images:
           _, buffer = cv2.imencode('.jpg', output_img)
           base64_img = base64.b64encode(buffer).decode('utf-8')
           results.append({
               "color": color,
               "image": f"data:image/jpeg;base64,{base64_img}"
           })
       return {"status": "success", "generated_images": results}
    """
    
    return JSONResponse(
        content={
            "status": "success",
            "message": "Model integration pending - replace this with your virtual try-on model",
            "received_colors": colors.split(','),
            "next_steps": "Integrate your model in this endpoint to generate actual images"
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("🎨 Personal Color Analysis API V2.0")
    print("="*70)
    print("\nStarting server on http://localhost:8000")
    print("API documentation: http://localhost:8000/docs")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
