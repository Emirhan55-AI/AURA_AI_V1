"""
Enhanced API with improved classification features
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import logging
from typing import List, Dict, Optional
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the enhanced classifier
try:
    from models.enhanced_classifier import EnhancedClothingClassifier
    USE_ENHANCED = True
except ImportError:
    from models.classifier import ClothingClassifier
    USE_ENHANCED = False
    logging.warning("Using fallback classifier")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Aura AI - Enhanced Clothing Classification API",
    description="Advanced AI-powered clothing classification with confidence scores",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize classifier
if USE_ENHANCED:
    classifier = EnhancedClothingClassifier()
    logger.info("Enhanced classifier initialized")
else:
    classifier = ClothingClassifier()
    logger.info("Fallback classifier initialized")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    classifier_info = classifier.get_model_info() if hasattr(classifier, 'get_model_info') else {
        "model_name": "google/vit-base-patch16-224",
        "num_categories": 10,
        "supports_confidence": False,
        "supports_top_k": False
    }
    
    return {
        "message": "Aura AI - Enhanced Clothing Classification API",
        "version": "2.0.0",
        "status": "running",
        "classifier": {
            "type": "enhanced" if USE_ENHANCED else "basic",
            **classifier_info
        },
        "endpoints": {
            "/": "API information",
            "/health": "Health check",
            "/classify": "Single image classification",
            "/classify-enhanced": "Enhanced classification with confidence scores"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "classifier_loaded": classifier is not None,
        "enhanced_features": USE_ENHANCED
    }

@app.post("/classify")
async def classify_clothing(file: UploadFile = File(...)):
    """
    Basic clothing classification (backward compatibility)
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Classify based on available classifier
        if USE_ENHANCED:
            predictions = classifier.classify_clothing_enhanced(image, top_k=1)
            result = predictions[0] if predictions else {"label": "unknown", "confidence": 0.0}
            prediction = result["label"]
        else:
            prediction = classifier.classify_clothing(image)
            result = {"label": prediction, "confidence": 1.0}
        
        return {
            "status": "success",
            "prediction": prediction,
            "confidence": result.get("confidence", 1.0),
            "filename": file.filename
        }
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.post("/classify-enhanced")
async def classify_clothing_enhanced(
    file: UploadFile = File(...),
    top_k: int = Query(3, ge=1, le=10, description="Number of top predictions to return"),
    min_confidence: float = Query(0.1, ge=0.0, le=1.0, description="Minimum confidence threshold")
):
    """
    Enhanced clothing classification with multiple predictions and confidence scores
    """
    try:
        # Check if enhanced classifier is available
        if not USE_ENHANCED:
            raise HTTPException(
                status_code=501, 
                detail="Enhanced classification not available. Using basic classifier."
            )
        
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Get enhanced predictions
        predictions = classifier.classify_clothing_enhanced(
            image, 
            top_k=top_k, 
            min_confidence=min_confidence
        )
        
        # Calculate total confidence
        total_confidence = sum(p["confidence"] for p in predictions)
        
        return {
            "status": "success",
            "filename": file.filename,
            "predictions": predictions,
            "summary": {
                "top_prediction": predictions[0]["label"] if predictions else "unknown",
                "num_predictions": len(predictions),
                "total_confidence": round(total_confidence, 4),
                "model_info": classifier.get_model_info()
            },
            "parameters": {
                "top_k": top_k,
                "min_confidence": min_confidence
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Enhanced classification failed: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get detailed information about the current model"""
    if hasattr(classifier, 'get_model_info'):
        return classifier.get_model_info()
    else:
        return {
            "model_name": "google/vit-base-patch16-224",
            "device": "cpu",
            "num_categories": 10,
            "categories": ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
                          "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
            "supports_confidence": False,
            "supports_top_k": False
        }

@app.get("/categories")
async def get_categories():
    """Get all available clothing categories"""
    if hasattr(classifier, 'get_model_info'):
        info = classifier.get_model_info()
        return {
            "categories": info.get("categories", []),
            "total": len(info.get("categories", []))
        }
    else:
        categories = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
                     "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
        return {
            "categories": categories,
            "total": len(categories)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
