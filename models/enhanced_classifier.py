"""
Enhanced Fashion Classifier with improved model and features
"""
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import logging
from typing import List, Dict, Tuple, Optional
import requests
from io import BytesIO

# Import our custom fashion classifier
from .fashion_classifier import FashionMNISTClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedClothingClassifier:
    def __init__(self, use_custom_fashion: bool = True):
        """
        Initialize the enhanced clothing classifier
        
        Args:
            use_custom_fashion: Whether to use custom fashion classifier
        """
        self.use_custom_fashion = use_custom_fashion
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        if use_custom_fashion:
            logger.info("Using custom Fashion-MNIST classifier")
            self.fashion_classifier = FashionMNISTClassifier()
            self.model_name = "Custom Fashion-MNIST Classifier"
        else:
            logger.info("Using HuggingFace model")
            self.fashion_classifier = None
            self.model_name = "google/vit-base-patch16-224"  # Fallback
            self._load_huggingface_model()
    
    def _load_huggingface_model(self):
        """Load HuggingFace model (fallback)"""
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Get model labels if available
            if hasattr(self.model.config, 'id2label') and self.model.config.id2label:
                self.labels = self.model.config.id2label
                logger.info(f"Model supports {len(self.labels)} categories")
            else:
                # Fashion-MNIST labels (fallback)
                self.labels = {
                    0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
                    5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"
                }
                logger.info("Using Fashion-MNIST labels")
                
        except Exception as e:
            logger.error(f"Error loading HuggingFace model: {e}")
            raise
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for model input (only needed for HuggingFace models)
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed image tensor
        """
        try:
            if not hasattr(self, 'processor'):
                raise ValueError("Processor not available")
                
            # Ensure image is RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Process image
            inputs = self.processor(images=image, return_tensors="pt")
            return inputs.pixel_values.to(self.device)
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def classify_clothing_enhanced(
        self, 
        image: Image.Image, 
        top_k: int = 3,
        min_confidence: float = 0.1
    ) -> List[Dict[str, any]]:
        """
        Enhanced clothing classification with multiple predictions
        
        Args:
            image: PIL Image object
            top_k: Number of top predictions to return
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of predictions with labels, confidence scores and categories
        """
        try:
            if self.use_custom_fashion and self.fashion_classifier:
                # Use custom fashion classifier
                return self.fashion_classifier.classify_clothing_enhanced(
                    image, top_k=top_k, min_confidence=min_confidence
                )
            else:
                # Use HuggingFace model (fallback)
                return self._classify_with_huggingface(image, top_k, min_confidence)
                
        except Exception as e:
            logger.error(f"Error classifying clothing: {e}")
            raise
    
    def _classify_with_huggingface(self, image: Image.Image, top_k: int, min_confidence: float):
        """Classify using HuggingFace model"""
        # Preprocess image
        pixel_values = self.preprocess_image(image)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(pixel_values)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, min(top_k, len(self.labels)))
        
        predictions = []
        for i in range(len(top_probs[0])):
            prob = top_probs[0][i].item()
            idx = top_indices[0][i].item()
            
            # Skip low confidence predictions
            if prob < min_confidence:
                continue
            
            # Map index to label
            if idx in self.labels:
                label = self.labels[idx]
            else:
                # For models with many classes, try to map intelligently
                if len(self.labels) == 10:  # Fashion-MNIST case
                    mapped_idx = idx % 10
                    label = self.labels.get(mapped_idx, f"Fashion_Category_{mapped_idx}")
                else:
                    label = f"Category_{idx}"
            
            prediction = {
                "label": label,
                "confidence": round(prob, 4),
                "category_id": idx,
                "percentage": round(prob * 100, 2)
            }
            predictions.append(prediction)
        
        # If no predictions above threshold, return top prediction
        if not predictions:
            prob = top_probs[0][0].item()
            idx = top_indices[0][0].item()
            
            # Map index to label
            if idx in self.labels:
                label = self.labels[idx]
            else:
                if len(self.labels) == 10:  # Fashion-MNIST case
                    mapped_idx = idx % 10
                    label = self.labels.get(mapped_idx, f"Fashion_Category_{mapped_idx}")
                else:
                    label = f"Category_{idx}"
            
            predictions = [{
                "label": label,
                "confidence": round(prob, 4),
                "category_id": idx,
                "percentage": round(prob * 100, 2)
            }]
        
        logger.info(f"Classified image with {len(predictions)} predictions")
        return predictions
    
    def classify_from_url(self, image_url: str, **kwargs) -> List[Dict[str, any]]:
        """
        Classify clothing from image URL
        
        Args:
            image_url: URL of the image
            **kwargs: Additional arguments for classify_clothing_enhanced
            
        Returns:
            List of predictions
        """
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            
            image = Image.open(BytesIO(response.content))
            return self.classify_clothing_enhanced(image, **kwargs)
            
        except Exception as e:
            logger.error(f"Error classifying from URL: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the current model"""
        if self.use_custom_fashion and self.fashion_classifier:
            return self.fashion_classifier.get_model_info()
        else:
            return {
                "model_name": self.model_name,
                "device": str(self.device),
                "num_categories": len(self.labels) if hasattr(self, 'labels') else 0,
                "categories": list(self.labels.values()) if hasattr(self, 'labels') else [],
                "supports_confidence": True,
                "supports_top_k": True
            }

# Test function
def test_enhanced_classifier():
    """Test the enhanced classifier"""
    try:
        classifier = EnhancedClothingClassifier()
        
        # Get model info
        info = classifier.get_model_info()
        print("🤖 Model Info:")
        print(f"   Model: {info['model_name']}")
        print(f"   Device: {info['device']}")
        print(f"   Categories: {info['num_categories']}")
        if info['categories']:
            print(f"   Available categories: {info['categories'][:5]}...")
        
        return classifier
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None

if __name__ == "__main__":
    test_enhanced_classifier()
