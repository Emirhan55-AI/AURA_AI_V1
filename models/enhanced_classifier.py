"""
Enhanced Fashion Classifier with State-of-the-Art Fashion Models
Supports multiple SOTA fashion classification models for optimal accuracy
"""
from transformers import (
    AutoImageProcessor, 
    AutoModelForImageClassification,
    AutoModel,
    AutoProcessor,
    AutoModelForObjectDetection
)
from PIL import Image
import torch
import torch.nn.functional as F
import logging
from typing import List, Dict, Tuple, Optional
import requests
from io import BytesIO
import numpy as np

# Import our custom fashion classifier
from .fashion_classifier import FashionMNISTClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedClothingClassifier:
    def __init__(self, model_name: str = "marqo-fashionsiglip"):
        """
        Initialize the enhanced clothing classifier with multiple model support
        
        Args:
            model_name: Model to use. Options:
                - "marqo-fashionsiglip": SOTA fashion zero-shot classifier
                - "fashion-object-detection": Specialized fashion object detection
                - "custom-fashion": Custom rule-based classifier
                - "fallback": Traditional ImageNet model
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Initializing model: {model_name}")
        
        # Initialize the appropriate model
        self._load_model(model_name)
    
    def _load_model(self, model_name: str):
        """Load the specified model"""
        try:
            if model_name == "marqo-fashionsiglip":
                self._load_marqo_fashion_model()
            elif model_name == "fashion-object-detection":
                self._load_fashion_detection_model()
            elif model_name == "custom-fashion":
                self._load_custom_fashion_model()
            else:  # fallback
                self._load_fallback_model()
                
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            logger.info("Falling back to custom fashion model")
            self._load_custom_fashion_model()
    
    def _load_marqo_fashion_model(self):
        """Load Marqo Fashion SigLIP model - SOTA for fashion"""
        try:
            logger.info("Loading Marqo Fashion SigLIP model...")
            self.model = AutoModel.from_pretrained('Marqo/marqo-fashionSigLIP', trust_remote_code=True)
            self.processor = AutoProcessor.from_pretrained('Marqo/marqo-fashionSigLIP', trust_remote_code=True)
            self.model.to(self.device)
            self.model.eval()
            
            # Fashion categories for zero-shot classification
            self.fashion_categories = [
                "T-shirt", "shirt", "top", "blouse",
                "trouser", "pants", "jeans", "shorts",
                "dress", "skirt", "gown",
                "coat", "jacket", "hoodie", "sweater",
                "shoes", "sneakers", "boots", "sandals",
                "bag", "handbag", "backpack", "purse",
                "hat", "cap", "beanie"
            ]
            
            self.model_type = "marqo-fashionsiglip"
            logger.info("Marqo Fashion SigLIP model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading Marqo model: {e}")
            raise
    
    def _load_fashion_detection_model(self):
        """Load Fashion Object Detection model"""
        try:
            logger.info("Loading Fashion Object Detection model...")
            self.model = AutoModelForObjectDetection.from_pretrained('yainage90/fashion-object-detection')
            self.processor = AutoImageProcessor.from_pretrained('yainage90/fashion-object-detection')
            self.model.to(self.device)
            self.model.eval()
            
            # Fashion detection labels: ['bag', 'bottom', 'dress', 'hat', 'shoes', 'outer', 'top']
            self.labels = {
                0: "bag", 1: "bottom", 2: "dress", 3: "hat", 
                4: "shoes", 5: "outer", 6: "top"
            }
            
            self.model_type = "fashion-object-detection"
            logger.info("Fashion Object Detection model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading fashion detection model: {e}")
            raise
    
    def _load_custom_fashion_model(self):
        """Load our custom rule-based fashion classifier"""
        try:
            logger.info("Loading custom Fashion-MNIST classifier...")
            self.fashion_classifier = FashionMNISTClassifier()
            self.model_type = "custom-fashion"
            self.labels = {
                0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
                5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"
            }
            logger.info("Custom Fashion classifier loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading custom fashion model: {e}")
            raise
    def _load_fallback_model(self):
        """Load fallback ImageNet model"""
        try:
            logger.info("Loading fallback ImageNet model...")
            model_name = "google/vit-base-patch16-224"
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageClassification.from_pretrained(model_name)
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
            
            self.model_type = "fallback"
            logger.info("Fallback model loaded successfully")
                
        except Exception as e:
            logger.error(f"Error loading fallback model: {e}")
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
            if self.model_type == "marqo-fashionsiglip":
                return self._classify_with_marqo(image, top_k, min_confidence)
            elif self.model_type == "fashion-object-detection":
                return self._classify_with_detection(image, top_k, min_confidence)
            elif self.model_type == "custom-fashion":
                return self.fashion_classifier.classify_clothing_enhanced(
                    image, top_k=top_k, min_confidence=min_confidence
                )
            else:  # fallback
                return self._classify_with_fallback(image, top_k, min_confidence)
                
        except Exception as e:
            logger.error(f"Error in enhanced classification: {e}")
            # Fallback to custom classifier
            if hasattr(self, 'fashion_classifier'):
                return self.fashion_classifier.classify_clothing_enhanced(
                    image, top_k=top_k, min_confidence=min_confidence
                )
            else:
                return [{"label": "Unknown", "confidence": 0.0, "category": "Unknown"}]
    
    def _classify_with_marqo(self, image: Image.Image, top_k: int, min_confidence: float) -> List[Dict[str, any]]:
        """Classify using Marqo Fashion SigLIP (zero-shot)"""
        try:
            # Prepare image and text inputs
            processed = self.processor(
                text=self.fashion_categories, 
                images=[image], 
                padding='max_length', 
                return_tensors="pt"
            )
            
            with torch.no_grad():
                image_features = self.model.get_image_features(
                    processed['pixel_values'].to(self.device), 
                    normalize=True
                )
                text_features = self.model.get_text_features(
                    processed['input_ids'].to(self.device), 
                    normalize=True
                )
                
                # Calculate similarities
                similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                probs = similarities[0].cpu().numpy()
            
            # Get top predictions
            top_indices = np.argsort(probs)[::-1][:top_k]
            predictions = []
            
            for idx in top_indices:
                confidence = float(probs[idx])
                if confidence >= min_confidence:
                    predictions.append({
                        "label": self.fashion_categories[idx],
                        "confidence": confidence,
                        "category": self._get_category(self.fashion_categories[idx])
                    })
            
            return predictions if predictions else [{"label": "Unknown", "confidence": 0.0, "category": "Unknown"}]
            
        except Exception as e:
            logger.error(f"Error in Marqo classification: {e}")
            raise
    
    def _classify_with_detection(self, image: Image.Image, top_k: int, min_confidence: float) -> List[Dict[str, any]]:
        """Classify using Fashion Object Detection"""
        try:
            # Process image
            inputs = self.processor(images=[image], return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model(**inputs.to(self.device))
                target_sizes = torch.tensor([[image.size[1], image.size[0]]])
                results = self.processor.post_process_object_detection(
                    outputs, threshold=min_confidence, target_sizes=target_sizes
                )[0]
            
            # Process results
            predictions = []
            label_counts = {}
            
            for score, label in zip(results["scores"], results["labels"]):
                score = score.item()
                label_id = label.item()
                label_name = self.labels[label_id]
                
                if label_name in label_counts:
                    label_counts[label_name] = max(label_counts[label_name], score)
                else:
                    label_counts[label_name] = score
            
            # Sort by confidence and take top_k
            sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
            
            for label_name, confidence in sorted_labels:
                predictions.append({
                    "label": label_name,
                    "confidence": confidence,
                    "category": self._get_category(label_name)
                })
            
            return predictions if predictions else [{"label": "Unknown", "confidence": 0.0, "category": "Unknown"}]
            
        except Exception as e:
            logger.error(f"Error in detection classification: {e}")
            raise
    
    def _classify_with_fallback(self, image: Image.Image, top_k: int, min_confidence: float) -> List[Dict[str, any]]:
        """Classify using fallback ImageNet model"""
        try:
            # Preprocess image
            pixel_values = self.preprocess_image(image)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(pixel_values)
                probabilities = F.softmax(outputs.logits, dim=-1)
                confidences, predicted_indices = torch.topk(probabilities, top_k)
            
            # Format results
            predictions = []
            for i in range(top_k):
                confidence = float(confidences[0][i])
                if confidence >= min_confidence:
                    label_id = int(predicted_indices[0][i])
                    label = self.labels.get(label_id, f"Class_{label_id}")
                    
                    predictions.append({
                        "label": label,
                        "confidence": confidence,
                        "category": self._get_category(label)
                    })
            
            return predictions if predictions else [{"label": "Unknown", "confidence": 0.0, "category": "Unknown"}]
            
        except Exception as e:
            logger.error(f"Error in fallback classification: {e}")
            raise
    
    def _get_category(self, label: str) -> str:
        """Map label to category"""
        label_lower = label.lower()
        if any(word in label_lower for word in ['shirt', 'top', 'blouse', 't-shirt']):
            return "Tops"
        elif any(word in label_lower for word in ['trouser', 'pants', 'jeans', 'bottom']):
            return "Bottoms"
        elif any(word in label_lower for word in ['dress', 'gown', 'skirt']):
            return "Dresses"
        elif any(word in label_lower for word in ['coat', 'jacket', 'hoodie', 'outer']):
            return "Outerwear"
        elif any(word in label_lower for word in ['shoes', 'sneaker', 'boot', 'sandal']):
            return "Footwear"
        elif any(word in label_lower for word in ['bag', 'handbag', 'backpack', 'purse']):
            return "Accessories"
        elif any(word in label_lower for word in ['hat', 'cap', 'beanie']):
            return "Headwear"
        else:
            return "Other"
    
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
        info = {
            "model_name": self.model_name,
            "model_type": getattr(self, 'model_type', 'unknown'),
            "device": str(self.device),
            "supports_confidence": True,
            "supports_top_k": True
        }
        
        if self.model_type == "marqo-fashionsiglip":
            info.update({
                "num_categories": len(self.fashion_categories),
                "categories": self.fashion_categories,
                "description": "SOTA fashion zero-shot classification model"
            })
        elif self.model_type == "fashion-object-detection":
            info.update({
                "num_categories": len(self.labels),
                "categories": list(self.labels.values()),
                "description": "Specialized fashion object detection model"
            })
        elif self.model_type == "custom-fashion":
            return self.fashion_classifier.get_model_info()
        else:  # fallback
            info.update({
                "num_categories": len(self.labels) if hasattr(self, 'labels') else 0,
                "categories": list(self.labels.values()) if hasattr(self, 'labels') else [],
                "description": "Fallback ImageNet classification model"
            })
        
        return info

# Test function
def test_enhanced_classifier():
    """Test the enhanced classifier with different models"""
    models_to_test = [
        "marqo-fashionsiglip",
        "fashion-object-detection", 
        "custom-fashion",
        "fallback"
    ]
    
    for model_name in models_to_test:
        try:
            print(f"\n🧪 Testing {model_name}...")
            classifier = EnhancedClothingClassifier(model_name=model_name)
            
            # Get model info
            info = classifier.get_model_info()
            print("🤖 Model Info:")
            print(f"   Model: {info['model_name']}")
            print(f"   Type: {info['model_type']}")
            print(f"   Device: {info['device']}")
            print(f"   Categories: {info['num_categories']}")
            if info.get('categories'):
                print(f"   Available categories: {info['categories'][:5]}...")
            
            print(f"✅ {model_name} loaded successfully!")
            
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")
            # Try fallback to custom
            try:
                classifier = EnhancedClothingClassifier(model_name="custom-fashion")
                print(f"✅ Fallback to custom-fashion successful!")
            except Exception as e2:
                print(f"❌ Even fallback failed: {e2}")

if __name__ == "__main__":
    test_enhanced_classifier()
