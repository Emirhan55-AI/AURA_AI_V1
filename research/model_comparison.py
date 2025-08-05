"""
Fashion Model Research and Testing
Better models for real-world clothing classification
"""
import sys
sys.path.append('.')

from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FashionModelTester:
    """Test different fashion classification models"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_results = {}
    
    def test_model(self, model_name: str, test_image: Image.Image):
        """Test a specific model with a test image"""
        try:
            logger.info(f"Testing model: {model_name}")
            
            # Load model and processor
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModelForImageClassification.from_pretrained(model_name)
            model.to(self.device)
            model.eval()
            
            # Process image
            inputs = processor(images=test_image, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(pixel_values)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
            
            # Get top 5 predictions
            top_probs, top_indices = torch.topk(probabilities, 5)
            
            # Get labels
            if hasattr(model.config, 'id2label') and model.config.id2label:
                labels = model.config.id2label
            else:
                labels = {i: f"Class_{i}" for i in range(logits.shape[-1])}
            
            # Format results
            predictions = []
            for i in range(5):
                prob = top_probs[0][i].item()
                idx = top_indices[0][i].item()
                label = labels.get(idx, f"Class_{idx}")
                
                predictions.append({
                    "label": label,
                    "confidence": round(prob, 4),
                    "percentage": round(prob * 100, 2)
                })
            
            result = {
                "model_name": model_name,
                "status": "success",
                "num_categories": len(labels),
                "sample_categories": list(labels.values())[:10],
                "predictions": predictions,
                "device": str(self.device)
            }
            
            self.test_results[model_name] = result
            logger.info(f"✅ {model_name} test successful - {len(labels)} categories")
            
            return result
            
        except Exception as e:
            error_result = {
                "model_name": model_name,
                "status": "failed",
                "error": str(e)
            }
            self.test_results[model_name] = error_result
            logger.error(f"❌ {model_name} test failed: {e}")
            return error_result
    
    def compare_models(self, models_to_test: list, test_image: Image.Image):
        """Compare multiple models on the same test image"""
        print("🔍 Fashion Model Comparison")
        print("=" * 60)
        
        for model_name in models_to_test:
            print(f"\n📦 Testing: {model_name}")
            result = self.test_model(model_name, test_image)
            
            if result["status"] == "success":
                print(f"   ✅ Categories: {result['num_categories']}")
                print(f"   🏷️  Sample labels: {result['sample_categories'][:3]}...")
                print(f"   🎯 Top prediction: {result['predictions'][0]['label']} ({result['predictions'][0]['percentage']:.1f}%)")
            else:
                print(f"   ❌ Error: {result['error']}")
        
        return self.test_results
    
    def recommend_best_model(self):
        """Recommend the best model based on test results"""
        successful_models = {name: result for name, result in self.test_results.items() 
                           if result.get("status") == "success"}
        
        if not successful_models:
            return None
        
        # Score models based on number of categories and confidence
        best_model = None
        best_score = 0
        
        for name, result in successful_models.items():
            # Score: more categories = better, higher confidence = better
            num_categories = result.get("num_categories", 0)
            top_confidence = result.get("predictions", [{}])[0].get("confidence", 0)
            
            score = (num_categories * 0.7) + (top_confidence * 30)  # Weight categories more
            
            if score > best_score:
                best_score = score
                best_model = name
        
        return best_model

def create_test_clothing_images():
    """Create test images for different clothing types"""
    from PIL import ImageDraw
    
    images = {}
    
    # T-shirt image
    tshirt = Image.new('RGB', (224, 224), color='white')
    draw = ImageDraw.Draw(tshirt)
    # Body
    draw.rectangle([50, 80, 174, 180], fill='blue', outline='black', width=2)
    # Sleeves  
    draw.rectangle([20, 80, 50, 120], fill='blue', outline='black', width=2)
    draw.rectangle([174, 80, 204, 120], fill='blue', outline='black', width=2)
    # Collar
    draw.rectangle([80, 60, 144, 80], fill='blue', outline='black', width=2)
    images['tshirt'] = tshirt
    
    # Dress image
    dress = Image.new('RGB', (224, 224), color='white')
    draw = ImageDraw.Draw(dress)
    # Upper body
    draw.rectangle([60, 60, 164, 120], fill='red', outline='black', width=2)
    # Skirt part (wider)
    draw.polygon([(60, 120), (164, 120), (180, 200), (44, 200)], fill='red', outline='black')
    # Sleeves
    draw.rectangle([30, 60, 60, 100], fill='red', outline='black', width=2)
    draw.rectangle([164, 60, 194, 100], fill='red', outline='black', width=2)
    images['dress'] = dress
    
    # Trouser image
    trouser = Image.new('RGB', (224, 224), color='white')
    draw = ImageDraw.Draw(trouser)
    # Left leg
    draw.rectangle([70, 80, 110, 200], fill='navy', outline='black', width=2)
    # Right leg
    draw.rectangle([114, 80, 154, 200], fill='navy', outline='black', width=2)
    # Waistband
    draw.rectangle([70, 70, 154, 80], fill='black', outline='black', width=2)
    images['trouser'] = trouser
    
    return images

def main():
    """Main function to test fashion models"""
    
    # Models to test (known fashion classification models)
    models_to_test = [
        "google/vit-base-patch16-224",  # Current model
        "microsoft/resnet-50",          # Alternative
        # Add more fashion-specific models here as we discover them
    ]
    
    # Create test images
    test_images = create_test_clothing_images()
    
    # Initialize tester
    tester = FashionModelTester()
    
    # Test with T-shirt image
    print("🧪 Testing with T-shirt image...")
    tshirt_results = tester.compare_models(models_to_test, test_images['tshirt'])
    
    # Get recommendation
    best_model = tester.recommend_best_model()
    
    print("\n" + "=" * 60)
    print("📊 RESULTS SUMMARY")
    print("=" * 60)
    
    for name, result in tester.test_results.items():
        if result.get("status") == "success":
            print(f"✅ {name}")
            print(f"   Categories: {result['num_categories']}")
            print(f"   Top prediction: {result['predictions'][0]['label']}")
            print(f"   Confidence: {result['predictions'][0]['percentage']:.1f}%")
        else:
            print(f"❌ {name} - {result.get('error', 'Unknown error')}")
    
    if best_model:
        print(f"\n🏆 RECOMMENDED MODEL: {best_model}")
    else:
        print("\n⚠️  No suitable model found")
    
    return tester.test_results

if __name__ == "__main__":
    main()
