"""
Model Accuracy Testing for Enhanced Fashion Classifier
Tests multiple models with known test images and measures accuracy
"""
import os
import sys
from PIL import Image
from typing import List, Dict, Tuple
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.enhanced_classifier import EnhancedClothingClassifier

class FashionModelAccuracyTester:
    def __init__(self):
        """Initialize the accuracy tester"""
        self.test_data_dir = "test_data"
        self.test_cases = [
            ("blue_tshirt.jpg", "T-shirt/top"),
            ("navy_trouser.jpg", "Trouser"), 
            ("black_coat.jpg", "Coat"),
            ("red_dress.jpg", "Dress"),
            ("brown_bag.jpg", "Bag"),
            ("white_sneaker.jpg", "Sneaker")
        ]
        
    def test_model_accuracy(self, model_name: str, min_confidence: float = 0.1) -> Dict[str, any]:
        """
        Test accuracy of a specific model
        
        Args:
            model_name: Name of the model to test
            min_confidence: Minimum confidence threshold
            
        Returns:
            Dictionary with test results
        """
        print(f"\n🧪 Testing {model_name} Model")
        print("=" * 50)
        
        try:
            # Initialize classifier
            classifier = EnhancedClothingClassifier(model_name=model_name)
            
            # Get model info
            model_info = classifier.get_model_info()
            print(f"Model: {model_info['model_name']}")
            print(f"Type: {model_info.get('model_type', 'unknown')}")
            print(f"Categories: {model_info['num_categories']}")
            
            # Test each image
            results = []
            correct_predictions = 0
            total_tests = len(self.test_cases)
            
            for image_file, expected_label in self.test_cases:
                image_path = os.path.join(self.test_data_dir, image_file)
                
                if not os.path.exists(image_path):
                    print(f"❌ Image not found: {image_path}")
                    continue
                    
                # Load image
                image = Image.open(image_path)
                
                # Get predictions
                predictions = classifier.classify_clothing_enhanced(
                    image, top_k=3, min_confidence=min_confidence
                )
                
                # Check if prediction is correct
                if predictions:
                    predicted_label = predictions[0]['label']
                    confidence = predictions[0]['confidence']
                    is_correct = self._is_label_match(predicted_label, expected_label)
                    
                    if is_correct:
                        correct_predictions += 1
                        status = "✅ PASS"
                    else:
                        status = "❌ FAIL"
                    
                    print(f"{status} {image_file}: {predicted_label} ({confidence:.3f}) - Expected: {expected_label}")
                    
                    results.append({
                        "image": image_file,
                        "expected": expected_label,
                        "predicted": predicted_label,
                        "confidence": confidence,
                        "correct": is_correct,
                        "all_predictions": predictions
                    })
                else:
                    print(f"❌ FAIL {image_file}: No predictions - Expected: {expected_label}")
                    results.append({
                        "image": image_file,
                        "expected": expected_label,
                        "predicted": "No prediction",
                        "confidence": 0.0,
                        "correct": False,
                        "all_predictions": []
                    })
            
            # Calculate accuracy
            accuracy = (correct_predictions / total_tests) * 100 if total_tests > 0 else 0
            
            print(f"\n📊 Results Summary:")
            print(f"Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_tests})")
            
            return {
                "model_name": model_name,
                "model_info": model_info,
                "accuracy": accuracy,
                "correct_predictions": correct_predictions,
                "total_tests": total_tests,
                "detailed_results": results
            }
            
        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")
            return {
                "model_name": model_name,
                "error": str(e),
                "accuracy": 0.0,
                "correct_predictions": 0,
                "total_tests": len(self.test_cases)
            }
    
    def _is_label_match(self, predicted: str, expected: str) -> bool:
        """Check if predicted label matches expected (with some flexibility)"""
        predicted_lower = predicted.lower()
        expected_lower = expected.lower()
        
        # Direct match
        if predicted_lower == expected_lower:
            return True
            
        # Flexible matching for similar terms
        synonyms = {
            "t-shirt/top": ["t-shirt", "top", "shirt", "tshirt", "t shirt"],
            "trouser": ["trouser", "pants", "jeans", "bottom", "trousers"],
            "dress": ["dress", "gown", "skirt"],
            "coat": ["coat", "jacket", "outer", "outerwear"],
            "bag": ["bag", "handbag", "purse", "backpack"],
            "sneaker": ["sneaker", "shoes", "shoe", "sneakers", "footwear"]
        }
        
        for standard_label, label_list in synonyms.items():
            if expected_lower in label_list or expected_lower == standard_label:
                if predicted_lower in label_list or predicted_lower == standard_label:
                    return True
        
        return False
    
    def test_all_models(self) -> Dict[str, any]:
        """Test all available models and compare results"""
        models_to_test = [
            "marqo-fashionsiglip",
            "fashion-object-detection", 
            "custom-fashion",
            "fallback"
        ]
        
        all_results = {}
        summary = []
        
        print("🚀 Fashion Classification Model Comparison")
        print("=" * 60)
        
        for model_name in models_to_test:
            result = self.test_model_accuracy(model_name)
            all_results[model_name] = result
            
            if 'error' not in result:
                summary.append((model_name, result['accuracy'], result['correct_predictions']))
        
        # Print comparison summary
        print(f"\n🏆 Model Accuracy Comparison:")
        print("=" * 60)
        summary.sort(key=lambda x: x[1], reverse=True)  # Sort by accuracy
        
        for i, (model, accuracy, correct) in enumerate(summary, 1):
            print(f"{i}. {model}: {accuracy:.1f}% ({correct}/{len(self.test_cases)})")
        
        # Save detailed results
        results_file = "model_accuracy_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n📄 Detailed results saved to: {results_file}")
        
        return all_results

def main():
    """Main function to run accuracy tests"""
    tester = FashionModelAccuracyTester()
    
    # Test all models
    results = tester.test_all_models()
    
    return results

if __name__ == "__main__":
    main()
