"""
Comprehensive Accuracy Testing for Fashion Classification
Test model accuracy with known clothing images and expected results
"""
import sys
sys.path.append('.')

from models.enhanced_classifier import EnhancedClothingClassifier
from PIL import Image, ImageDraw
import requests
import io
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FashionAccuracyTester:
    """Comprehensive accuracy testing for fashion classification"""
    
    def __init__(self):
        self.classifier = EnhancedClothingClassifier()
        self.test_results = {}
        self.accuracy_threshold = 0.5  # Minimum confidence for correct classification
        
    def create_test_dataset(self) -> Dict[str, Tuple[Image.Image, str]]:
        """Create a test dataset with known labels"""
        test_images = {}
        
        # Test 1: Blue T-shirt
        tshirt = Image.new('RGB', (224, 224), color='white')
        draw = ImageDraw.Draw(tshirt)
        # Body
        draw.rectangle([50, 80, 174, 180], fill='blue', outline='black', width=2)
        # Sleeves
        draw.rectangle([20, 80, 50, 120], fill='blue', outline='black', width=2)
        draw.rectangle([174, 80, 204, 120], fill='blue', outline='black', width=2)
        # Collar
        draw.rectangle([80, 60, 144, 80], fill='blue', outline='black', width=2)
        test_images['blue_tshirt'] = (tshirt, 'T-shirt/top')
        
        # Test 2: Red Dress
        dress = Image.new('RGB', (224, 224), color='white')
        draw = ImageDraw.Draw(dress)
        # Upper body
        draw.rectangle([60, 60, 164, 120], fill='red', outline='black', width=2)
        # Skirt part (wider)
        draw.polygon([(60, 120), (164, 120), (180, 200), (44, 200)], fill='red', outline='black')
        # Sleeves
        draw.rectangle([30, 60, 60, 100], fill='red', outline='black', width=2)
        draw.rectangle([164, 60, 194, 100], fill='red', outline='black', width=2)
        test_images['red_dress'] = (dress, 'Dress')
        
        # Test 3: Navy Trousers
        trouser = Image.new('RGB', (224, 224), color='white')
        draw = ImageDraw.Draw(trouser)
        # Left leg
        draw.rectangle([70, 80, 110, 200], fill='navy', outline='black', width=2)
        # Right leg
        draw.rectangle([114, 80, 154, 200], fill='navy', outline='black', width=2)
        # Waistband
        draw.rectangle([70, 70, 154, 80], fill='black', outline='black', width=2)
        test_images['navy_trouser'] = (trouser, 'Trouser')
        
        # Test 4: Black Coat
        coat = Image.new('RGB', (224, 224), color='white')
        draw = ImageDraw.Draw(coat)
        # Main body (longer than shirt)
        draw.rectangle([40, 70, 184, 210], fill='black', outline='gray', width=3)
        # Sleeves
        draw.rectangle([10, 70, 40, 140], fill='black', outline='gray', width=2)
        draw.rectangle([184, 70, 214, 140], fill='black', outline='gray', width=2)
        # Collar
        draw.rectangle([70, 50, 154, 70], fill='black', outline='gray', width=2)
        # Buttons
        for y in range(90, 180, 20):
            draw.ellipse([108, y, 116, y+8], fill='white', outline='black')
        test_images['black_coat'] = (coat, 'Coat')
        
        # Test 5: Brown Bag
        bag = Image.new('RGB', (224, 224), color='white')
        draw = ImageDraw.Draw(bag)
        # Main bag body
        draw.rectangle([60, 100, 164, 180], fill='brown', outline='black', width=3)
        # Handle
        draw.arc([80, 70, 144, 110], 0, 180, fill='brown', width=5)
        # Closure
        draw.rectangle([95, 90, 129, 100], fill='gold', outline='black', width=1)
        test_images['brown_bag'] = (bag, 'Bag')
        
        return test_images
    
    def test_single_image(self, image: Image.Image, expected_label: str, test_name: str) -> Dict:
        """Test classification accuracy for a single image"""
        try:
            # Get predictions
            predictions = self.classifier.classify_clothing_enhanced(
                image, top_k=5, min_confidence=0.01
            )
            
            # Find if expected label is in predictions
            found_expected = False
            expected_rank = None
            expected_confidence = 0
            
            for i, pred in enumerate(predictions):
                if expected_label.lower() in pred['label'].lower():
                    found_expected = True
                    expected_rank = i + 1
                    expected_confidence = pred['confidence']
                    break
            
            # Determine if test passed
            test_passed = (
                found_expected and 
                expected_rank == 1 and 
                expected_confidence >= self.accuracy_threshold
            )
            
            result = {
                'test_name': test_name,
                'expected_label': expected_label,
                'predictions': predictions,
                'found_expected': found_expected,
                'expected_rank': expected_rank,
                'expected_confidence': expected_confidence,
                'test_passed': test_passed,
                'top_prediction': predictions[0]['label'] if predictions else 'None',
                'top_confidence': predictions[0]['confidence'] if predictions else 0
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error testing {test_name}: {e}")
            return {
                'test_name': test_name,
                'expected_label': expected_label,
                'error': str(e),
                'test_passed': False
            }
    
    def run_accuracy_tests(self) -> Dict:
        """Run comprehensive accuracy tests"""
        print("🧪 Fashion Classification Accuracy Tests")
        print("=" * 60)
        
        # Get test dataset
        test_dataset = self.create_test_dataset()
        
        # Run tests
        results = {}
        passed_tests = 0
        total_tests = len(test_dataset)
        
        for test_name, (image, expected_label) in test_dataset.items():
            print(f"\n📋 Testing: {test_name}")
            print(f"   Expected: {expected_label}")
            
            result = self.test_single_image(image, expected_label, test_name)
            results[test_name] = result
            
            if 'error' in result:
                print(f"   ❌ Error: {result['error']}")
                continue
            
            # Display results
            print(f"   Top prediction: {result['top_prediction']} ({result['top_confidence']:.3f})")
            
            if result['found_expected']:
                print(f"   Expected found at rank #{result['expected_rank']} ({result['expected_confidence']:.3f})")
            else:
                print(f"   ⚠️  Expected label '{expected_label}' not found in top 5")
            
            if result['test_passed']:
                print(f"   ✅ PASSED")
                passed_tests += 1
            else:
                print(f"   ❌ FAILED")
                if result['found_expected'] and result['expected_rank'] != 1:
                    print(f"      Reason: Expected at rank #{result['expected_rank']}, not #1")
                elif result['found_expected'] and result['expected_confidence'] < self.accuracy_threshold:
                    print(f"      Reason: Confidence {result['expected_confidence']:.3f} < {self.accuracy_threshold}")
                else:
                    print(f"      Reason: Expected label not found")
        
        # Calculate overall accuracy
        accuracy = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print("\n" + "=" * 60)
        print("📊 ACCURACY TEST RESULTS")
        print("=" * 60)
        print(f"Tests Passed: {passed_tests}/{total_tests}")
        print(f"Overall Accuracy: {accuracy:.1f}%")
        print(f"Confidence Threshold: {self.accuracy_threshold}")
        
        # Detailed breakdown
        print("\n📋 Detailed Results:")
        for test_name, result in results.items():
            if 'error' not in result:
                status = "✅ PASS" if result['test_passed'] else "❌ FAIL"
                print(f"   {status} {test_name}: {result['top_prediction']} ({result['top_confidence']:.3f})")
        
        # Recommendations
        print("\n💡 Recommendations:")
        if accuracy < 50:
            print("   ⚠️  Model accuracy is low. Consider:")
            print("      - Using a fashion-specific model")
            print("      - Implementing custom preprocessing")
            print("      - Training on fashion dataset")
        elif accuracy < 80:
            print("   📈 Model shows promise but needs improvement:")
            print("      - Fine-tune confidence thresholds")
            print("      - Add data augmentation")
            print("      - Improve preprocessing")
        else:
            print("   ✅ Model performance is good!")
            print("      - Consider adding more test cases")
            print("      - Test with real-world images")
        
        # Save results
        self.test_results = {
            'accuracy': accuracy,
            'passed_tests': passed_tests,
            'total_tests': total_tests,
            'threshold': self.accuracy_threshold,
            'detailed_results': results
        }
        
        return self.test_results
    
    def save_test_report(self, filename: str = "accuracy_test_report.json"):
        """Save test results to a JSON file"""
        import json
        
        if not self.test_results:
            print("No test results to save. Run tests first.")
            return
        
        # Convert results to JSON-serializable format
        serializable_results = {}
        for key, value in self.test_results.items():
            if key == 'detailed_results':
                serializable_results[key] = {}
                for test_name, test_result in value.items():
                    # Remove non-serializable items
                    clean_result = {k: v for k, v in test_result.items() 
                                  if k != 'predictions' or isinstance(v, (str, int, float, bool, list))}
                    serializable_results[key][test_name] = clean_result
            else:
                serializable_results[key] = value
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            print(f"📄 Test report saved to: {filename}")
        except Exception as e:
            print(f"❌ Error saving report: {e}")

def main():
    """Main function to run accuracy tests"""
    tester = FashionAccuracyTester()
    results = tester.run_accuracy_tests()
    tester.save_test_report()
    
    return results

if __name__ == "__main__":
    main()
