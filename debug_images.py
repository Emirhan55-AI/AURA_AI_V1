"""
Debug test images to understand classification issues
"""
import sys
sys.path.append('.')

from tests.accuracy_test import FashionAccuracyTester
from PIL import Image
import numpy as np

def debug_test_images():
    """Debug test images to understand why classification fails"""
    
    tester = FashionAccuracyTester()
    test_dataset = tester.create_test_dataset()
    
    print("🔍 DEBUG: Test Image Analysis")
    print("=" * 50)
    
    for test_name, (image, expected_label) in test_dataset.items():
        print(f"\n📋 Analyzing: {test_name}")
        print(f"   Expected: {expected_label}")
        
        # Convert to numpy for analysis
        img_array = np.array(image.convert('RGB'))
        height, width = img_array.shape[:2]
        
        # Color analysis
        avg_color = np.mean(img_array, axis=(0, 1))
        r, g, b = avg_color
        brightness = np.mean(avg_color)
        
        print(f"   📊 Color Analysis:")
        print(f"      RGB: ({r:.1f}, {g:.1f}, {b:.1f})")
        print(f"      Brightness: {brightness:.1f}")
        
        # Find non-white pixels
        white_threshold = 200
        non_white_mask = np.any(img_array < white_threshold, axis=2)
        clothing_pixels = np.sum(non_white_mask)
        
        if clothing_pixels > 0:
            coords = np.where(non_white_mask)
            min_y, max_y = np.min(coords[0]), np.max(coords[0])
            min_x, max_x = np.min(coords[1]), np.max(coords[1])
            
            clothing_height = max_y - min_y
            clothing_width = max_x - min_x
            aspect_ratio = clothing_height / max(clothing_width, 1)
            
            center_y = (min_y + max_y) / 2
            is_bottom_heavy = center_y > height * 0.6
            
            width_ratio = clothing_width / width
            is_wide = width_ratio > 0.5
            
            print(f"   📏 Shape Analysis:")
            print(f"      Clothing pixels: {clothing_pixels}")
            print(f"      Dimensions: {clothing_width} x {clothing_height}")
            print(f"      Aspect ratio: {aspect_ratio:.2f}")
            print(f"      Bottom heavy: {is_bottom_heavy}")
            print(f"      Wide: {is_wide}")
            print(f"      Width ratio: {width_ratio:.2f}")
        
        # Test classification
        predictions = tester.classifier.classify_clothing_enhanced(image, top_k=3)
        print(f"   🎯 Predictions:")
        for i, pred in enumerate(predictions, 1):
            print(f"      {i}. {pred['label']} - {pred['percentage']:.1f}%")
        
        # Save image for manual inspection
        image.save(f"debug_{test_name}.png")
        print(f"   💾 Saved: debug_{test_name}.png")

if __name__ == "__main__":
    debug_test_images()
