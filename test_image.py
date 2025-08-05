"""
Image classification test with real images
"""
import requests
import os
from PIL import Image, ImageDraw, ImageFont
import io

def create_test_image():
    """Create a simple test image resembling clothing"""
    # Create a simple t-shirt like image
    img = Image.new('RGB', (224, 224), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw a simple t-shirt shape
    # Body
    draw.rectangle([50, 80, 174, 180], fill='blue', outline='black', width=2)
    # Sleeves
    draw.rectangle([20, 80, 50, 120], fill='blue', outline='black', width=2)
    draw.rectangle([174, 80, 204, 120], fill='blue', outline='black', width=2)
    # Collar
    draw.rectangle([80, 60, 144, 80], fill='blue', outline='black', width=2)
    
    return img

def test_with_image():
    """Test API with a sample image"""
    base_url = "http://127.0.0.1:8001"
    
    print("🖼️  Image Classification Test")
    print("=" * 40)
    
    # Create test image
    test_img = create_test_image()
    
    # Save image to bytes
    img_bytes = io.BytesIO()
    test_img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    # Test basic classification
    print("\n1. Testing basic classification...")
    try:
        files = {'file': ('test_tshirt.png', img_bytes.getvalue(), 'image/png')}
        response = requests.post(f"{base_url}/classify", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Basic classification successful")
            print(f"   Prediction: {data.get('prediction', 'N/A')}")
            print(f"   Confidence: {data.get('confidence', 0):.3f}")
        else:
            print(f"❌ Basic classification failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Basic classification error: {e}")
    
    # Test enhanced classification
    print("\n2. Testing enhanced classification...")
    try:
        img_bytes.seek(0)  # Reset buffer
        files = {'file': ('test_tshirt.png', img_bytes.getvalue(), 'image/png')}
        params = {'top_k': 3, 'min_confidence': 0.1}
        response = requests.post(f"{base_url}/classify-enhanced", files=files, params=params)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Enhanced classification successful")
            print(f"   Top prediction: {data.get('summary', {}).get('top_prediction', 'N/A')}")
            print(f"   Number of predictions: {data.get('summary', {}).get('num_predictions', 0)}")
            
            predictions = data.get('predictions', [])
            for i, pred in enumerate(predictions, 1):
                print(f"   {i}. {pred['label']} - {pred['percentage']:.1f}% (conf: {pred['confidence']:.3f})")
        else:
            print(f"❌ Enhanced classification failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Enhanced classification error: {e}")
    
    print("\n" + "=" * 40)
    print("✨ Image test completed!")

if __name__ == "__main__":
    test_with_image()
