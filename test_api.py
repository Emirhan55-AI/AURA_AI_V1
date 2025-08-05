"""
Enhanced API test script
"""
import requests
import json

def test_enhanced_api():
    """Test the enhanced API endpoints"""
    base_url = "http://127.0.0.1:8001"
    
    print("🧪 Enhanced API Test Suite")
    print("=" * 50)
    
    # Test 1: Root endpoint
    print("\n1. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            data = response.json()
            print("✅ Root endpoint working")
            print(f"   Version: {data.get('version', 'N/A')}")
            print(f"   Classifier: {data.get('classifier', {}).get('type', 'N/A')}")
            print(f"   Categories: {data.get('classifier', {}).get('num_categories', 'N/A')}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
    
    # Test 2: Health check
    print("\n2. Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check working")
            print(f"   Status: {data.get('status', 'N/A')}")
            print(f"   Enhanced features: {data.get('enhanced_features', False)}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    # Test 3: Model info
    print("\n3. Testing model info...")
    try:
        response = requests.get(f"{base_url}/model-info")
        if response.status_code == 200:
            data = response.json()
            print("✅ Model info working")
            print(f"   Model: {data.get('model_name', 'N/A')}")
            print(f"   Device: {data.get('device', 'N/A')}")
            print(f"   Categories: {data.get('num_categories', 0)}")
        else:
            print(f"❌ Model info failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Model info error: {e}")
    
    # Test 4: Categories
    print("\n4. Testing categories endpoint...")
    try:
        response = requests.get(f"{base_url}/categories")
        if response.status_code == 200:
            data = response.json()
            print("✅ Categories working")
            print(f"   Total categories: {data.get('total', 0)}")
            categories = data.get('categories', [])
            if categories:
                print(f"   Sample categories: {categories[:3]}...")
        else:
            print(f"❌ Categories failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Categories error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 API tests completed!")
    print("\n📋 Next steps:")
    print("   1. Test with real images using /classify endpoint")
    print("   2. Test enhanced features with /classify-enhanced")
    print("   3. Compare predictions with different confidence thresholds")

if __name__ == "__main__":
    test_enhanced_api()
