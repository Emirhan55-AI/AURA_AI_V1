"""
Simple test script for enhanced classifier
"""
import sys
import os

# Add project root to path
sys.path.append('.')

def test_basic_imports():
    """Test basic imports"""
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        from transformers import AutoImageProcessor
        print("✅ Transformers imported")
        
        from PIL import Image
        print("✅ PIL imported")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_classifier():
    """Test basic classifier"""
    try:
        from models.classifier import ClothingClassifier
        classifier = ClothingClassifier()
        print("✅ Basic classifier loaded")
        return True
    except Exception as e:
        print(f"❌ Basic classifier error: {e}")
        return False

def test_enhanced_classifier():
    """Test enhanced classifier"""
    try:
        from models.enhanced_classifier import EnhancedClothingClassifier
        classifier = EnhancedClothingClassifier()
        print("✅ Enhanced classifier loaded")
        
        # Get model info
        info = classifier.get_model_info()
        print(f"   Model: {info['model_name']}")
        print(f"   Categories: {info['num_categories']}")
        
        return True
    except Exception as e:
        print(f"❌ Enhanced classifier error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 AURA AI Classifier Test")
    print("=" * 40)
    
    print("\n1. Testing imports...")
    imports_ok = test_basic_imports()
    
    print("\n2. Testing basic classifier...")
    basic_ok = test_basic_classifier()
    
    print("\n3. Testing enhanced classifier...")
    enhanced_ok = test_enhanced_classifier()
    
    print("\n" + "=" * 40)
    print("📊 Test Results:")
    print(f"   Imports: {'✅' if imports_ok else '❌'}")
    print(f"   Basic Classifier: {'✅' if basic_ok else '❌'}")
    print(f"   Enhanced Classifier: {'✅' if enhanced_ok else '❌'}")
    
    if basic_ok:
        print("\n🎉 System is ready for basic classification!")
    if enhanced_ok:
        print("🚀 Enhanced features are available!")
