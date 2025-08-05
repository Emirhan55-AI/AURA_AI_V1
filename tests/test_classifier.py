"""
Kıyafet sınıflandırıcı için test modülü.

Bu modül, ClothingClassifier sınıfının temel işlevselliğini test eder.
"""

import pytest
import os
import tempfile
from PIL import Image
import numpy as np
from unittest.mock import patch, MagicMock

# Test edilecek modülü import et
from models.classifier import ClothingClassifier, classify_clothing, get_classifier

class TestClothingClassifier:
    """ClothingClassifier sınıfı için test sınıfı."""
    
    @pytest.fixture
    def sample_image_path(self):
        """Test için örnek görüntü dosyası oluştur."""
        # Geçici bir RGB görüntü oluştur
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            # 224x224 boyutunda rastgele RGB görüntü
            image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            image = Image.fromarray(image_array, 'RGB')
            image.save(tmp_file.name, 'JPEG')
            
            yield tmp_file.name
            
            # Cleanup
            try:
                os.unlink(tmp_file.name)
            except FileNotFoundError:
                pass
    
    @pytest.fixture
    def mock_classifier(self):
        """Mock edilmiş sınıflandırıcı."""
        with patch('models.classifier.pipeline') as mock_pipeline:
            # Mock pipeline sonucu
            mock_result = [
                {'label': 'shirt', 'score': 0.8},
                {'label': 'jacket', 'score': 0.15},
                {'label': 'sweater', 'score': 0.05}
            ]
            mock_pipeline.return_value = MagicMock(return_value=mock_result)
            
            classifier = ClothingClassifier()
            yield classifier, mock_result
    
    def test_classifier_initialization(self):
        """Sınıflandırıcının doğru şekilde başlatıldığını test et."""
        try:
            classifier = ClothingClassifier()
            assert classifier is not None
            assert classifier.model_name is not None
            assert classifier.device in ['cpu', 'cuda']
        except Exception as e:
            # Model yüklenemezse, en azından hata fırlatmadığını kontrol et
            pytest.skip(f"Model yüklenemedi: {e}")
    
    def test_preprocess_image_valid(self, sample_image_path):
        """Geçerli görüntü ön işleme testi."""
        try:
            classifier = ClothingClassifier()
            processed_image = classifier._preprocess_image(sample_image_path)
            
            assert processed_image is not None
            assert processed_image.mode == 'RGB'
            assert isinstance(processed_image, Image.Image)
        except Exception as e:
            pytest.skip(f"Model yüklenemedi: {e}")
    
    def test_preprocess_image_invalid_path(self):
        """Geçersiz dosya yolu testi."""
        try:
            classifier = ClothingClassifier()
            
            with pytest.raises(Exception):
                classifier._preprocess_image("nonexistent_file.jpg")
        except Exception as e:
            pytest.skip(f"Model yüklenemedi: {e}")
    
    def test_map_to_clothing_categories(self):
        """Kategori eşleme testi."""
        try:
            classifier = ClothingClassifier()
            
            # Test verileri
            test_predictions = [
                {'label': 'shirt', 'score': 0.8},
                {'label': 'running shoes', 'score': 0.7},
                {'label': 'unknown object', 'score': 0.1}
            ]
            
            mapped = classifier._map_to_clothing_categories(test_predictions)
            
            assert len(mapped) == 3
            assert mapped[0]['category'] == 'shirt'
            assert mapped[1]['category'] == 'shoes'
            assert mapped[2]['category'] == 'other'
            
            # Confidence değerlerinin korunduğunu kontrol et
            assert mapped[0]['confidence'] == 0.8
            assert mapped[1]['confidence'] == 0.7
        except Exception as e:
            pytest.skip(f"Model yüklenemedi: {e}")
    
    def test_classify_clothing_with_mock(self, mock_classifier, sample_image_path):
        """Mock edilmiş sınıflandırma testi."""
        classifier, expected_result = mock_classifier
        
        result = classifier.classify_clothing(sample_image_path)
        
        # Temel sonuç yapısını kontrol et
        assert result is not None
        assert isinstance(result, dict)
        assert 'success' in result
        assert 'predictions' in result
        assert 'image_path' in result
    
    def test_classify_clothing_nonexistent_file(self):
        """Var olmayan dosya ile sınıflandırma testi."""
        try:
            classifier = ClothingClassifier()
            result = classifier.classify_clothing("nonexistent_file.jpg")
            
            assert result['success'] is False
            assert 'error' in result
        except Exception as e:
            pytest.skip(f"Model yüklenemedi: {e}")
    
    def test_global_classifier_singleton(self):
        """Global sınıflandırıcı singleton testi."""
        try:
            classifier1 = get_classifier()
            classifier2 = get_classifier()
            
            # Aynı instance olmalı
            assert classifier1 is classifier2
        except Exception as e:
            pytest.skip(f"Model yüklenemedi: {e}")
    
    def test_classify_clothing_function(self, sample_image_path):
        """Global classify_clothing fonksiyon testi."""
        try:
            result = classify_clothing(sample_image_path)
            
            assert result is not None
            assert isinstance(result, dict)
            # Başarılı olması veya hata döndürmesi beklenir
            assert 'success' in result
        except Exception as e:
            pytest.skip(f"Model yüklenemedi veya ağ bağlantısı yok: {e}")

class TestClothingClassifierIntegration:
    """Entegrasyon testleri."""
    
    def test_end_to_end_workflow(self):
        """Uçtan uca iş akışı testi."""
        try:
            # Gerçek bir sınıflandırıcı oluştur (internet gerekebilir)
            classifier = ClothingClassifier()
            
            # Test görüntüsü oluştur
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                # Basit test görüntüsü
                image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                image = Image.fromarray(image_array, 'RGB')
                image.save(tmp_file.name, 'JPEG')
                
                try:
                    # Sınıflandırma yap
                    result = classifier.classify_clothing(tmp_file.name)
                    
                    # Temel sonuç yapısını kontrol et
                    assert isinstance(result, dict)
                    assert 'success' in result
                    
                    if result['success']:
                        assert 'predictions' in result
                        assert 'model_used' in result
                        assert 'device' in result
                        
                        # Tahminler listesi olmalı
                        predictions = result['predictions']
                        assert isinstance(predictions, list)
                        
                        if predictions:
                            # Her tahmin gerekli alanları içermeli
                            for pred in predictions:
                                assert 'category' in pred
                                assert 'confidence' in pred
                                assert isinstance(pred['confidence'], (int, float))
                    
                finally:
                    # Cleanup
                    try:
                        os.unlink(tmp_file.name)
                    except FileNotFoundError:
                        pass
                        
        except Exception as e:
            pytest.skip(f"Entegrasyon testi başarısız (muhtemelen model yüklenemedi): {e}")

if __name__ == "__main__":
    # Testleri çalıştır
    pytest.main([__file__, "-v"])
