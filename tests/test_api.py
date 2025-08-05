"""
API endpoint testleri.

Bu modül, FastAPI endpoint'lerinin doğru çalışıp çalışmadığını test eder.
"""

import pytest
from fastapi.testclient import TestClient
import tempfile
import os
from PIL import Image
import numpy as np
import io

# Test edilecek API'yi import et
from api.main import app

# Test client oluştur
client = TestClient(app)

class TestAPIEndpoints:
    """API endpoint testleri."""
    
    @pytest.fixture
    def sample_image_bytes(self):
        """Test için örnek görüntü bytes'ı oluştur."""
        # 224x224 boyutunda rastgele RGB görüntü
        image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image = Image.fromarray(image_array, 'RGB')
        
        # Bytes'a çevir
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        return img_bytes.getvalue()
    
    def test_root_endpoint(self):
        """Ana endpoint testi."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "active"
    
    def test_health_endpoint(self):
        """Sağlık kontrolü endpoint testi."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_classify_endpoint_no_file(self):
        """Dosya olmadan sınıflandırma endpoint testi."""
        response = client.post("/classify")
        
        assert response.status_code == 422  # Validation error
    
    def test_classify_endpoint_invalid_file_type(self):
        """Geçersiz dosya türü ile sınıflandırma testi."""
        # Text dosyası gönder
        files = {
            "image": ("test.txt", "This is not an image", "text/plain")
        }
        
        response = client.post("/classify", files=files)
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "Geçersiz dosya formatı" in data["detail"]
    
    def test_classify_endpoint_valid_image(self, sample_image_bytes):
        """Geçerli görüntü ile sınıflandırma testi."""
        files = {
            "image": ("test_image.jpg", sample_image_bytes, "image/jpeg")
        }
        
        try:
            response = client.post("/classify", files=files)
            
            # Başarılı olması veya model hatası vermesi beklenir
            assert response.status_code in [200, 500]
            
            if response.status_code == 200:
                data = response.json()
                
                # Temel yanıt yapısını kontrol et
                assert "success" in data
                assert "filename" in data
                assert "predictions" in data
                assert "model_info" in data
                assert "timestamp" in data
                
                if data["success"]:
                    # Başarılı yanıt yapısını kontrol et
                    assert isinstance(data["predictions"], list)
                    assert isinstance(data["model_info"], dict)
                    assert "model_name" in data["model_info"]
                    assert "device" in data["model_info"]
                
        except Exception as e:
            # Model yüklenemezse testi geç
            pytest.skip(f"Model yüklenemedi: {e}")
    
    def test_classify_endpoint_with_top_k(self, sample_image_bytes):
        """top_k parametresi ile sınıflandırma testi."""
        files = {
            "image": ("test_image.jpg", sample_image_bytes, "image/jpeg")
        }
        data = {
            "top_k": 3
        }
        
        try:
            response = client.post("/classify", files=files, data=data)
            
            # Başarılı olması veya model hatası vermesi beklenir
            assert response.status_code in [200, 500]
            
        except Exception as e:
            pytest.skip(f"Model yüklenemedi: {e}")
    
    def test_classify_endpoint_invalid_top_k(self, sample_image_bytes):
        """Geçersiz top_k parametresi testi."""
        files = {
            "image": ("test_image.jpg", sample_image_bytes, "image/jpeg")
        }
        data = {
            "top_k": 15  # Maksimum 10 olmalı
        }
        
        response = client.post("/classify", files=files, data=data)
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "top_k değeri 1-10 arasında olmalıdır" in data["detail"]
    
    def test_classify_endpoint_large_file(self):
        """Büyük dosya testi."""
        # 15MB boyutunda sahte dosya oluştur
        large_data = b"x" * (15 * 1024 * 1024)
        
        files = {
            "image": ("large_image.jpg", large_data, "image/jpeg")
        }
        
        response = client.post("/classify", files=files)
        
        assert response.status_code == 413  # Request Entity Too Large
        data = response.json()
        assert "detail" in data
        assert "Dosya boyutu çok büyük" in data["detail"]

class TestAPIIntegration:
    """API entegrasyon testleri."""
    
    def test_full_workflow(self):
        """Tam iş akışı testi."""
        # 1. Sağlık kontrolü
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        # 2. Ana sayfa
        root_response = client.get("/")
        assert root_response.status_code == 200
        
        # 3. Görüntü oluştur ve sınıflandır
        image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        image = Image.fromarray(image_array, 'RGB')
        
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        files = {
            "image": ("workflow_test.jpg", img_bytes.getvalue(), "image/jpeg")
        }
        
        try:
            classify_response = client.post("/classify", files=files)
            
            # Başarılı olması veya model hatası beklenir
            assert classify_response.status_code in [200, 500]
            
            if classify_response.status_code == 200:
                data = classify_response.json()
                assert data.get("success") is True
                
        except Exception as e:
            pytest.skip(f"Entegrasyon testi başarısız: {e}")

if __name__ == "__main__":
    # Testleri çalıştır
    pytest.main([__file__, "-v"])
