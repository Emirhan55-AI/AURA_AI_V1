# Aura AI MVP - Kıyafet Sınıflandırma Sistemi

Bu proje, kullanıcıların gönderdiği kıyafet fotoğraflarını analiz ederek türlerini belirleyen bir AI sistemidir. FastAPI tabanlı REST API sunucu olarak geliştirilmiştir.

## 🎯 Özellikler

- **Kıyafet Sınıflandırma**: Fotoğraftaki kıyafetleri otomatik olarak tespit ve sınıflandırma
- **RESTful API**: Mobil ve web uygulamalar için kolay entegrasyon
- **AI Model Desteği**: Hugging Face transformers ile güçlendirilmiş
- **Docker Desteği**: Kolay deployment ve scalability
- **Test Coverage**: Kapsamlı test suite

## 🚀 Hızlı Başlangıç

### Gereksinimler

- Python 3.10+
- pip
- (Opsiyonel) Docker

### Kurulum

1. **Depoyu klonlayın:**
   ```bash
   git clone <repo-url>
   cd aura_ai_mvp
   ```

2. **Sanal ortam oluşturun:**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Bağımlılıkları yükleyin:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Sunucuyu başlatın:**
   ```bash
   python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```

5. **API dokümantasyonuna erişin:**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Docker ile Çalıştırma

1. **Docker image'ını oluşturun:**
   ```bash
   docker build -t aura-ai-mvp .
   ```

2. **Container'ı çalıştırın:**
   ```bash
   docker run -p 8000:8000 aura-ai-mvp
   ```

## 📡 API Kullanımı

### Endpoint'ler

#### `GET /` - Ana Sayfa
API durumunu ve temel bilgileri döndürür.

#### `GET /health` - Sağlık Kontrolü
Servisin sağlık durumunu kontrol eder.

#### `POST /classify` - Kıyafet Sınıflandırma
Yüklenen fotoğraftaki kıyafetleri sınıflandırır.

**Parametreler:**
- `image` (file): Sınıflandırılacak kıyafet fotoğrafı
- `top_k` (int, opsiyonel): Döndürülecek en yüksek tahmin sayısı (1-10, varsayılan: 5)

**Örnek Kullanım (curl):**
```bash
curl -X POST "http://localhost:8000/classify" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "image=@shirt.jpg" \
     -F "top_k=3"
```

**Örnek Yanıt:**
```json
{
  "success": true,
  "filename": "shirt.jpg",
  "predictions": [
    {
      "category": "shirt",
      "original_label": "polo shirt",
      "confidence": 0.892
    },
    {
      "category": "jacket", 
      "original_label": "cardigan",
      "confidence": 0.076
    }
  ],
  "model_info": {
    "model_name": "google/vit-base-patch16-224",
    "device": "cpu"
  },
  "timestamp": "2025-01-01T12:00:00.000Z"
}
```

## 🧪 Test Etme

### Unit Testler
```bash
# Tüm testleri çalıştır
pytest

# Verbose çıktı ile
pytest -v

# Belirli bir test dosyası
pytest tests/test_classifier.py -v

# Coverage raporu ile
pytest --cov=models --cov=api tests/
```

### Manuel Test
```bash
# API'yi test etmek için
python -m pytest tests/test_api.py -v

# Model testleri için
python -m pytest tests/test_classifier.py -v
```

## 🏗️ Proje Yapısı

```
aura_ai_mvp/
├── api/                    # FastAPI endpoint'leri
│   ├── __init__.py
│   └── main.py            # Ana API sunucusu
├── models/                # AI modelleri
│   ├── __init__.py
│   └── classifier.py      # Kıyafet sınıflandırıcı
├── utils/                 # Yardımcı fonksiyonlar
│   ├── __init__.py
│   └── helpers.py         # Genel yardımcı fonksiyonlar
├── tests/                 # Test dosyaları
│   ├── __init__.py
│   ├── test_api.py        # API testleri
│   └── test_classifier.py # Model testleri
├── test_data/            # Test görüntüleri
├── temp_images/          # Geçici dosyalar
├── requirements.txt      # Python bağımlılıkları
├── Dockerfile           # Docker konfigürasyonu
├── .gitignore          # Git ignore kuralları
└── README.md           # Bu dosya
```

## 🔧 Konfigürasyon

### Ortam Değişkenleri

Sistem aşağıdaki ortam değişkenlerini destekler:

- `MODEL_NAME`: Kullanılacak Hugging Face model adı (varsayılan: google/vit-base-patch16-224)
- `MAX_FILE_SIZE`: Maksimum dosya boyutu (varsayılan: 10MB)
- `LOG_LEVEL`: Log seviyesi (varsayılan: INFO)

### Model Seçenekleri

Sistem şu modelleri destekler:
- `google/vit-base-patch16-224` (varsayılan)
- Diğer Vision Transformer modelleri
- Fashion-MNIST eğitilmiş özel modeller

## 🚧 Geliştirme

### Yeni Özellik Ekleme

1. Özelliği geliştirin
2. Testler yazın
3. Dokümantasyonu güncelleyin
4. Pull request oluşturun

### Kod Standartları

- PEP 8 Python stil rehberine uyun
- Type hints kullanın
- Docstring'leri ekleyin
- Test coverage %80'in üzerinde tutun

## 🐛 Sorun Giderme

### Yaygın Problemler

1. **Model yüklenemedi hatası:**
   - İnternet bağlantınızı kontrol edin
   - Hugging Face modeli mevcut mu kontrol edin
   - Disk alanınız yeterli mi kontrol edin

2. **Dosya yükleme hatası:**
   - Dosya boyutunu kontrol edin (max 10MB)
   - Dosya formatını kontrol edin (JPG, PNG, etc.)

3. **Docker build hatası:**
   - Docker daemon'un çalıştığından emin olun
   - Yeterli disk alanı olduğundan emin olun

### Log Kontrolü

```bash
# API loglarını görüntüle
python -m uvicorn api.main:app --log-level debug

# Test loglarını görüntüle
pytest -v -s --log-cli-level=DEBUG
```

## 📈 Performans

### Benchmarks

- **Model yükleme süresi**: ~5-10 saniye (ilk kez)
- **Sınıflandırma süresi**: ~1-3 saniye (CPU)
- **API yanıt süresi**: ~2-4 saniye (model + overhead)

### Optimizasyon İpuçları

1. GPU kullanın (CUDA destekli)
2. Model cache'ini etkinleştirin
3. Batch processing kullanın
4. Image preprocessing'i optimize edin

## 🔮 Gelecek Özellikler

- [ ] Daha gelişmiş kıyafet modelleri
- [ ] Çoklu kıyafet tespiti
- [ ] Stil analizi
- [ ] Renk tespiti
- [ ] Materyal tanıma
- [ ] Marka tespiti

## 📄 Lisans

Bu proje MIT lisansı altında yayınlanmıştır.

## 🤝 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some amazing feature'`)
4. Branch'i push edin (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## 📞 İletişim

- Proje Maintainer: Aura AI Team
- E-posta: team@aura-ai.com
- Issue Tracker: [GitHub Issues](https://github.com/aura-ai/mvp/issues)

---

**Not**: Bu MVP versiyonu olup, production kullanımı için ek optimizasyonlar gerekebilir.
