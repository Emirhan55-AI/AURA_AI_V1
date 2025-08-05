"""
Aura AI - Kıyafet Sınıflandırma API

Bu modül, kıyafet fotoğraflarını analiz eden FastAPI sunucusunu içerir.
Kullanıcılar POST isteği ile fotoğraf gönderebilir ve kıyafet türlerini alabilir.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile
import os
import shutil
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import uuid

# Local imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.classifier import classify_clothing

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI uygulamasını oluştur
app = FastAPI(
    title="Aura AI - Kıyafet Sınıflandırma API",
    description="Kıyafet fotoğraflarını analiz ederek türlerini belirleyen AI servisi",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware ekle (mobil uygulama için gerekli)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Prod'da bunu kısıtla
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Geçici dosya dizini
TEMP_DIR = "temp_images"
os.makedirs(TEMP_DIR, exist_ok=True)

# Desteklenen dosya formatları
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def validate_image_file(file: UploadFile) -> bool:
    """
    Yüklenen dosyanın geçerli bir görüntü dosyası olup olmadığını kontrol et.
    
    Args:
        file (UploadFile): Yüklenen dosya
        
    Returns:
        bool: Dosya geçerliyse True
    """
    # Dosya uzantısını kontrol et
    if file.filename:
        file_ext = os.path.splitext(file.filename.lower())[1]
        if file_ext not in ALLOWED_EXTENSIONS:
            return False
    
    # Content type kontrol et
    if file.content_type and not file.content_type.startswith("image/"):
        return False
    
    return True

async def save_uploaded_file(file: UploadFile) -> str:
    """
    Yüklenen dosyayı geçici dizine kaydet.
    
    Args:
        file (UploadFile): Yüklenen dosya
        
    Returns:
        str: Kaydedilen dosyanın yolu
        
    Raises:
        HTTPException: Dosya kaydetme hatası durumunda
    """
    try:
        # Benzersiz dosya adı oluştur
        file_id = str(uuid.uuid4())
        file_ext = os.path.splitext(file.filename or "image.jpg")[1]
        temp_filename = f"{file_id}{file_ext}"
        temp_filepath = os.path.join(TEMP_DIR, temp_filename)
        
        # Dosyayı kaydet
        with open(temp_filepath, "wb") as buffer:
            content = await file.read()
            
            # Dosya boyutunu kontrol et
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Dosya boyutu çok büyük. Maksimum {MAX_FILE_SIZE // (1024*1024)}MB olmalı."
                )
            
            buffer.write(content)
        
        logger.info(f"Dosya kaydedildi: {temp_filepath}")
        return temp_filepath
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dosya kaydetme hatası: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Dosya kaydedilemedi"
        )

def cleanup_temp_file(filepath: str) -> None:
    """
    Geçici dosyayı sil.
    
    Args:
        filepath (str): Silinecek dosyanın yolu
    """
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Geçici dosya silindi: {filepath}")
    except Exception as e:
        logger.warning(f"Geçici dosya silinemedi: {e}")

@app.get("/")
async def root() -> Dict[str, str]:
    """
    Ana endpoint - API durumunu kontrol et.
    
    Returns:
        Dict[str, str]: API durum bilgisi
    """
    return {
        "message": "Aura AI - Kıyafet Sınıflandırma API",
        "version": "1.0.0",
        "status": "active",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Sağlık kontrolü endpoint'i.
    
    Returns:
        Dict[str, str]: Sağlık durumu
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/classify")
async def classify_clothing_endpoint(
    image: UploadFile = File(..., description="Sınıflandırılacak kıyafet fotoğrafı"),
    top_k: Optional[int] = 5
) -> Dict[str, Any]:
    """
    Kıyafet fotoğrafını sınıflandır.
    
    Args:
        image (UploadFile): Yüklenen kıyafet fotoğrafı
        top_k (Optional[int]): Döndürülecek en yüksek tahmin sayısı (varsayılan: 5)
        
    Returns:
        Dict[str, Any]: Sınıflandırma sonuçları
        
    Raises:
        HTTPException: Çeşitli hata durumlarında
    """
    temp_filepath = None
    
    try:
        # Dosya validasyonu
        if not validate_image_file(image):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Geçersiz dosya formatı. Desteklenen formatlar: JPG, JPEG, PNG, BMP, TIFF, WEBP"
            )
        
        # Top_k validasyonu
        if top_k and (top_k < 1 or top_k > 10):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="top_k değeri 1-10 arasında olmalıdır"
            )
        
        # Dosyayı kaydet
        temp_filepath = await save_uploaded_file(image)
        
        # Sınıflandırma yap
        logger.info(f"Sınıflandırma başlatılıyor: {image.filename}")
        result = classify_clothing(temp_filepath, top_k or 5)
        
        # Başarı durumunu kontrol et
        if not result.get('success', False):
            error_message = result.get('error', 'Bilinmeyen hata')
            logger.error(f"Sınıflandırma hatası: {error_message}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Sınıflandırma hatası: {error_message}"
            )
        
        # Sonucu temizle (dosya yolunu kaldır)
        clean_result = {
            "success": True,
            "filename": image.filename,
            "predictions": result['predictions'],
            "model_info": {
                "model_name": result['model_used'],
                "device": result['device']
            },
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Sınıflandırma başarılı: {image.filename}")
        return clean_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Beklenmeyen hata: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Sunucu hatası"
        )
    
    finally:
        # Geçici dosyayı temizle
        if temp_filepath:
            cleanup_temp_file(temp_filepath)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global hata yakalayıcı.
    
    Args:
        request: HTTP isteği
        exc: Yakalanan istisna
        
    Returns:
        JSONResponse: Hata yanıtı
    """
    logger.error(f"Yakalanmamış hata: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Sunucu hatası",
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Aura AI - Kıyafet Sınıflandırma API başlatılıyor...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
