"""
Yardımcı fonksiyonlar modülü.

Bu modül, projenin çeşitli yerlerinde kullanılan yardımcı fonksiyonları içerir.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Logging yapılandırmasını ayarla.
    
    Args:
        log_level (str): Log seviyesi (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        logging.Logger: Yapılandırılmış logger
    """
    logger = logging.getLogger("aura_ai")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def validate_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
    """
    Dosya uzantısının geçerli olup olmadığını kontrol et.
    
    Args:
        filename (str): Dosya adı
        allowed_extensions (List[str]): İzin verilen uzantılar listesi
        
    Returns:
        bool: Uzantı geçerliyse True
    """
    if not filename:
        return False
    
    file_ext = os.path.splitext(filename.lower())[1]
    return file_ext in [ext.lower() for ext in allowed_extensions]

def generate_file_hash(filepath: str) -> Optional[str]:
    """
    Dosyanın MD5 hash'ini hesapla.
    
    Args:
        filepath (str): Dosya yolu
        
    Returns:
        Optional[str]: MD5 hash veya None (hata durumunda)
    """
    try:
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception:
        return None

def format_prediction_results(predictions: List[Dict], confidence_threshold: float = 0.1) -> List[Dict]:
    """
    Tahmin sonuçlarını formatla ve filtrele.
    
    Args:
        predictions (List[Dict]): Ham tahmin sonuçları
        confidence_threshold (float): Minimum güven eşiği
        
    Returns:
        List[Dict]: Formatlanmış ve filtrelenmiş sonuçlar
    """
    formatted_results = []
    
    for pred in predictions:
        confidence = pred.get('confidence', 0)
        
        if confidence >= confidence_threshold:
            formatted_results.append({
                'category': pred.get('category', 'unknown'),
                'confidence': round(confidence, 3),
                'confidence_percentage': round(confidence * 100, 1),
                'original_label': pred.get('original_label', ''),
            })
    
    # Güven skoruna göre sırala
    formatted_results.sort(key=lambda x: x['confidence'], reverse=True)
    
    return formatted_results

def create_response_metadata(success: bool, **kwargs) -> Dict[str, Any]:
    """
    API yanıtı için metadata oluştur.
    
    Args:
        success (bool): İşlem başarılı mı
        **kwargs: Ek metadata alanları
        
    Returns:
        Dict[str, Any]: Metadata dictionary
    """
    metadata = {
        'success': success,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }
    
    metadata.update(kwargs)
    return metadata

class FileManager:
    """Dosya yönetimi için yardımcı sınıf."""
    
    @staticmethod
    def ensure_directory_exists(directory_path: str) -> bool:
        """
        Dizinin var olduğundan emin ol, yoksa oluştur.
        
        Args:
            directory_path (str): Dizin yolu
            
        Returns:
            bool: İşlem başarılıysa True
        """
        try:
            os.makedirs(directory_path, exist_ok=True)
            return True
        except Exception:
            return False
    
    @staticmethod
    def safe_delete_file(filepath: str) -> bool:
        """
        Dosyayı güvenli şekilde sil.
        
        Args:
            filepath (str): Silinecek dosya yolu
            
        Returns:
            bool: Silme işlemi başarılıysa True
        """
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                return True
            return True  # Dosya zaten yok
        except Exception:
            return False
    
    @staticmethod
    def get_file_size(filepath: str) -> Optional[int]:
        """
        Dosya boyutunu bytes cinsinden al.
        
        Args:
            filepath (str): Dosya yolu
            
        Returns:
            Optional[int]: Dosya boyutu veya None (hata durumunda)
        """
        try:
            return os.path.getsize(filepath)
        except Exception:
            return None

class ImageUtils:
    """Görüntü işleme yardımcı fonksiyonları."""
    
    @staticmethod
    def get_image_info(image_path: str) -> Dict[str, Any]:
        """
        Görüntü bilgilerini al.
        
        Args:
            image_path (str): Görüntü dosyası yolu
            
        Returns:
            Dict[str, Any]: Görüntü bilgileri
        """
        try:
            from PIL import Image
            
            with Image.open(image_path) as img:
                return {
                    'width': img.width,
                    'height': img.height,
                    'mode': img.mode,
                    'format': img.format,
                    'size_bytes': os.path.getsize(image_path)
                }
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def is_valid_image(image_path: str) -> bool:
        """
        Dosyanın geçerli bir görüntü olup olmadığını kontrol et.
        
        Args:
            image_path (str): Görüntü dosyası yolu
            
        Returns:
            bool: Geçerli görüntüyse True
        """
        try:
            from PIL import Image
            
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception:
            return False
