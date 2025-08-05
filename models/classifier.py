"""
Kıyafet sınıflandırma modülü.

Bu modül, görüntülerdeki kıyafetleri tespit edip sınıflandırır.
Hugging Face transformers kütüphanesi kullanılarak geliştirilmiştir.
"""

import torch
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import logging
from typing import Dict, List, Tuple, Any
import os

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClothingClassifier:
    """
    Kıyafet sınıflandırıcı sınıfı.
    
    Bu sınıf, kıyafet görüntülerini analiz edip kategorilerine ayırmak için
    önceden eğitilmiş bir Vision Transformer (ViT) modeli kullanır.
    """
    
    def __init__(self, model_name: str = "google/vit-base-patch16-224"):
        """
        Sınıflandırıcıyı başlat.
        
        Args:
            model_name (str): Kullanılacak Hugging Face model adı
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Cihaz olarak {self.device} kullanılacak")
        
        # Model ve pipeline'ı yükle
        try:
            self._load_model()
            logger.info("Model başarıyla yüklendi")
        except Exception as e:
            logger.error(f"Model yüklenirken hata: {e}")
            raise
    
    def _load_model(self):
        """Model ve pipeline'ı yükle."""
        try:
            # İlk olarak image classification pipeline ile deneyelim
            self.classifier = pipeline(
                "image-classification",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1
            )
            logger.info(f"Pipeline başarıyla yüklendi: {self.model_name}")
        except Exception as e:
            logger.warning(f"Pipeline yüklenemedi: {e}")
            # Fallback: Manuel model yükleme
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.classifier = None
            logger.info("Model manuel olarak yüklendi")
    
    def _preprocess_image(self, image_path: str) -> Image.Image:
        """
        Görüntüyü ön işleme tabi tut.
        
        Args:
            image_path (str): Görüntü dosyasının yolu
            
        Returns:
            PIL.Image: İşlenmiş görüntü
        """
        try:
            image = Image.open(image_path)
            
            # RGBA'yı RGB'ye çevir
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
        except Exception as e:
            logger.error(f"Görüntü ön işleme hatası: {e}")
            raise
    
    def _map_to_clothing_categories(self, predictions: List[Dict]) -> List[Dict]:
        """
        Genel sınıfları kıyafet kategorilerine eşle.
        
        Args:
            predictions (List[Dict]): Model tahminleri
            
        Returns:
            List[Dict]: Kıyafet kategorilerine eşlenmiş tahminler
        """
        # Kıyafet ile ilgili anahtar kelimeler
        clothing_keywords = {
            'shirt': ['shirt', 'blouse', 'top', 'tee'],
            'pants': ['pants', 'trousers', 'jeans', 'leggings'],
            'dress': ['dress', 'gown', 'frock'],
            'jacket': ['jacket', 'coat', 'blazer', 'cardigan'],
            'shoes': ['shoe', 'boot', 'sneaker', 'sandal'],
            'skirt': ['skirt', 'mini', 'maxi'],
            'shorts': ['shorts', 'short'],
            'sweater': ['sweater', 'pullover', 'jumper'],
            'hat': ['hat', 'cap', 'beanie'],
            'bag': ['bag', 'purse', 'backpack', 'handbag']
        }
        
        mapped_predictions = []
        
        for pred in predictions:
            label = pred['label'].lower()
            score = pred['score']
            
            # En uygun kıyafet kategorisini bul
            best_category = 'other'
            for category, keywords in clothing_keywords.items():
                if any(keyword in label for keyword in keywords):
                    best_category = category
                    break
            
            mapped_predictions.append({
                'category': best_category,
                'original_label': pred['label'],
                'confidence': score
            })
        
        return mapped_predictions
    
    def classify_clothing(self, image_path: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Kıyafet görüntüsünü sınıflandır.
        
        Args:
            image_path (str): Sınıflandırılacak görüntünün yolu
            top_k (int): Döndürülecek en yüksek tahmin sayısı
            
        Returns:
            Dict[str, Any]: Sınıflandırma sonuçları
        """
        try:
            # Dosya varlığını kontrol et
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Görüntü dosyası bulunamadı: {image_path}")
            
            # Görüntüyü ön işle
            image = self._preprocess_image(image_path)
            
            # Sınıflandırma yap
            if self.classifier:
                # Pipeline kullan
                predictions = self.classifier(image, top_k=top_k)
            else:
                # Manuel tahmin
                inputs = self.processor(image, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                # En yüksek skorları al
                top_scores, top_indices = torch.topk(probabilities, top_k)
                
                predictions = []
                for score, idx in zip(top_scores[0], top_indices[0]):
                    label = self.model.config.id2label[idx.item()]
                    predictions.append({
                        'label': label,
                        'score': score.item()
                    })
            
            # Kıyafet kategorilerine eşle
            clothing_predictions = self._map_to_clothing_categories(predictions)
            
            # Sonuçları formatla
            result = {
                'success': True,
                'image_path': image_path,
                'predictions': clothing_predictions,
                'raw_predictions': predictions,
                'model_used': self.model_name,
                'device': self.device
            }
            
            logger.info(f"Sınıflandırma başarılı: {image_path}")
            return result
            
        except Exception as e:
            logger.error(f"Sınıflandırma hatası: {e}")
            return {
                'success': False,
                'error': str(e),
                'image_path': image_path
            }

# Global classifier instance (lazy loading)
_classifier_instance = None

def get_classifier() -> ClothingClassifier:
    """
    Global classifier instance'ını döndür (singleton pattern).
    
    Returns:
        ClothingClassifier: Sınıflandırıcı örneği
    """
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = ClothingClassifier()
    return _classifier_instance

def classify_clothing(image_path: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Kıyafet sınıflandırma için kolay kullanım fonksiyonu.
    
    Args:
        image_path (str): Sınıflandırılacak görüntünün yolu
        top_k (int): Döndürülecek en yüksek tahmin sayısı
        
    Returns:
        Dict[str, Any]: Sınıflandırma sonuçları
    """
    classifier = get_classifier()
    return classifier.classify_clothing(image_path, top_k)
