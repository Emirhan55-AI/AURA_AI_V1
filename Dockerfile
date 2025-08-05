# Python 3.10 base image kullan
FROM python:3.10-slim

# Sistem bağımlılıklarını yükle
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizinini ayarla
WORKDIR /app

# Gereksinimler dosyasını kopyala ve bağımlılıkları yükle
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodunu kopyala
COPY . .

# Geçici dizin oluştur
RUN mkdir -p temp_images

# Port'u açığa çıkar
EXPOSE 8000

# Uygulamayı çalıştır
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
