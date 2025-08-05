"""
Create synthetic test images for fashion classification testing
"""
from PIL import Image, ImageDraw
import os

def create_test_images():
    """Create synthetic test images with clear fashion items"""
    test_data_dir = "test_data"
    
    # 1. Blue T-shirt
    blue_tshirt = Image.new('RGB', (224, 224), color='white')
    draw = ImageDraw.Draw(blue_tshirt)
    # T-shirt shape: rectangular top
    draw.rectangle([60, 80, 164, 150], fill=(0, 100, 255), outline='black', width=2)
    blue_tshirt.save(os.path.join(test_data_dir, "blue_tshirt.jpg"))
    
    # 2. Navy trouser
    navy_trouser = Image.new('RGB', (224, 224), color='white')
    draw = ImageDraw.Draw(navy_trouser)
    # Trouser shape: two legs
    draw.rectangle([70, 120, 100, 200], fill=(20, 30, 80), outline='black', width=2)  # Left leg
    draw.rectangle([124, 120, 154, 200], fill=(20, 30, 80), outline='black', width=2)  # Right leg
    draw.rectangle([70, 100, 154, 130], fill=(20, 30, 80), outline='black', width=2)  # Waist
    navy_trouser.save(os.path.join(test_data_dir, "navy_trouser.jpg"))
    
    # 3. Black coat
    black_coat = Image.new('RGB', (224, 224), color='white')
    draw = ImageDraw.Draw(black_coat)
    # Coat shape: large rectangular
    draw.rectangle([50, 70, 174, 180], fill=(20, 20, 25), outline='black', width=2)
    # Coat collar
    draw.rectangle([80, 70, 144, 85], fill=(30, 30, 35), outline='black', width=1)
    black_coat.save(os.path.join(test_data_dir, "black_coat.jpg"))
    
    # 4. Red dress
    red_dress = Image.new('RGB', (224, 224), color='white')
    draw = ImageDraw.Draw(red_dress)
    # Dress shape: A-line dress
    draw.polygon([112, 80, 80, 160, 144, 160], fill=(235, 50, 50), outline='black', width=2)
    draw.rectangle([95, 80, 129, 120], fill=(235, 50, 50), outline='black', width=2)  # Top part
    red_dress.save(os.path.join(test_data_dir, "red_dress.jpg"))
    
    # 5. Brown bag
    brown_bag = Image.new('RGB', (224, 224), color='white')
    draw = ImageDraw.Draw(brown_bag)
    # Bag shape: small rectangular
    draw.rectangle([90, 130, 134, 170], fill=(139, 69, 19), outline='black', width=2)
    # Handle
    draw.line([(100, 130), (100, 120), (124, 120), (124, 130)], fill='black', width=3)
    brown_bag.save(os.path.join(test_data_dir, "brown_bag.jpg"))
    
    # 6. White sneaker
    white_sneaker = Image.new('RGB', (224, 224), color='white')
    draw = ImageDraw.Draw(white_sneaker)
    # Sneaker shape
    draw.ellipse([70, 160, 154, 190], fill=(240, 240, 240), outline='black', width=2)
    draw.rectangle([85, 150, 139, 165], fill=(240, 240, 240), outline='black', width=2)
    white_sneaker.save(os.path.join(test_data_dir, "white_sneaker.jpg"))
    
    print("✅ Test images created successfully!")
    return [
        ("blue_tshirt.jpg", "T-shirt/top"),
        ("navy_trouser.jpg", "Trouser"), 
        ("black_coat.jpg", "Coat"),
        ("red_dress.jpg", "Dress"),
        ("brown_bag.jpg", "Bag"),
        ("white_sneaker.jpg", "Sneaker")
    ]

if __name__ == "__main__":
    create_test_images()
