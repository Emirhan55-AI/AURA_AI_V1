"""
Tests package init dosyası.
"""

# Test configuration
import pytest
import sys
import os

# Proje kök dizinini Python path'ine ekle
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Test konfigürasyonu
pytest_plugins = []
