"""
OpenBharatOCR - Optical Character Recognition for Indian Government Documents

Author: Kunal Kumar Kushwaha
Website: http://www.essentia.dev
License: Apache 2.0
"""

__version__ = "0.4.2"
__author__ = "Kunal Kumar Kushwaha"
__email__ = "kunal@essentia.dev"

# Import the main API functions
from openbharatocr.ocr.pan import PANCardExtractor
from openbharatocr.ocr.aadhaar import AadhaarOCR
from openbharatocr.ocr.driving_licence import driving_licence
from openbharatocr.ocr.passport import passport
from openbharatocr.ocr.voter_id import voter_id_front, voter_id_back
from openbharatocr.ocr.vehicle_registration import vehicle_registration
from openbharatocr.ocr.water_bill import water_bill
from openbharatocr.ocr.birth_certificate import birth_certificate
from openbharatocr.ocr.degree import degree

# Create convenience functions for the API
def pan(image_path):
    """Extract PAN card information from an image."""
    extractor = PANCardExtractor()
    return extractor.extract_pan_details(image_path)


def front_aadhaar(image_path):
    """Extract Aadhaar card front side information from an image."""
    ocr = AadhaarOCR()
    return ocr.extract_front_aadhaar_details(image_path)


def back_aadhaar(image_path):
    """Extract Aadhaar card back side information from an image."""
    ocr = AadhaarOCR()
    return ocr.extract_back_aadhaar_details(image_path)


# Export public API
__all__ = [
    "pan",
    "front_aadhaar",
    "back_aadhaar",
    "driving_licence",
    "passport",
    "voter_id_front",
    "voter_id_back",
    "vehicle_registration",
    "water_bill",
    "birth_certificate",
    "degree",
    "__version__",
    "__author__",
    "__email__",
]
