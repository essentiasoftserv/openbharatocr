## author:    Kunal Kumar Kushwaha
# website:   http://www.essentia.dev

# import the required packages
from openbharatocr.ocr.api import (
    PANCardExtractor,
    AadhaarOCR,
    passport,
    voter_id_front,
    voter_id_back,
    vehicle_registration,
    water_bill,
    birth_certificate,
    degree,
)


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
]
