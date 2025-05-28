import re
import pytesseract
from PIL import Image
import cv2
import numpy as np


def preprocess_for_bold_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    contrast = cv2.addWeighted(opening, 2, opening, -0.5, 0)
    _, binary = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    sharpened = cv2.filter2D(
        binary, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    )
    return sharpened


def clean_extracted_text(text):
    cleaned_text = re.sub(r"\s+", " ", text)
    cleaned_text = re.sub(r"\n+", "\n", cleaned_text)
    return cleaned_text


def extract_name(text):
    match = re.search(r'\bNAME\s+([A-Z. ]+)\s+SEX\b', text)
    if match:
        return match.group(1).strip()
    return ""
def extract_address(text):
    # Find "MAIN ROAD" and capture everything up to the first "INDIA"
    pattern = r'(MAIN ROAD.*?INDIA)'
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        address = match.group(1).strip()
        # Clean up whitespace and fix spacing before dots
        address = re.sub(r'\s+', ' ', address)
        address = address.replace(" .", ".")
        return address
    else:
        return ""

def extract_date_of_birth(text):
    # Look for date near "DATE OF BIRTH"
    match = re.search(r'DATE\s+OF\s+BIRTH\s+PLACE\s+OF\s+BIRTH\s+(\d{2})\s*(\d{2})\s*(\d{4})', text)
    if match:
        return f"{match.group(1)}/{match.group(2)}/{match.group(3)}"
    return ""

def extract_date_of_issue(text):
    # Looks for "DATE OF ISSUE" followed by a date (may be separated by spaces or dashes)
    match = re.search(r'DATE\s+OF\s+ISSUE.*?(\d{2})[-\s]?(\d{2})[-\s]?(\d{4})', text)
    return f"{match.group(1)}/{match.group(2)}/{match.group(3)}" if match else ""

def extract_registration_no(text):
    # Find pattern like "REGISTRATION NO; REGISTRATION DATE 1310/2007 01.08 2007"
    match = re.search(r'REGISTRATION\s+NO[:;]?\s*REGISTRATION\s+DATE\s*(\d{3,}/\d{4})', text)
    if match:
        return match.group(1).strip()
    
    # Fallback: look for first occurrence of NNNN/YYYY (at least 3 digits before slash, 4 digits after)
    fallback_match = re.search(r'(\d{3,}/\d{4})', text)
    return fallback_match.group(1).strip() if fallback_match else ""


def extract_birth_certificate_details(image_path):
    image_cv = cv2.imread(image_path)
    preprocessed = preprocess_for_bold_text(image_cv)

    pil_img = Image.fromarray(preprocessed)
    custom_config = r'--oem 3 --psm 4'
    extracted_text = pytesseract.image_to_string(pil_img, config=custom_config)

    cleaned_text = clean_extracted_text(extracted_text)
    print("----- OCR EXTRACTED TEXT -----")
    print(cleaned_text)
    print("------------------------------")

    return {
        "name": extract_name(cleaned_text),
        "address": extract_address(cleaned_text),
        "date_of_birth": extract_date_of_birth(cleaned_text),
        "date_of_issue": extract_date_of_issue(cleaned_text),
        "registration_no": extract_registration_no(cleaned_text),
    }


def birth_certificate(image_path):
    return extract_birth_certificate_details(image_path)


if __name__ == "__main__":  
    image_path = "/home/rishabh/openbharatocr/openbharatocr/ocr/BC1.jpeg"
    details = birth_certificate(image_path)
    print(details)