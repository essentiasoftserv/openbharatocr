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


def extract_name(text):
    name_regex = r"Name\s*:\s*([A-Za-z]+\s+[A-Za-z]+).*?(?=(\n[A-Za-z\s]*:|$))"
    match = re.search(name_regex, text, re.IGNORECASE)
    name = match.group(1).strip() if match else ""

    if name == "":
        name_regex = (
            r"Name\s*of\s*Child\s*:\s*([A-Za-z]+\s+[A-Za-z]+).*?(?=(\n[A-Za-z\s]*:|$))"
        )
        match = re.search(name_regex, text, re.IGNORECASE)
        name = match.group(1).strip() if match else ""

    return name


def extract_address(text):
    address_regex = r"Permanent\s*Address\s*of\s*parents\s*:\s*([A-Z0-9\s]+(?:\s[A-Z0-9\s]+)*?)\s[A-Z]+\b"
    match = re.search(address_regex, text, re.IGNORECASE)
    address = match.group(1).strip() if match else ""

    if address == "":
        address_regex = (
            r"Permanent\s*Address\s*:\s*([A-Z0-9\s]+(?:\s[A-Z0-9\s]+)*?)\s[A-Z]+\b"
        )
        match = re.search(address_regex, text, re.IGNORECASE)
        address = match.group(1).strip() if match else ""

    return address


def extract_date_of_birth(text):
    date_of_birth_regex = r"Date\s*of\s*Birth\s*:\s*([0-9A-Za-z\s-]+)"
    match = re.search(date_of_birth_regex, text, re.IGNORECASE)
    date_of_birth = match.group(1).strip() if match else ""
    return date_of_birth


def extract_date_of_issue(text):
    date_of_issue_regex = r"Date\s*of\s*Issue\s*[\+:\s]*([0-9A-Za-z\s-]+?\d{4})"
    match = re.search(date_of_issue_regex, text, re.IGNORECASE)
    date_of_issue = match.group(1).strip() if match else ""
    return date_of_issue


def extract_registration_no(text):
    registration_no_regex = r"Registration\s*[:No]*\s*:\s*(\d+)"
    match = re.search(registration_no_regex, text, re.IGNORECASE)
    registration_no = match.group(1).strip() if match else ""

    if registration_no == "":
        registration_no_regex = r"Registration\s*no.\s*[:No]*\s*:\s*(\d+)"
        match = re.search(registration_no_regex, text, re.IGNORECASE)
        registration_no = match.group(1).strip() if match else ""

    if registration_no == "":
        registration_no_regex = r"Registration\s*Number\s*[:No]*\s*:\s*(\d+)"
        match = re.search(registration_no_regex, text, re.IGNORECASE)
        registration_no = match.group(1).strip() if match else ""

    return registration_no


def clean_extracted_text(text):
    # Remove unnecessary extra words, spaces, and lines
    cleaned_text = re.sub(r"\s+", " ", text)
    cleaned_text = re.sub(r"\n+", "\n", cleaned_text)
    return cleaned_text


def extract_birth_certificate_details(image_path):
    image = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(image)
    cleaned_text = clean_extracted_text(extracted_text)

    name = extract_name(cleaned_text)
    address = extract_address(cleaned_text)
    date_of_birth = extract_date_of_birth(cleaned_text)
    date_of_issue = extract_date_of_issue(cleaned_text)
    registration_no = extract_registration_no(cleaned_text)

    return {
        "name": name,
        "address": address,
        "date_of_birth": date_of_birth,
        "date_of_issue": date_of_issue,
        "registration_no": registration_no,
    }


def birth_certificate(image_path):
    return extract_birth_certificate_details(image_path)
