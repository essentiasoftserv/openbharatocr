import re
import pytesseract
from PIL import Image
import cv2
import numpy as np


def preprocess_for_bold_text(image):
    """
    Preprocesses an image to enhance the extraction of bold text.

    This function applies morphological operations (opening and sharpening) to the grayscale version of the input image. These operations aim to improve the contrast and definition of bold text features, potentially leading to better OCR results.

    Args:
        image: The input image as a NumPy array.

    Returns:
        A NumPy array representing the preprocessed image.
    """
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
    """
    Extracts the recipient's name from birth certificate text using regular expressions.

    This function attempts to match patterns like "Name: [Name]" or "Name of Child: [Name]" in the provided text. It extracts the captured name after these patterns and returns it.

    Args:
        text: The extracted text from the birth certificate image.

    Returns:
        str: The extracted name if found, otherwise an empty string.
    """
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
    """
    Extracts the permanent address from birth certificate text using regular expressions.

    This function targets patterns like "Permanent Address of Parents: [Address]" or "Permanent Address: [Address]" to extract the address information.

    Args:
        text: The extracted text from the birth certificate image.

    Returns:
        str: The extracted address if found, otherwise an empty string.
    """

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
    """
    Extracts the date of birth from birth certificate text using regular expressions.

    This function searches for patterns like "Date of Birth: [Date of Birth]" and returns the captured date information.

    Args:
        text: The extracted text from the birth certificate image.

    Returns:
        str: The extracted date of birth if found, otherwise an empty string.
    """
    date_of_birth_regex = r"Date\s*of\s*Birth\s*:\s*([0-9A-Za-z\s-]+)"
    match = re.search(date_of_birth_regex, text, re.IGNORECASE)
    date_of_birth = match.group(1).strip() if match else ""
    return date_of_birth


def extract_date_of_issue(text):
    """
    Extracts the date of issue from birth certificate text using regular expressions.

    This function searches for patterns like "Date of Issue: [Date of Issue]" and returns the captured date information.

    Args:
        text: The extracted text from the birth certificate image.

    Returns:
        str: The extracted date of issue if found, otherwise an empty string.
    """
    date_of_issue_regex = r"Date\s*of\s*Issue\s*[\+:\s]*([0-9A-Za-z\s-]+?\d{4})"
    match = re.search(date_of_issue_regex, text, re.IGNORECASE)
    date_of_issue = match.group(1).strip() if match else ""
    return date_of_issue


def extract_registration_no(text):
    """
    Extracts the registration number from birth certificate text using regular expressions.

    This function attempts to match variations of "Registration No.: [Registration Number]" or "Registration Number: [Registration Number]" patterns to extract the registration number.

    Args:
        text: The extracted text from the birth certificate image.

    Returns:
        str: The extracted registration number if found, otherwise an empty string.
    """

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
    """
    Cleans the extracted text by removing unnecessary spaces and line breaks.

    This function uses regular expressions to remove extra spaces and combine multiple newlines into single newlines. This can improve the readability and processing of the extracted information.

    Args:
        text: The extracted text from the birth certificate image.

    Returns:
        str: The cleaned text.
    """
    # Remove unnecessary extra words, spaces, and lines
    cleaned_text = re.sub(r"\s+", " ", text)
    cleaned_text = re.sub(r"\n+", "\n", cleaned_text)
    return cleaned_text


def extract_birth_certificate_details(image_path):
    """
    Extracts birth certificate details from an image using OCR and text processing.

    This function performs the following steps:
        1. Opens the image using Pillow (PIL Fork).
        2. Extracts text from the image using Tesseract OCR.
        3. Cleans the extracted text using `clean_extracted_text`.
        4. Extracts various details like name, address, date of birth, date of issue, and registration number using specific extraction functions (`extract_name`, `extract_address`, etc.).
        5. Returns a dictionary containing the extracted details.

    Args:
        image_path: The path to the birth certificate image file.

    Returns:
        dict: A dictionary containing the extracted birth certificate details (name, address, etc.).
    """
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
    """
    Convenience function to extract birth certificate details.

    This function simply calls `extract_birth_certificate_details(image_path)` and returns the resulting dictionary.

    Args:
        image_path: The path to the birth certificate image file.

    Returns:
        dict: A dictionary containing the extracted birth certificate details (same as the output of `extract_birth_certificate_details`).
    """
    return extract_birth_certificate_details(image_path)
