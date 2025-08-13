import re
import pytesseract
from PIL import Image
import cv2
import numpy as np


def preprocess_for_bold_text(image):
    """
    Preprocesses an image to enhance bold text for improved OCR extraction.

    This function performs several image processing steps:

    1. Converts the image to grayscale.
    2. Applies morphological opening to reduce noise.
    3. Increases contrast to make bold text more prominent.
    4. Applies binarization with Otsu's thresholding.
    5. Applies sharpening to further enhance text edges.

    Args:
        image (numpy.ndarray): The image to preprocess.

    Returns:
        numpy.ndarray: The preprocessed image with enhanced bold text.
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


def extract_names(input):
    """
    Extracts first and last name from the given text using regular expressions.

    This function searches for patterns containing "Names" or "Surname" followed by
    one or more words, considering case-insensitivity.

    Args:
        input (str): The text to extract names from.

    Returns:
        tuple: A tuple containing the extracted first name and last name (both strings),
               or empty strings if not found.
    """
    name_regex = r"Names[\s:]+([A-Za-z\s]+)(?:\n|$)"
    surname_regex = r"Surname[\s:]+([A-Za-z\s]+)(?:\n|$)"

    name_match = re.search(name_regex, input, re.IGNORECASE)
    surname_match = re.search(surname_regex, input, re.IGNORECASE)

    name = name_match.group(1).strip() if name_match else ""
    surname = surname_match.group(1).strip() if surname_match else ""

    return name, surname


import datetime


def extract_all_dates(input):
    regex = r"\b(\d{2}[/\-.]\d{2}[/\-.](?:\d{4}|\d{2}))\b"
    matches = re.findall(regex, input)

    unique_dates = set()
    for match in matches:
        try:
            if match[2] == "/":
                date_obj = datetime.strptime(match, "%d/%m/%Y")
            else:
                date_obj = datetime.strptime(match, "%d-%m-%Y")
            unique_dates.add(date_obj)
        except ValueError:
            continue

    sorted_dates = sorted(unique_dates)
    return [date.strftime("%d-%m-%Y") for date in sorted_dates]


def extract_all_places(input):
    # Split input by lines
    lines = input.splitlines()
    places = []
    for line in lines:
        if (
            re.match(r'^[A-Z,. -\'"]+$', line)
            and line.strip()
            and line.strip().count(" ") > 0
        ):
            places.append(line.strip())
    return places


def extract_passport_number(input):
    """
    Extracts the passport number from the given text using a regular expression.

    This function searches for a pattern starting with a capital letter followed by 7 digits.

    Args:
        input (str): The text to extract the passport number from.

    Returns:
        str: The extracted passport number, or an empty string if not found.
    """
    regex = r"[A-Z][0-9]{7}"
    match = re.search(regex, input)
    passport_number = match.group(0) if match else ""

    return passport_number


def extract_details(input):
    """
    Extracts name, surname, and gender from the given text using a combination of
    regular expressions and heuristics.

    This function assumes lines with only uppercase characters separated by spaces
    represent the name and surname. It also checks the last line for "M" or "F" to infer gender.

    Args:
        input (str): The text to extract details from.

    Returns:
        tuple: A tuple containing extracted gender (string), first name (string),
               and last name (string). Empty strings are returned if not found.
    """
    lines = input.split("\n")
    clean = []
    for line in lines:
        clean_line = re.sub(r"[^a-zA-Z\s]", "", line)
        clean_line = " ".join(word for word in clean_line.split() if len(word) > 1)
        if (
            re.match(r"^[A-Z\s]{3,}$", clean_line)
            and "INDIA" not in clean_line
            and "REPUBLIC" not in clean_line
            and "PASSPORT" not in clean_line
        ):
            clean.append(clean_line.strip())

    name = clean[1] if len(clean) > 1 else ""
    surname = clean[0] if len(clean) > 0 else ""

    gender = ""

    if len(clean) > 1:
        if "M" in clean[-1]:
            gender = "Male"
        elif "F" in clean[-1]:
            gender = "Female"

    return gender, name, surname


def extract_passport_details(image_path):
    """
    Extracts passport details from an image using a combination of OCR and text processing.

    This function performs the following steps:

    1. Reads the image using Pillow.
    2. Extracts text using Tesseract (saves a JPEG copy for pre-processing).
    3. Preprocesses the image (JPEG copy) to enhance bold text for OCR.
    4. Extracts text again using Tesseract on the preprocessed image.
    5. Extracts dates (DoB, DoI, Expiry) using regular expressions from the original text.
    6. Extracts passport number using a regular expression from the original text.
    7. Extracts places (PoB, PoI) based on text following the last date.
    8. Extracts name, surname, and gender using heuristics on cleaned preprocessed text.

    Args:
        image_path (str): The path to the passport image.

    Returns:
        dict: A dictionary containing extracted passport details with keys like
              "Name", "Surname", "Passport Number", etc.
    """
    image = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(image)
    image.save("image.jpg", "JPEG")
    image = cv2.imread("image.jpg")
    preprocessed = preprocess_for_bold_text(image)
    cv2.imwrite("preprocessed_image.jpg", preprocessed)

    image = Image.open("preprocessed_image.jpg")
    clean_text = pytesseract.image_to_string(image)

    dates = extract_all_dates(extracted_text)
    dob = dates[0] if len(dates) > 0 else ""
    doi = dates[1] if len(dates) > 1 else ""
    expiry_date = dates[2] if len(dates) > 2 else ""

    passport_number = extract_passport_number(extracted_text)

    places = extract_all_places(extracted_text)
    pob = places[-2] if len(places) > 1 else ""
    poi = places[-1] if len(places) > 0 else ""

    gender, name, surname = extract_details(clean_text)

    return {
        "Name": name,
        "Surname": surname,
        "Passport Number": passport_number,
        "Gender": gender,
        "Place of Birth": pob,
        "Date of Birth": dob,
        "Place of Issue": poi,
        "Date of Issue": doi,
        "Expiry Date": expiry_date,
    }


def passport(image_path):
    """
    Extracts passport details from an image using the extract_passport_details function.

    Args:
        image_path (str): The path to the passport image.

    Returns:
        dict: A dictionary containing extracted passport details.
    """
    return extract_passport_details(image_path)
