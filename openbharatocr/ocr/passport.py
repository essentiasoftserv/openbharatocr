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


def extract_names(input):
    name_regex = r"Names[\s:]+([A-Za-z\s]+)(?:\n|$)"
    surname_regex = r"Surname[\s:]+([A-Za-z\s]+)(?:\n|$)"

    name_match = re.search(name_regex, input, re.IGNORECASE)
    surname_match = re.search(surname_regex, input, re.IGNORECASE)

    name = name_match.group(1).strip() if name_match else ""
    surname = surname_match.group(1).strip() if surname_match else ""

    return name, surname


def extract_all_dates(input):
    regex = r"\b(\d{2}[/\-.]\d{2}[/\-.](?:\d{4}|\d{2}))\b"
    dates = re.findall(regex, input)
    dates = sorted(dates, key=lambda x: int(re.split(r"[-/]", x)[-1]))

    seen = set()
    unique_dates = []

    for date in dates:
        if date not in seen:
            seen.add(date)
            unique_dates.append(date)

    return unique_dates


def extract_all_places(input):
    dates = re.findall(r"\b(\d{2}[/\-.]\d{2}[/\-.](?:\d{4}|\d{2}))\b", input)
    last_date = dates[-1] if dates else None

    all_places = []

    if last_date:
        places = input.split("\n")
        for place in places:
            if re.match(r'^[A-Z,. -\'"]+$', place) and place.strip():
                all_places.append(place)

    return all_places


def extract_passport_number(input):
    regex = r"[A-Z][0-9]{7}"
    match = re.search(regex, input)
    passport_number = match.group(0) if match else ""

    return passport_number


def extract_details(input):
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
    return extract_passport_details(image_path)
