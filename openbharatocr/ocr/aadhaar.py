import re
import cv2
import pytesseract
from PIL import Image
import tempfile
import uuid


def extract_name(input):
    name_regex = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"
    names = re.findall(name_regex, input)
    full_name = ""
    for name in names:
        if "Government" not in name and "India" not in name:
            full_name = name
            break

    return full_name


def extract_fathers_name(input):
    regex = r"(?:S.?O|D.?O)[:\s]*([A-Za-z]+(?: [A-Za-z]+)*)"
    match = re.findall(regex, input)
    fathers_name = ""
    if match:
        fathers_name = match[-1]

    return fathers_name


def extract_aadhaar(input):
    regex = r"\b\d{4}\s?\d{4}\s?\d{4}\b"
    match = re.search(regex, input)
    aadhaar_number = match.group(0) if match else ""

    return aadhaar_number


def extract_dob(input):
    regex = r"\b(\d{2}/\d{2}/\d{4})\b"
    match = re.search(regex, input)
    dob = match.group(0) if match else ""

    return dob


def extract_yob(input):
    regex = r"\b\d{4}\b"
    match = re.search(regex, input)
    yob = match.group(0) if match else ""

    return yob


def extract_gender(input):
    if re.search("Female", input) or re.search("FEMALE", input):
        return "Female"
    if re.search("Male", input) or re.search("MALE", input):
        return "Male"
    return "Other"


def extract_address(input):
    regex = r"Address:\s*((?:.|\n)*?\d{6})"
    match = re.search(regex, input)
    address = match.group(1) if match else ""

    return address


def extract_back_aadhaar_details(image_path):
    image = Image.open(image_path)

    extracted_text = pytesseract.image_to_string(image)

    fathers_name = extract_fathers_name(extracted_text)
    address = extract_address(extracted_text)

    return {
        "Father's Name": fathers_name,
        "Address": address,
    }


def extract_front_aadhaar_details(image_path):
    image = Image.open(image_path)

    extracted_text = pytesseract.image_to_string(image)

    full_name = extract_name(extracted_text)
    dob = extract_dob(extracted_text)
    gender = extract_gender(extracted_text)
    aadhaar_number = extract_aadhaar(extracted_text)

    if dob == "":
        dob = extract_yob(extracted_text)

    return {
        "Full Name": full_name,
        "Date/Year of Birth": dob,
        "Gender": gender,
        "Aadhaar Number": aadhaar_number,
    }


def front_aadhaar(image_path):
    return extract_front_aadhaar_details(image_path)


def back_aadhaar(image_path):
    return extract_back_aadhaar_details(image_path)
