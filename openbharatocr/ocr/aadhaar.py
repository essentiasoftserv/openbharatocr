import re
import cv2
import pytesseract
from PIL import Image
import tempfile
import uuid


def extract_name(input_text):
    """
    Extracts the full name from the given text using a regular expression.

    Args:
        input_text (str): The text to extract the name from.

    Returns:
        str: The extracted full name, or an empty string if no name is found.
    """
    name_regex = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"
    names = re.findall(name_regex, input_text)
    full_name = ""
    for name in names:
        if "Government" not in name and "India" not in name:
            full_name = name
            break

    return full_name


def extract_fathers_name(input_text):
    """
    Extracts the father's name from the given text using a regular expression.

    Args:
        input_text (str): The text to extract the father's name from.

    Returns:
        str: The extracted father's name, or an empty string if not found.
    """
    regex = r"(?:S.?O|D.?O)[:\s]*([A-Za-z]+(?: [A-Za-z]+)*)"
    match = re.findall(regex, input_text)
    fathers_name = ""
    if match:
        fathers_name = match[-1]

    return fathers_name


def extract_aadhaar(input_text):
    """
    Extracts the Aadhaar number from the given text using a regular expression.

    Args:
        input_text (str): The text to extract the Aadhaar number from.

    Returns:
        str: The extracted Aadhaar number, or an empty string if not found.
    """
    regex = r"\b\d{4}\s?\d{4}\s?\d{4}\b"
    match = re.search(regex, input_text)
    aadhaar_number = match.group(0) if match else ""

    return aadhaar_number


def extract_dob(input_text):
    """
    Extracts the date of birth from the given text using a regular expression.

    Args:
        input_text (str): The text to extract the date of birth from.

    Returns:
        str: The extracted date of birth in DD/MM/YYYY format, or an empty string if not found.
    """
    regex = r"\b(\d{2}/\d{2}/\d{4})\b"
    match = re.search(regex, input_text)
    dob = match.group(0) if match else ""

    return dob


def extract_yob(input_text):
    """
    Extracts the year of birth from the given text using a regular expression.

    Used as a fallback if the date of birth is not found in DD/MM/YYYY format.

    Args:
        input_text (str): The text to extract the year of birth from.

    Returns:
        str: The extracted year of birth in YYYY format, or an empty string if not found.
    """
    regex = r"\b\d{4}\b"
    match = re.search(regex, input_text)
    yob = match.group(0) if match else ""

    return yob


def extract_gender(input_text):
    """
    Extracts the gender from the given text using string comparisons.

    Args:
        input_text (str): The text to extract the gender from.

    Returns:
        str: "Female", "Male", or "Other" based on the extracted information.
    """
    if re.search("Female", input_text) or re.search("FEMALE", input_text):
        return "Female"
    if re.search("Male", input_text) or re.search("MALE", input_text):
        return "Male"
    return "Other"


def extract_address(input_text):
    """
    Extracts the address from the given text using a regular expression.

    Args:
        input_text (str): The text to extract the address from.

    Returns:
        str: The extracted address, or an empty string if not found.
    """
    regex = r"Address:\s*((?:.|\n)*?\d{6})"
    match = re.search(regex, input_text)
    address = match.group(1) if match else ""

    return address


def extract_back_aadhaar_details(image_path):
    """
    Extracts details from the back side of an Aadhaar card image.

    Uses Tesseract OCR to convert the image to text and then extracts relevant information
    using regular expressions.

    Args:
        image_path (str): The path to the image file of the Aadhaar card back side.

    Returns:
        dict: A dictionary containing the extracted details, including:
            - Father's Name (str)
            - Address (str)
    """
    image = Image.open(image_path)

    extracted_text = pytesseract.image_to_string(image)

    fathers_name = extract_fathers_name(extracted_text)
    address = extract_address(extracted_text)

    return {
        "Father's Name": fathers_name,
        "Address": address,
    }


def extract_front_aadhaar_details(image_path):
    """
    Extracts details from the front side of an Aadhaar card image.

    Uses Tesseract OCR to convert the image to text and then extracts relevant information
    using regular expressions.

    Args:
        image_path (str): The path to the image file of the Aadhaar card front side.

    Returns:
        dict: A dictionary containing the extracted details, including:
            - Full Name (str)
            - Date/Year of Birth (str)
            - Gender (str)
            - Aadhaar Number (str)
    """
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
    """
    Extracts details from the front side of an Aadhaar card image.

    Calls the `extract_front_aadhaar_details` function to perform the extraction.

    Args:
        image_path (str): The path to the image file of the Aadhaar card front side.

    Returns:
        dict: A dictionary containing the extracted details from the front side.
    """
    return extract_front_aadhaar_details(image_path)


def back_aadhaar(image_path):
    """
    Extracts details from the back side of an Aadhaar card image.

    Calls the `extract_back_aadhaar_details` function to perform the extraction.

    Args:
        image_path (str): The path to the image file of the Aadhaar card back side.

    Returns:
        dict: A dictionary containing the extracted details from the back side.
    """
    return extract_back_aadhaar_details(image_path)
