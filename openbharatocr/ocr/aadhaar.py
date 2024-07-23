import re
import cv2
import pytesseract
from PIL import Image
import tempfile
import uuid
import imghdr
import numpy as np
import io


def preprocess_to_processed(image_path):
    """
    Preprocesses an image to convert it into a high-contrast black and white format
    for improved text extraction.

    Args:
        image_path (str): The path to the input image.

    Returns:
        numpy.ndarray: The preprocessed image in a black and white format.
    """
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Apply GaussianBlur to smooth the image and reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to convert the image to binary (black and white)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )

    # Invert the colors of the image
    inverted_image = cv2.bitwise_not(binary)

    # Apply dilation and erosion to make the text bolder and cleaner
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(inverted_image, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    # Sharpen the image using a kernel
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(eroded, -1, sharpen_kernel)

    return sharpened_image


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


def extract_front_aadhaar_details_version2(image_path):
    """
    Extracts details from the front side of an Aadhaar card image
    using a pre-processing step that converts the image to a sketch-like format.

    This version aims to improve extraction accuracy in cases where Version 1 might struggle.

    Args:
        image_path (str): The path to the front Aadhaar card image.

    Returns:
        dict: A dictionary containing extracted front Aadhaar Details details.
    """
    preprocessed_image = preprocess_to_processed(image_path)
    _, preprocessed_image_encoded = cv2.imencode(".jpg", preprocessed_image)
    image_pil = Image.open(io.BytesIO(preprocessed_image_encoded.tobytes()))
    extracted_text = pytesseract.image_to_string(image_pil)
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


def extract_back_aadhaar_details_version2(image_path):
    """
    Extracts details from the back side of an Aadhaar card image
    using a pre-processing step that converts the image to a sketch-like format.

    This version aims to improve extraction accuracy in cases where Version 1 might struggle.

    Args:
        image_path (str): The path to the back Aadhaar card image.

    Returns:
        dict: A dictionary containing extracted back Aadhaar Details details.
    """
    preprocessed_image = preprocess_to_processed(image_path)
    _, preprocessed_image_encoded = cv2.imencode(".jpg", preprocessed_image)
    image_pil = Image.open(io.BytesIO(preprocessed_image_encoded.tobytes()))
    extracted_text = pytesseract.image_to_string(image_pil)
    fathers_name = extract_fathers_name(extracted_text)
    address = extract_address(extracted_text)
    return {
        "Father's Name": fathers_name,
        "Address": address,
    }


def validate_results(result1, result2):
    """
    This function matches both the versions and validates which version is more accurate and use info from that version.
    """
    validated_result = {}
    for key in result1:
        if result1[key] == result2[key]:
            validated_result[key] = result1[key]
        else:
            # Use more reliable data or a default if both are incorrect
            validated_result[key] = (
                result1[key] if len(result1[key]) > len(result2[key]) else result2[key]
            )
    return validated_result


def front_aadhaar(image_path):
    """
    Extracts details from the front side of an Aadhaar card image.

    Calls the `extract_front_aadhaar_details` function to perform the extraction.

    Args:
        image_path (str): The path to the image file of the Aadhaar card front side.

    Returns:
        dict: A dictionary containing the extracted details from the front side.
    """
    result_v1 = extract_front_aadhaar_details(image_path)
    result_v2 = extract_front_aadhaar_details_version2(image_path)
    final_result = validate_results(result_v1, result_v2)
    return final_result


def back_aadhaar(image_path):
    """
    Extracts details from the back side of an Aadhaar card image.

    Calls the `extract_back_aadhaar_details` function to perform the extraction.

    Args:
        image_path (str): The path to the image file of the Aadhaar card back side.

    Returns:
        dict: A dictionary containing the extracted details from the back side.
    """
    result_v1 = extract_back_aadhaar_details(image_path)
    result_v2 = extract_back_aadhaar_details_version2(image_path)
    final_result = validate_results(result_v1, result_v2)
    return final_result
