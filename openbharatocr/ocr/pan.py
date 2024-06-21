import re
import pytesseract
import imghdr
from PIL import Image
import cv2
import numpy as np


def clean_input(match):
    """
    Cleans the extracted text by splitting lines and removing stopwords.

    Args:
        match (list): A list of extracted text chunks.

    Returns:
        list: A cleaned list of individual names.
    """
    cleaned = []

    for name in match:
        split_name = name.split("\n")
        for chunk in split_name:
            cleaned.append(chunk)

    stopwords = ["INDIA", "OF", "TAX", "GOVT", "DEPARTMENT", "INCOME"]

    names = [
        name.strip()
        for name in cleaned
        if not any(word in name for word in stopwords) and len(name.strip()) > 3
    ]

    return names


def extract_all_names(input):
    """
    Extracts all names from the given text using a regular expression and performs basic cleaning.

    Args:
        input (str): The text to extract names from.

    Returns:
        list: A list of extracted names.
    """
    regex = r"\n[A-Z\s]+\b"
    match = re.findall(regex, input)

    names = []
    cleaned = clean_input(match)
    return cleaned


def extract_pan(input):
    """
    Extracts the PAN number from the given text using a regular expression.

    Args:
        input (str): The text to extract the PAN number from.

    Returns:
        str: The extracted PAN number, or an empty string if not found.
    """
    regex = r"[A-Z]{5}[0-9]{4}[A-Z]"
    match = re.search(regex, input)
    pan_number = match.group(0) if match else ""

    return pan_number


def extract_dob(input):
    """
    Extracts the date of birth from the given text using a regular expression.

    Args:
        input (str): The text to extract the date of birth from.

    Returns:
        str: The extracted date of birth in a common format (DD/MM/YYYY), or an empty string if not found.
    """
    regex = r"\b(\d{2}[/\-.]\d{2}[/\-.](?:\d{4}|\d{2}))\b"
    match = re.search(regex, input)
    dob = match.group(0) if match else ""

    return dob


def extract_pan_details(image_path):
    """
    Extracts PAN details (full name, parent's name, date of birth, PAN number) from a PAN card image.

    This version attempts extraction from the original image and a converted JPEG version
    to improve compatibility.

    Args:
        image_path (str): The path to the PAN card image.

    Returns:
        dict: A dictionary containing extracted PAN details.
    """
    image = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(image)

    format = imghdr.what(image_path)
    if format != "jpeg":
        image.save("image.jpg", "JPEG")

        converted_image = Image.open("image.jpg")
        converted_image_text = pytesseract.image_to_string(converted_image)

        extracted_text += converted_image_text

    names = extract_all_names(extracted_text)
    full_name = names[0] if len(names) > 0 else ""
    parents_name = names[1] if len(names) > 1 else ""
    dob = extract_dob(extracted_text)
    pan_number = extract_pan(extracted_text)

    return {
        "Full Name": full_name,
        "Parent's Name": parents_name,
        "Date of Birth": dob,
        "PAN Number": pan_number,
    }


def preprocess_for_sketch(image_path):
    """
    Preprocesses an image to convert it into a black and white sketch-like look
    for improved text extraction.

    This function performs several image processing steps:

    1. Reads the image using OpenCV.
    2. Converts the image to grayscale.
    3. Applies Gaussian blur to smooth the image and reduce noise.
    4. Applies adaptive thresholding to convert the image to binary (black and white).
    5. Applies morphological operations (opening) to reduce noise and enhance text regions.
    6. Inverts the image colors for better text recognition by Tesseract.

    Args:
        image_path (str): The path to the image.

    Returns:
        numpy.ndarray: The preprocessed image in a black and white sketch-like format.
    """
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to smooth the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to binarize the image
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 9
    )

    # Apply morphological operations to reduce noise and enhance text regions
    kernel = np.ones((1, 1), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # Invert the colors of the image
    inverted_image = cv2.bitwise_not(opened)

    return inverted_image


def extract_pan_details_version2(image_path):
    """
    Extracts PAN details (full name, parent's name, date of birth, PAN number) from a PAN card image
    using a pre-processing step that converts the image to a sketch-like format.

    This version aims to improve extraction accuracy in cases where Version 1 might struggle.

    Args:
        image_path (str): The path to the PAN card image.

    Returns:
        dict: A dictionary containing extracted PAN details.
    """

    # Preprocess the image to convert it into a black and white sketch-like look
    preprocessed_image = preprocess_for_sketch(image_path)

    # Perform text extraction using Tesseract on the preprocessed image
    extracted_text = pytesseract.image_to_string(preprocessed_image)

    # Extract information from the extracted text
    names = extract_all_names(extracted_text)
    full_name = names[0] if len(names) > 0 else ""
    parents_name = names[1] if len(names) > 1 else ""
    dob = extract_dob(extracted_text)
    pan_number = extract_pan(extracted_text)

    return {
        "Full Name": full_name,
        "Parent's Name": parents_name,
        "Date of Birth": dob,
        "PAN Number": pan_number,
    }


def pan(image_path):
    """
    Extracts PAN details (full name, parent's name, date of birth, PAN number) from a PAN card image.

    This function attempts extraction using two versions:

    1. Version 1: Extracts details from the original image and a converted JPEG version
       to improve compatibility.
    2. Version 2: If any details are missing from Version 1, it applies a pre-processing
       step that converts the image to a sketch-like format and then extracts details.

    Args:
        image_path (str): The path to the PAN card image.

    Returns:
        dict: A dictionary containing extracted PAN details, with missing details from
              Version 1 filled in by Version 2 if necessary.
    """
    # Run Version 1
    result = extract_pan_details(image_path)

    # Check if all details are extracted
    if not all(
        [
            result["Full Name"],
            result["Parent's Name"],
            result["Date of Birth"],
            result["PAN Number"],
        ]
    ):
        # Run Version 2 if any detail is missing
        result_v2 = extract_pan_details_version2(image_path)
        # Update the missing details with results from Version 2
        result["Full Name"] = result["Full Name"] or result_v2["Full Name"]
        result["Parent's Name"] = result["Parent's Name"] or result_v2["Parent's Name"]
        result["Date of Birth"] = result["Date of Birth"] or result_v2["Date of Birth"]
        result["PAN Number"] = result["PAN Number"] or result_v2["PAN Number"]

    return result
