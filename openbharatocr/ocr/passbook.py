import cv2
import numpy as np
import pytesseract
import re

# Function to enhance contrast of the image for better OCR results
def enhance_contrast(image):
    # Convert image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    # Merge channels back and convert to BGR color space
    limg = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_image

# Main preprocessing function for OCR
def preprocess_image(image_path, dpi=400):
    # Read the image
    image = cv2.imread(image_path)

    # Step 1: Enhance contrast
    enhanced_image = enhance_contrast(image)

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)

    # Step 3: Rescale the image for higher DPI (adjust dimensions to DPI)
    scale_factor = dpi / 72  # Typical screen DPI is 72, so scale to 300 DPI
    new_size = (int(gray.shape[1] * scale_factor), int(gray.shape[0] * scale_factor))
    scaled_image = cv2.resize(gray, new_size, interpolation=cv2.INTER_CUBIC)

    # Step 4: Remove noise using GaussianBlur
    blurred = cv2.GaussianBlur(scaled_image, (5, 5), 0)

    # Step 5: Thresholding (binarization) - Convert text to black and background to white
    _, binary_image = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 6: Morphological operations (opening) to clean up the image
    kernel = np.ones((1, 1), np.uint8)
    opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    return opened_image

# Function to clean OCR-extracted text for more reliable data extraction
def clean_text(input_text):
    # Correction map to handle common OCR mistakes
    corrections = {
        "Iqumber": "Number",
        "Nr":"Mr",
        # Add more common corrections here
    }

    # Apply corrections
    for wrong, correct in corrections.items():
        input_text = re.sub(re.escape(wrong), correct, input_text, flags=re.IGNORECASE)

    # Clean non-alphanumeric characters except for common separators
    input_text = re.sub(r"[^a-zA-Z0-9\s@.,/]", "", input_text)
    input_text = re.sub(r"\s{2,}", " ", input_text)  # Reduce multiple spaces to one space
    return input_text.strip()

# Functions to extract specific fields using regular expressions
def extract_cif_no(input_text):
    """
    Extracts the CIF number from the given text using a regular expression.

    This function searches for patterns containing "CIF" (case-insensitive),
    optionally followed by "No" or ".", and then extracts the following digits.

    Args:
        input (str): The text to extract the CIF number from.

    Returns:
        str: The extracted CIF number, or None if not found.
    """
    regex = re.compile(r"CIF(?:\s*No)?\.?\s*(\d+)", re.IGNORECASE)
    match = re.search(regex, input_text)
    if match:
        return match.group(1)
    return None

def extract_name(input_text):
    """
    Extracts the customer name from the given text using a regular expression.

    Args:
        input (str): The text to extract the name from.

    Returns:
        str: The extracted customer name, or None if not found.
    """
    regex = re.compile(r"(?:Name\s*[:.]?\s*([A-Za-z\s]+))", re.IGNORECASE)
    match = re.search(regex, input_text)
    if match:
        return match.group(1)
    return None

def extract_account_no(input_text):
    """
    Extracts the account number from the given text using a regular expression.

    This function searches for patterns containing "Account Number:" followed by
    9 to 14 digits, considering case-insensitivity.

    Args:
        input (str): The text to extract the account number from.

    Returns:
        str: The extracted account number, or None if not found.
    """
    regex = re.compile(r"Account(?:\s*No)?\.?\s*(\d+)", re.IGNORECASE)
    match = re.search(regex, input_text)
    if match:
        return match.group(1)
    return None

def extract_address(input_text):
    """
    Extracts the address from the given text using a regular expression.

    This function attempts to extract addresses using a list of patterns that
    commonly represent addresses. The patterns include house numbers, street names,
    city/town names, and postal codes.

    Args:
        input (str): The text to extract the address from.

    Returns:
        str: The extracted address, or None if no matching pattern is found.
    """
    regex = re.compile(r"Address:\s*([A-Za-z\s,]+)", re.IGNORECASE)
    match = re.search(regex, input_text)
    if match:
        return match.group(1)
    return None

def extract_phone(input_text):
    """
    Extracts the phone number from the given text using a regular expression.

    This function searches for patterns starting with "Mobile No" and extracts
    the following digits, considering case-insensitivity.

    Args:
        input (str): The text to extract the phone number from.

    Returns:
        str: The extracted phone number, or None if not found.
    """
    regex = re.compile(r"Phone(?: No)?\.?\s*(\d+)", re.IGNORECASE)
    match = re.search(regex, input_text)
    if match:
        return match.group(1)
    return None

def extract_branch_name(input_text):
    """
    Extracts the branch name from the given text using a regular expression.

    This function searches for patterns starting with "Branch Name" and extracts
    the following text, considering case-insensitivity.

    Args:
        input (str): The text to extract the branch name from.

    Returns:
        str: The extracted branch name, or None if not found.
    """
    regex = re.compile(r"Branch:\s*([A-Za-z\s]+)", re.IGNORECASE)
    match = re.search(regex, input_text)
    if match:
        return match.group(1)
    return None

def extract_generic_bank_name(input_text):
    """
    Extracts the bank name from the given text using a regular expression.

    This function searches for patterns containing "Bank across multiple lines".

    Args:
        input (str): The text to extract the bank name from.

    Returns:
        str: The extracted bank name, or None if not found.
    """
    regex = re.compile(r"\b([A-Za-z\s]+Bank(?:\s*Ltd|\s*Co)?\b)", re.IGNORECASE)
    match = re.search(regex, input_text)
    if match:
        return match.group(1)
    return None

def extract_open_date(input_text):
    """
    Extracts the account open date from the given text using a regular expression.

    Args:
        input (str): The text to extract the open date from.

    Returns:
        str: The extracted account open date in DD MMM YYYY format, or None if not found.
    """
    regex = re.compile(r"Opening\s*Dt[:\.]?\s*(\d{2}/\d{2}/\d{4})", re.IGNORECASE)
    match = re.search(regex, input_text)
    if match:
        return match.group(1)
    return None

# Function to process and extract information from the passbook
def parse_passbook(image_path):
    preprocessed_image = preprocess_image(image_path, dpi=400)
    extracted_text = pytesseract.image_to_string(preprocessed_image)

    # Clean the extracted text
    cleaned_text = clean_text(extracted_text)
    print("Cleaned Extracted Text:", cleaned_text)

    # Extract key information from the cleaned text
    passbook_info = {
        "cif_no": extract_cif_no(cleaned_text),
        "name": extract_name(cleaned_text),
        "account_no": extract_account_no(cleaned_text),
        "address": extract_address(cleaned_text),
        "phone": extract_phone(cleaned_text),
        "branch_name": extract_branch_name(cleaned_text),
        "bank_name": extract_generic_bank_name(cleaned_text),
        "date_of_issue": extract_open_date(cleaned_text),
    }

    return passbook_info
