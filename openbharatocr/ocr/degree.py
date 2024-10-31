import cv2
import pytesseract
import re
from pytesseract import Output
from PIL import Image

def preprocess_image(image_path):
    """
    Preprocesses the image to enhance text for OCR.

    Args:
        image_path (str): The path to the image.

    Returns:
        numpy.ndarray: The preprocessed image.
    """
    # Read the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to smooth the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to create a binary image
    binary = cv2.adaptiveThreshold(blurred, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Invert colors for better OCR performance
    inverted_image = cv2.bitwise_not(binary)

    return inverted_image
import re

def extract_name(text):
    """
    Extracts the recipient's name from the given text.

    This function uses a regular expression to search for patterns commonly found in degree certificates that indicate the recipient's name.

    Args:
    text: The text extracted from the degree certificate image.

    Returns:
    The extracted recipient's name as a string, or None if no name is found.
    """
    patterns = [
        r"conferred on",
        r"conferred upon",
        r"awarded to",
        r"certify that",
        r"certifies that",
        r"testify that",
        r"known that",
        r"admits",
        r"granted"
    ]
    
    # Create a regex pattern by joining all patterns with an optional whitespace and capturing the name
    name_pattern = r"(?:{})\s+([A-Z][a-zA-Z' -]+(?:\s[A-Z][a-zA-Z' -]+)*)".format("|".join(patterns))
    
    # Compile the regex with case insensitivity
    regex = re.compile(name_pattern, re.IGNORECASE)
    
    # Search for the pattern in the input text
    match = regex.search(text)
    
    if match:
        return match.group(1).strip()
    
    return None


def extract_degree_name(input):
    """
    Extracts the degree name from the given text.

    This function uses a regular expression to match common degree abbreviations (e.g., B.A., Ph.D.) and full names (e.g., Bachelor of Science) found in degree certificates.

    Args:
    text: The text extracted from the degree certificate image.

    Returns:
    The extracted degree name as a string, or None if no degree name is found.
    """
    regex = re.compile(
        r"\b(?:Bachelor|Bachelors|Master|Doctor|Associate|B\.A\.|B\.Sc\.|M\.A\.|M\.Sc\.|Ph\.D\.|M\.B\.A\.|B\.E\.|B\.Tech|M\.E\.|M\.Tech|B\.Com|M\.Com|B\.Ed|M\.Ed|B\.Pharm|M\.Pharm|B\.Arch|M\.Arch|LL\.B|LL\.M|D\.Phil|D\.Lit|BFA|MFA|MRes|MSt)\s*(?:of\s*[A-Za-z]+)?\b",
        re.IGNORECASE,
    )
    match = re.search(regex, input)
    if match:
        return match.group(0).strip()
    return None


def extract_institution_name(input):
    """
    Extracts the institution name (university, college, etc.) from the given text.

    This function uses a regular expression to match various formats of institution names that might be present in degree certificates. It covers names like "Massachusetts Institute of Technology" or "University of California, Berkeley".

    Args:
    text: The text extracted from the degree certificate image.

    Returns:
    The extracted institution name as a string, or None if no institution name is found.
    """
    regex = re.compile(
        r"\b(?:College of [A-Za-z\s]+|[A-Z][a-z]*\sInstitute of [A-Za-z]+|(?:UNIVERSITY OF [A-Za-z]+|[w A-Za-z]*\s(University|Aniversity)?))",
        re.IGNORECASE,
    )
    match = re.search(regex, input)
    if match:
        return match.group(0).strip()
    return None


def extract_year_of_passing(input):
    """

    This function uses a regular expression to search for common patterns indicating the year of passing in degree certificates, such as "year of passing" or "in the year".

    Args:
    text: The text extracted from the degree certificate image.

    Returns:
    The extracted year of passing as a string, or None if no year of passing is found.
    """
    regex = re.compile(
        r"\b(nineteen (hundred|hundred and) (one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|twenty[- ]one|twenty[- ]two|twenty[- ]three|twenty[- ]four|twenty[- ]five|twenty[- ]six|twenty[- ]seven|twenty[- ]eight|twenty[- ]nine|thirty|forty|fifty|sixty|seventy|eighty|ninety)([- ](one|two|three|four|five|six|seven|eight|nine))?|\d{4}|(two|too|tfoo|tw)\s*(thousand|thousand and)\s*(one|two|three|four|five|six|seven|eight|nine|ten|tex|eleven|twelve|thirteen|fourteen|fifteen|fiventy|sixteen|seventeen|eighteen|nineteen|twenty|twenty[- ]one|twenty[- ]two|twenty[- ]three|twenty[- ]four|twenty[- ]five|twenty[- ]six|twenty[- ]seven|twenty[- ]eight|twenty[- ]nine))\b",
        re.IGNORECASE,
    )
    match = re.search(regex, input)
    if match:
        return match.group(1)
    return None


def check_image_quality(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    variance_of_laplacian = cv2.Laplacian(image, cv2.CV_64F).var()
    sharpness_threshold = 150.0

    mean_brightness = image.mean()
    brightness_threshold = 150.0

    if (
        variance_of_laplacian < sharpness_threshold
        or mean_brightness < brightness_threshold
    ):
        return False
    return True


def parse_degree_certificate(image_path):
    """
    Parses information from a degree certificate image.

    This function takes the path to a degree certificate image and attempts to extract the following information using regular expressions and Tesseract OCR:

        * Recipient's Name
        * Degree Name
        * University Name
        * Year of Passing

    Args:
        image_path (str): The path to the degree certificate image file.

    Returns:
        dict: A dictionary containing the extracted information with keys "Name", "Degree Name", "University Name", and "Year of Passing". The values can be None if the corresponding information is not found in the image.
    """
    if not check_image_quality(image_path):
        return "Image quality is too low to process."

    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    preprocessed_image = preprocess_image(image_path)
    extracted_text = pytesseract.image_to_string(gray_image, output_type=Output.STRING)
    
    degree_info = {
        "Name": extract_name(extracted_text),
        "Degree Name": extract_degree_name(extracted_text),
        "University Name": extract_institution_name(extracted_text),
        "Year of Passing": extract_year_of_passing(extracted_text),
    }

    return degree_info


def degree(image_path):
    """
    Convenience function to parse degree certificate information.

    This function simply calls `parse_degree_certificate` and returns the resulting dictionary.

    Args:
        image_path (str): The path to the degree certificate image file.

    Returns:
        dict: A dictionary containing the extracted information from the degree certificate (same as the output of `parse_degree_certificate`).
    """
    return parse_degree_certificate(image_path)
