import cv2
import pytesseract
import re
from pytesseract import Output


def extract_name(input):
    """
    Extracts the recipient's name from the given text.

    This function uses a regular expression to search for patterns commonly found in degree certificates that indicate the recipient's name. These patterns include phrases like "conferred upon [Name]" or "awarded to [Name]".

    Args:
    text: The text extracted from the degree certificate image.

    Returns:
    The extracted recipient's name as a string, or None if no name is found.
    """
    regex = re.compile(
        r"(?: conferred upon|Certify that)\s+([A-Z][a-zA-Z' -]+([A-Z][a-zA-Z' -]))|awarded to\s+([A-Z][a-z]+\s[A-Z][a-z]+(?:\s[A-Z][a-z]))",
        re.IGNORECASE,
    )
    match = re.search(regex, input)
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
        r"\b(?:Bachelor|Master|Doctor|Associate|B\.A\.|B\.Sc\.|M\.A\.|M\.Sc\.|Ph\.D\.|M\.B\.A\.|B\.E\.|B\.Tech|M\.E\.|M\.Tech|B\.Com|M\.Com|B\.Ed|M\.Ed|B\.Pharm|M\.Pharm|B\.Arch|M\.Arch|LL\.B|LL\.M|D\.Phil|D\.Lit|BFA|MFA|MRes|MSt)\s*(?:of\s*[A-Za-z]+)?\b",
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
        r"\b(?:[A-Za-z\s&]+(?:College|University|Institute|Academy|School|Polytechnic|Center|Centre|Faculty|Campus|School of [A-Za-z\s]+|College of [A-Za-z\s]+|Institute of [A-Za-z\s]+|University of [A-Za-z\s]+))\b",
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
        r"(?:year\s*of\s*passing|in\s*the\s*year|having\s*passed\s*the\s*examination\s*of|passed\s*in|dated)\s*[:\-]?\s*(\d{4})",
        re.IGNORECASE,
    )
    match = re.search(regex, input)
    if match:
        return match.group(1)
    return None


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
    image = cv2.imread(image_path)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
