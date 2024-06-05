import cv2
import pytesseract
import re
from pytesseract import Output


def extract_name(input):
    regex = re.compile(
        r"(?: conferred upon|Certify that)\s+([A-Z][a-zA-Z' -]+([A-Z][a-zA-Z' -]))|awarded to\s+([A-Z][a-z]+\s[A-Z][a-z]+(?:\s[A-Z][a-z]))",
        re.IGNORECASE,
    )
    match = re.search(regex, input)
    if match:
        return match.group(1).strip()
    return None


def extract_degree_name(input):
    regex = re.compile(
        r"\b(?:Bachelor|Master|Doctor|Associate|B\.A\.|B\.Sc\.|M\.A\.|M\.Sc\.|Ph\.D\.|M\.B\.A\.|B\.E\.|B\.Tech|M\.E\.|M\.Tech|B\.Com|M\.Com|B\.Ed|M\.Ed|B\.Pharm|M\.Pharm|B\.Arch|M\.Arch|LL\.B|LL\.M|D\.Phil|D\.Lit|BFA|MFA|MRes|MSt)\s*(?:of\s*[A-Za-z]+)?\b",
        re.IGNORECASE,
    )
    match = re.search(regex, input)
    if match:
        return match.group(0).strip()
    return None


def extract_institution_name(input):
    regex = re.compile(
        r"\b(?:[A-Za-z\s&]+(?:College|University|Institute|Academy|School|Polytechnic|Center|Centre|Faculty|Campus|School of [A-Za-z\s]+|College of [A-Za-z\s]+|Institute of [A-Za-z\s]+|University of [A-Za-z\s]+))\b",
        re.IGNORECASE,
    )
    match = re.search(regex, input)
    if match:
        return match.group(0).strip()
    return None


def extract_year_of_passing(input):
    regex = re.compile(
        r"(?:year\s*of\s*passing|in\s*the\s*year|having\s*passed\s*the\s*examination\s*of|passed\s*in|dated)\s*[:\-]?\s*(\d{4})",
        re.IGNORECASE,
    )
    match = re.search(regex, input)
    if match:
        return match.group(1)
    return None


def parse_degree_certificate(image_path):

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
