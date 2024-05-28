import pytesseract
from PIL import Image
import re


def extract_name(input):
    regex = re.compile(r'Customer Name\s+([A-Z\s]+)\s+A/C')
    match = re.search(regex, input)
    if match:
        return match.group(1).strip()
    else:
        return None


def extract_open_date(input):
    regex = re.compile(r'(?:AIC|A/C) Open Date:\s*(\d{2}\s\w{3}\s\d{4})')
    match = re.search(regex, input)
    if match:
        return match.group(1)
    else:
        return None


def extract_bank_name(input):
    regex = re.compile(r'^[A-Z\s]+BANK\sLTD\.', re.MULTILINE)
    match = re.search(regex, input)
    if match:
        return match.group(0).strip()
    else:
        return None



def extract_branch_name(input):
    regex = re.compile(r"Branch Name\s*:\s*(.+)", re.IGNORECASE)
    match = re.search(regex, input)
    if match:
        return match.group(1).strip()
    else:
        return None


def extract_nomination_name(input):
    regex = re.compile(r'Nomina(?:non|tion)\s+([A-Z][a-z]+\s[A-Z][a-z]+)')
    match = re.search(regex, input)
    if match:
        return match.group(1).strip()
    else:
        return None


def extract_email(input):
    regex = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
    match = re.search(regex, input)
    if match:
        return match.group(0)
    else:
        return None


def extract_account_no(input):
    regex = re.compile(r"Account Number:\s*(\d{9,12})", re.IGNORECASE)
    match = re.search(regex, input)
    if match:
        return match.group(1)
    else:
        return None


def extract_cif_no(input):
    regex = re.compile(r'CIF No\s*[>:]\s*(\d+)')
    match = re.search(regex, input)
    if match:
        return match.group(1)
    else:
        return None


def extract_address(input):
    regex = [
        r'\d+\s[A-Za-z\s,]+(?:Road|Street|Avenue|Boulevard|Lane|Drive|Court|Place|Square|Plaza|Terrace|Trail|Parkway|Circle)\s*,?\s*(?:\d{5}|\d{5}-\d{4})?',
        r'\d+\s[A-Za-z\s,]+(?:Road|Street|Avenue|Boulevard|Lane|Drive|Court|Place|Square|Plaza|Terrace|Trail|Parkway|Circle)', 
        r'\d{1,5}\s[A-Za-z\s]+,\s*[A-Za-z\s]+,\s*[A-Za-z\s]+,\s*\d{5}',
        r'\d{1,5}\s[A-Za-z\s]+,\s*[A-Za-z\s]+,\s*[A-Za-z\s]+',
        r'\d{1,5}\s[A-Za-z\s]+,\s*[A-Za-z\s]+',
        r'[A-Za-z\s]+,\s*[A-Za-z\s]+,\s*[A-Za-z\s]+,\s*\d{5}',
        r'[A-Za-z\s]+,\s*[A-Za-z\s]+,\s*\d{5}',
        r'[A-Za-z\s]+,\s*\d{5}' 
    ]
    
    for pattern in regex:
        match = re.search(pattern, input)
        if match:
            return match.group(0).strip()
    return None


def extract_phone(input):
    regex = re.compile(r"Mobile No\s*:\s*(.*)", re.IGNORECASE)
    match = re.search(regex, input)
    if match:
        return match.group(1).strip()
    else:
        return None


def parse_passbook_frontpage(image_path):
    image = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(image)
    print(extracted_text)
    passbook_info = {
        "cif_no": extract_cif_no(extracted_text),
        "name": extract_name(extracted_text),
        "account_no": extract_account_no(extracted_text),
        "address": extract_address(extracted_text),
        "phone": extract_phone(extracted_text),
        "email": extract_email(extracted_text),
        "nomination_name": extract_nomination_name(extracted_text),
        "branch_name": extract_branch_name(extracted_text),
        "bank_name": extract_bank_name(extracted_text),
        "date_of_issue": extract_open_date(extracted_text),
    }

    return passbook_info

parse_passbook_frontpage("/home/rishabh/openbharatocr/openbharatocr/ocr/passbook1.jpeg")