import pytesseract
from PIL import Image
import re


def extract_name(input):
    regex = re.compile(r"Name\s*:\s*(.*)", re.IGNORECASE)
    match = re.search(regex, input)
    if match:
        return match.group(1).strip()
    else:
        return None


def extract_cif_no(input):
    regex = re.compile(r"CIF\s*:\s*(\d+)", re.IGNORECASE)
    match = re.search(regex, input)
    if match:
        return match.group(1)
    else:
        return None


def extract_account_no(input):
    regex = re.compile(r"Account\s*No\s*:\s*(\d+)", re.IGNORECASE)
    match = re.search(regex, input)
    if match:
        return match.group(1)
    else:
        return None


def extract_address(input):
    regex = re.compile(r"Address\s*:\s*(.*)", re.IGNORECASE)
    match = regex.search(input)
    if match:
        return match.group(1).strip()
    else:
        return None


def extract_phone(input):
    regex = re.compile(r"Phone\s*:\s*(.*)", re.IGNORECASE)
    match = re.search(regex, input)
    if match:
        return match.group(1).strip()
    else:
        return None


def extract_email(input):
    regex = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    match = re.search(regex, input)
    if match:
        return match.group(0)
    else:
        return None


def extract_nomination_name(input):
    regex = re.compile(r"Nomination\s*:\s*(.*)", re.IGNORECASE)
    match = re.search(regex, input)
    if match:
        return match.group(1).strip()
    else:
        return None


def extract_branch_name(input):
    regex = re.compile(r"Branch\s*:\s*(.*)", re.IGNORECASE)
    match = re.search(regex, input)
    if match:
        return match.group(1).strip()
    else:
        return None


def extract_bank_name(input):
    regex = re.compile(r"Bank\s*:\s*(.*)", re.IGNORECASE)
    match = re.search(regex, input)
    if match:
        return match.group(1).strip()
    else:
        return None


def extract_date_of_issue(input):
    regex = re.compile(r"Date\s*of\s*Issue\s*:\s*(\d{2}/\d{2}/\d{4})", re.IGNORECASE)
    match = re.search(regex, input)
    if match:
        return match.group(1)
    else:
        return None


def parse_passbook_frontpage(image_path):
    image = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(image)

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
        "date_of_issue": extract_date_of_issue(extracted_text),
    }

    return passbook_info
