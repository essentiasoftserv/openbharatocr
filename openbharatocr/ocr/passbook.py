import cv2
import easyocr
import re


def extract_name(input):
    regex = re.compile(r"Customer Name\s+([A-Z\s]+)")
    match = re.search(regex, input)
    if match:
        return match.group(1).strip()
    return None


def extract_open_date(input):
    regex = re.compile(r"Open Date\s*(\d{1,2} \w{3} \d{4})")
    match = re.search(regex, input)
    if match:
        return match.group(1)
    return None


def extract_bank_name(input):
    regex = re.compile(
        r"\b[A-Za-z\s&]+(?:BANK|BANK LTD|BANK LIMITED|CREDIT UNION)\b", re.MULTILINE
    )
    match = re.search(regex, input)
    if match:
        return match.group(0).strip()
    else:
        return None


def extract_phone(input):
    regex = re.compile(r"Mobile No\s*(\d+)", re.IGNORECASE)
    match = re.search(regex, input)
    if match:
        return match.group(1).strip()
    return None


def extract_branch_name(input):
    regex = re.compile(r"Branch Name\s*([A-Za-z\d\s-]+)", re.IGNORECASE)
    match = re.search(regex, input)
    if match:
        return match.group(1).strip()
    return None


def extract_nomination_name(input):
    regex = re.compile(r"Nomina(?:non|tion)\s+([A-Z][a-z]+\s[A-Z][a-z]+)")
    match = re.search(regex, input)
    if match:
        return match.group(1).strip()
    return None


def extract_email(input):
    regex = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    match = re.search(regex, input)
    if match:
        return match.group(0)
    return None


def extract_account_no(input):
    regex = re.compile(r"Account Number:\s*(\d{9,12})", re.IGNORECASE)
    match = re.search(regex, input)
    if match:
        return match.group(1)
    return None


def extract_cif_no(input):
    regex = re.compile(r"CIF(?: No)?\.?\s*(\d+)", re.IGNORECASE)
    match = re.search(regex, input)
    if match:
        return match.group(1)
    return None


def extract_address(input):
    regex = [
        r"\d+\s[A-Za-z\s,]+(?:Road|Street|Avenue|Boulevard|Lane|Drive|Court|Place|Square|Plaza|Terrace|Trail|Parkway|Circle)\s*,?\s*(?:\d{5}|\d{5}-\d{4})?",
        r"\d+\s[A-Za-z\s,]+(?:Road|Street|Avenue|Boulevard|Lane|Drive|Court|Place|Square|Plaza|Terrace|Trail|Parkway|Circle)",
        r"\d{1,5}\s[A-Za-z\s]+,\s*[A-Za-z\s]+,\s*[A-Za-z\s]+,\s*\d{5}",
        r"\d{1,5}\s[A-Za-z\s]+,\s*[A-Za-z\s]+,\s*[A-Za-z\s]+",
        r"\d{1,5}\s[A-Za-z\s]+,\s*[A-Za-z\s]+",
        r"[A-Za-z\s]+,\s*[A-Za-z\s]+,\s*[A-Za-z\s]+,\s*\d{5}",
        r"[A-Za-z\s]+,\s*[A-Za-z\s]+,\s*\d{5}",
        r"[A-Za-z\s]+,\s*\d{5}",
    ]
    for pattern in regex:
        match = re.search(pattern, input)
        if match:
            return match.group(0).strip()
    return None


def parse_passbook_frontpage(image_path):
    reader = easyocr.Reader(["en"])

    image = cv2.imread(image_path)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    results = reader.readtext(gray_image)

    extracted_text = " ".join([text for _, text, _ in results])

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
