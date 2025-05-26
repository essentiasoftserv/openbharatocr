import cv2
import pytesseract
import re
import numpy as np
from datetime import datetime
from fuzzywuzzy import fuzz
from fuzzywuzzy import process


# Preprocess the image to enhance bold text for OCR
def preprocess_for_bold_text(image):
    """Preprocesses an image to enhance bold text for OCR.

    Args:
        image (numpy.ndarray): The image to preprocess.

    Returns:
        numpy.ndarray: The preprocessed image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    contrast = cv2.addWeighted(opening, 2, opening, -0.5, 0)

    _, binary = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    sharpened = cv2.filter2D(
        binary, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    )

    return sharpened


# FuzzyWuzzy correction function
def fuzzy_correct(text, possible_corrections):
    """Corrects text using fuzzy matching from possible corrections."""
    if not text:
        return text
    best_match = process.extractOne(text, possible_corrections)
    return best_match[0] if best_match and best_match[1] > 80 else text


# Extracting the amount from the image (using various strategies)
def extract_amount(extracted_text):
    amount_keywords = [
        "Total Amount Due",
        "Amount Due",
        "Net Amount",
        "Amount to be Paid",
        "TOTAL AMOUNT DUE",
        "Amount Due Rs",
        "Amount Due in Rs",
        "OTAL DUE",
        "TOTAL",
    ]

    for keyword in amount_keywords:
        keyword_regex = rf"(?i){keyword}\s*:?\s*(\D*\d+\.?\d*)"
        match = re.search(keyword_regex, extracted_text)
        if match:
            amount = re.sub(r"[^\d.]", "", match.group(1))
            return float(amount) if amount else "Amount not found"

    amount_regex = r"(?i)(?:\D?)(\d{1,3}(?:[\s.,]?\d{3})*(?:[\s.,]?\d{1,2})?)"
    amounts = re.findall(amount_regex, extracted_text)
    valid_amounts = [
        amount.replace(" ", "").replace(",", "") for amount in amounts if amount
    ]
    valid_amounts = [
        float(amount)
        for amount in valid_amounts
        if amount.replace(".", "", 1).isdigit()
    ]
    return max(valid_amounts) if valid_amounts else "Amount not found"


# Extracting phone numbers
def extract_phone_numbers(extracted_text):
    phone_regex = r"(?i)(?:phone|telephone|mobile|contact|tel|cell|no|number)\s*[:\-]?\s*(\+?\d{1,3}[-.\s]?)?(\(?\d{3,5}\)?[-.\s]?)?\d{3,5}[-.\s]?\d{4,7}"
    phone_numbers = re.findall(phone_regex, extracted_text)
    phone_numbers = ["".join(part for part in match if part) for match in phone_numbers]
    return phone_numbers if phone_numbers else ["Phone number not found"]


# Extracting bill number using regex and fuzzy correction
def extract_bill_number(extracted_text):
    bill_number_regex = r"(?i)(?:bill no\.?|bill number|bill id|invoice no\.?|soa #|transaction id)\s*[:\-]?\s*(\d{10,})"
    bill_match = re.search(bill_number_regex, extracted_text)
    if bill_match:
        return fuzzy_correct(
            bill_match.group(1).strip(), ["bill", "number", "no", "id"]
        )

    additional_bill_number_regex = (
        r"(?i)(?:bill id|transaction id|invoice no)\s*[:\-]?\s*(\d{10,})"
    )
    additional_bill_match = re.search(additional_bill_number_regex, extracted_text)
    return (
        fuzzy_correct(
            additional_bill_match.group(1).strip(), ["bill", "number", "no", "id"]
        )
        if additional_bill_match
        else "Bill number not found"
    )


# Extracting account number using regex and fuzzy correction (updated regex for new format)
def extract_account_number(extracted_text):
    account_keywords = [
        "Contract Account No",
        "Account No",
        "Account Number",
        "Contract Account",
        "Account ID",
        "Customer ID",
        "Seq No",
        "Seq No",
    ]
    # Build a regex pattern that looks for these keywords
    account_number_regex = (
        r"(?i)(" + "|".join(account_keywords) + r")\s*[:\-]?\s*(\d{8,20})"
    )
    account_match = re.search(account_number_regex, extracted_text)
    return (
        fuzzy_correct(account_match.group(2).strip(), ["account", "number", "seq"])
        if account_match
        else "Account number not found"
    )


# Extracting meter number using regex and fuzzy correction (updated regex for new format)
def extract_meter_number(extracted_text):
    meter_number_regex = (
        r"(?i)(?:meter\s*no\.?|meter\s*number|meter\s*id|mru\s*no)\s*[:\-]?\s*([\w\s]+)"
    )
    meter_match = re.search(meter_number_regex, extracted_text)
    return (
        fuzzy_correct(meter_match.group(1).strip(), ["meter", "number", "no", "id"])
        if meter_match
        else "Meter number not found"
    )


# Extracting ID number using regex and fuzzy correction
def extract_id_number(extracted_text):
    id_number_regex = (
        r"(?i)(?:id no\.?|id number|tin|seq no)\s*[:\-]?\s*(\d{3}[-\d]{3,})"
    )
    id_match = re.search(id_number_regex, extracted_text)
    return (
        fuzzy_correct(id_match.group(1).strip(), ["id", "number", "seq"])
        if id_match
        else "ID number not found"
    )


# Extracting area code
def extract_area_code(extracted_text):
    area_code_regex = r"(?i)(?:area code|region code)\s*[:\-]?\s*(\d{3,5})"
    area_code_match = re.search(area_code_regex, extracted_text)
    return (
        area_code_match.group(1).strip() if area_code_match else "Area code not found"
    )


# Extracting name from the image (customer, owner, etc.)
def extract_name(extracted_text):
    name_regex = (
        r"(?i)(?:name|account name|consumer name|owner name)\s*[:\-]?\s*([A-Za-z\s]+)"
    )
    name_match = re.search(name_regex, extracted_text)
    return name_match.group(1).strip() if name_match else "Name not found"


# Extracting address from the image
def extract_address(extracted_text):
    address_regex = r"(?i)(?:address)\s*[:\-]?\s*([^\n]+)"
    address_match = re.search(address_regex, extracted_text)
    return address_match.group(1).strip() if address_match else "Address not found"


# Extracting reading and due dates
def extract_dates(extracted_text):
    date_regex = r"(?i)(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
    dates = re.findall(date_regex, extracted_text)
    parsed_dates = []
    for date_str in dates:
        try:
            parsed_date = datetime.strptime(date_str, "%d/%m/%Y")
        except ValueError:
            try:
                parsed_date = datetime.strptime(date_str, "%m/%d/%Y")
            except ValueError:
                continue
        parsed_dates.append(parsed_date)

    if parsed_dates:
        parsed_dates.sort()
        reading_date = parsed_dates[0].strftime("%d/%m/%Y")
        due_date = parsed_dates[-1].strftime("%d/%m/%Y")
    else:
        reading_date = "Reading Date not found"
        due_date = "Due Date not found"

    return reading_date, due_date
