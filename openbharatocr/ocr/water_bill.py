import cv2
import pytesseract
import re
import numpy as np
from datetime import datetime
from fuzzywuzzy import fuzz, process


def preprocess_for_bold_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    morph = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)
    sharpen = cv2.filter2D(morph, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
    _, binary = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def fuzzy_correct(text, possible_corrections):
    if not text:
        return text
    best_match = process.extractOne(text, possible_corrections)
    return best_match[0] if best_match and best_match[1] > 80 else text


def extract_amount(text):
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
        pattern = rf"{keyword}[\s:]*[\₹Rs\.]*\s*([\d,]+(?:\.\d{{1,2}})?)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            amount = match.group(1).replace(",", "")
            return float(amount)
    alt_pattern = r"(?<!\d)(\d{1,3}(?:[.,\s]?\d{3})*(?:[.,]\d{2})?)(?!\d)"
    amounts = re.findall(alt_pattern, text)
    values = [
        float(a.replace(",", "").replace(" ", "").replace(".", "", a.count(".") - 1))
        for a in amounts
        if re.match(r"\d", a)
    ]
    return max(values) if values else "Amount not found"


def extract_phone_numbers(text):
    pattern = (
        r"(?:phone|mobile|contact|tel|cell|no|number)[\s:]*[^\d]*(\+?\d[\d\s\-]{8,15})"
    )
    phones = re.findall(pattern, text, re.IGNORECASE)
    return list(set([re.sub(r"\D", "", phone) for phone in phones])) or [
        "Phone number not found"
    ]


def extract_bill_number(text):
    pattern = r"(?:bill\s*no|invoice\s*no|transaction\s*id|soa\s*#)[\s:]*#?\s*([A-Z0-9\-]{6,})"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else "Bill number not found"


def extract_account_number(text):
    pattern = r"(?:contract\s*account\s*no|account\s*(no|id|number)|customer\s*id|seq\s*no)[\s:]*([\dA-Z\-]{8,20})"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(2).strip() if match else "Account number not found"


def extract_meter_number(text):
    pattern = r"(?:meter\s*(no|number|id)|mru\s*no)[\s:]*([A-Z0-9\-]{6,20})"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(2).strip() if match else "Meter number not found"


def extract_id_number(text):
    pattern = r"(?:id\s*(no|number)|tin|seq\s*no)[\s:]*([A-Z0-9\-]{6,})"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(2).strip() if match else "ID number not found"


def extract_area_code(text):
    pattern = r"(?:area\s*code|region\s*code)[\s:]*([0-9]{3,6})"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else "Area code not found"


def extract_name(text):
    pattern = r"(?:name|account\s*name|consumer\s*name|owner\s*name)[\s:]*([A-Z\s]{3,})"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else "Name not found"


def extract_address(text):
    pattern = r"(?:address)[\s:]*([\w\s,.\-#/]+)"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else "Address not found"


def extract_dates(text):
    pattern = r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b"
    matches = re.findall(pattern, text)
    parsed = []
    for m in matches:
        for fmt in (
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%d-%m-%Y",
            "%m-%d-%Y",
            "%d/%m/%y",
            "%m/%d/%y",
        ):
            try:
                parsed.append(datetime.strptime(m, fmt))
                break
            except:
                continue
    if parsed:
        parsed.sort()
        return parsed[0].strftime("%d/%m/%Y"), parsed[-1].strftime("%d/%m/%Y")
    return "Reading Date not found", "Due Date not found"


def extract_water_bill_details(image_path):
    image = cv2.imread(image_path)
    processed = preprocess_for_bold_text(image)
    text = pytesseract.image_to_string(processed, lang="eng")
    details = {
        "Amount": extract_amount(text),
        "Phone Numbers": extract_phone_numbers(text),
        "Bill Number": extract_bill_number(text),
        "Account Number": extract_account_number(text),
        "Meter Number": extract_meter_number(text),
        "ID Number": extract_id_number(text),
        "Area Code": extract_area_code(text),
        "Name": extract_name(text),
        "Address": extract_address(text),
        "Reading Date": "",
        "Due Date": "",
    }
    reading_date, due_date = extract_dates(text)
    details["Reading Date"] = reading_date
    details["Due Date"] = due_date
    return details


def water_bill(image_path):
    return extract_water_bill_details(image_path)
