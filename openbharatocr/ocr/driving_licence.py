import re
import cv2
import pytesseract
from PIL import Image
import numpy as np
from dateutil.parser import parse


def preprocess_for_sketch(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    inverted_image = cv2.bitwise_not(binary)
    edges = cv2.Canny(inverted_image, 50, 150)
    sketch = cv2.addWeighted(inverted_image, 0.8, edges, 0.2, 0)

    kernel = np.ones((1, 1), np.uint8)
    processed_image = cv2.morphologyEx(sketch, cv2.MORPH_CLOSE, kernel, iterations=1)

    return processed_image


def extract_driving_licence_number(text):

    licence_keywords = ["dl no", "driving licence", "license number", "licence no", "dl number", "licence ", "licence:", "No.:"]
    regex = r"\b[A-Z]{2}[-\s]?\d{13}\b|\b[A-Z]{2}[0-9]{2}\s[0-9]{11}\b"
    lines = text.lower().splitlines()

    for line in lines:
        if any(keyword in line for keyword in licence_keywords):
            match = re.search(regex, line, re.IGNORECASE)
            if match:
                return match.group(0).strip()

    match = re.search(regex, text, re.IGNORECASE)
    return match.group(0).strip() if match else ""


def extract_dates(input_text):
    regex = r"\b\d{2}[/-]\d{2}[/-]\d{4}\b"
    return sorted(set(re.findall(regex, input_text)),
                  key=lambda x: int(re.split(r"[-/]", x)[-1]))


def parse_date_safe(date_str):
    try:
        return parse(date_str, dayfirst=True)
    except Exception:
        return None


def extract_dob(input_text):
    dates = extract_dates(input_text)
    return dates[0] if dates else ""

def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

def extract_dates(text):
    regex = r"\b\d{2}[-/.]\d{2}[-/.]\d{4}\b"
    return re.findall(regex, text)

def extract_issue_dates(input_text):
    issue_keywords = ["issue date", "issued on", "Org. Isuue Dt", "Date of Issue", "valid from"]
    lines = input_text.lower().splitlines()

    for line in lines:
        if any(keyword in line for keyword in issue_keywords):
            dates = extract_dates(line)
            if dates:
                return [dates[0]] 
    return []

def extract_valid_till_dates(input_text):
    valid_keywords = ["valid till", "valid upto", "valid up to", "Validity", "valid until", "valid to"]
    lines = input_text.lower().splitlines()

    for line in lines:
        if any(keyword in line for keyword in valid_keywords):
            dates = extract_dates(line)
            if dates:
                return [dates[0]]
    return []

def extract_dates_from_image(image_path):
    text = extract_text_from_image(image_path)
    issue_dates = extract_issue_dates(text)
    valid_till_dates = extract_valid_till_dates(text)
    return {
        "issue_dates": issue_dates,
        "valid_till_dates": valid_till_dates
    }



def extract_all_names(input_text):
    name_keywords = ["Name", "NAME", "Name of Holder", "NAME OF HOLDER", "Holder", "HOLDER"]
    stopwords = {"INDIA", "OF", "UNION", "PRADESH", "TRANSPORT", "DRIVING", "LICENCE", "FORM"}

    lines = input_text.splitlines()
    for line in lines:
        for keyword in name_keywords:
            if keyword in line:
                name = line.split(keyword)[-1].strip()
                name = re.sub(r"[^A-Za-z\s]", "", name)
                if not any(stopword in name.upper() for stopword in stopwords):
                    return name
    return ""


def extract_address_regex(input_text):
    regex_list = [
        r"Address\s*:\s*\n*(.*?)(?=\n\n|\Z)",
        r"Add\b\s*(.*?)(?=\bPIN|$)",
        r"ADDRESS\s*[:-]?\s*((?:.|\n)*?)\b(?:PIN|Valid|Till|$)",
    ]
    for regex in regex_list:
        matches = re.findall(regex, input_text, re.DOTALL)
        for match in matches:
            if match.strip():
                return re.sub(r"\s+", " ", match.strip())
    return ""


def extract_address(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)

    if "Add" not in text and "ADD" not in text:
        return ""

    rgb = image.convert("RGB")
    gray_image = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2GRAY)

    config = r"--oem 3 --psm 6"
    boxes_data = pytesseract.image_to_data(gray_image, config=config).splitlines()

    for box in boxes_data[1:]:
        parts = box.split()
        if len(parts) == 12 and ("Add" in parts[11] or "ADD" in parts[11]):
            left = int(parts[6])
            top = int(parts[7])
            h, w = gray_image.shape
            roi = gray_image[top:top + int(0.18 * h), left:left + int(0.4 * w)]
            address = pytesseract.image_to_string(roi, config=config)
            return " ".join(address.split()[1:])
    return extract_address_regex(text)


def extract_auth_allowed(input_text):
    auth_keywords = ["MCWG", "MCWOG", "LMV", "LMV-NT", "HMV", "TRANS", "MGV", "HTV", "TRAILER", "INVCRG", "TVPM"]
    return [auth for auth in auth_keywords if auth in input_text]


def extract_dl_info(image_path, sketch=False):
    if sketch:
        processed = preprocess_for_sketch(image_path)
        image_pil = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB))
    else:
        image_pil = Image.open(image_path)

    text = pytesseract.image_to_string(image_pil)

    return {
        "Driving Licence Number": extract_driving_licence_number(text),
        "Date of Birth": extract_dob(text),
        "Date of Issue": extract_issue_dates(text),
        "Valid Till": extract_valid_till_dates(text),
        "Name": extract_all_names(text),
        "Address": extract_address(image_path),
        "Authorization Allowed": extract_auth_allowed(text),
    }


def compare_results(r1, r2):
    final = {}
    for key in r1:
        val1, val2 = r1[key], r2[key]
        if not val1 and val2:
            final[key] = val2
        elif val1 and not val2:
            final[key] = val1
        elif isinstance(val1, list) and isinstance(val2, list):
            final[key] = val1 if len(val1) >= len(val2) else val2
        elif isinstance(val1, str) and isinstance(val2, str):
            final[key] = val1 if len(val1) >= len(val2) else val2
        else:
            final[key] = val1 or val2 or None
    return final


def extract_driving_license_data(image_path):
    normal = extract_dl_info(image_path, sketch=False)
    sketch = extract_dl_info(image_path, sketch=True)
    return compare_results(normal, sketch)


def driving_licence(image_path):
    return extract_driving_license_data(image_path)


if __name__ == "__main__":
    image_path = "/home/rishabh/openbharatocr/openbharatocr/ocr/DL_sample/1.jpeg"
    result = driving_licence(image_path)
    print(result)