import re
import cv2
import pytesseract
from PIL import Image
from datetime import datetime
from dateutil.parser import parse
import tempfile
import uuid


def extract_driving_licence_number(input):
    """
    Extracts the driving license number from the provided text using regular expressions.

    This function uses a regular expression pattern to match two common formats of driving license numbers in India:
        - `[A-Z]{2}[-\s]\d{13}` (e.g., KA-01 123456789012)
        - `[A-Z]{2}[0-9]{2}\s[0-9]{11}` (e.g., KA02 12345678901)

    The function searches for these patterns in the input text and returns the matched driving license number if found, otherwise an empty string.

    Args:
        input: The text extracted from the driving license image.

    Returns:
        str: The extracted driving license number (if found), otherwise an empty string.
    """
    regex = r"[A-Z]{2}[-\s]\d{13}|[A-Z]{2}[0-9]{2}\s[0-9]{11}"
    match = re.search(regex, input)
    driving_licence_number = match.group(0) if match else ""

    return driving_licence_number


def extract_all_dates(input):
    """
    Extracts all dates from the provided text using regular expressions and processes them.

    This function uses a regular expression `\b\d{2}[/-]\d{2}[/-]\d{4}\b` to match dates in the format DD-MM-YYYY. It then performs the following:
        1. Sorts the extracted dates in ascending order based on the year.
        2. Removes duplicates using a set to ensure unique dates.
        3. Identifies the Date of Birth (DOB) as the earliest date.
        4. Extracts the Date of Issue (DOI) as the second date if it exists.
        5. Analyzes the remaining dates (if any) based on the year difference with DOI:
            - Dates within a specific validity duration (default 8 years) after DOI are considered additional DOI entries.
            - Dates beyond the validity duration are considered validity period dates.

    Args:
        input: The text extracted from the driving license image.

    Returns:
        tuple: A tuple containing three elements:
            - dob (str): The extracted Date of Birth.
            - doi (list): A list of extracted Dates of Issue (can be empty).
            - validity (list): A list of extracted validity period dates (can be empty).
    """
    regex = r"\b\d{2}[/-]\d{2}[/-]\d{4}\b"
    dates = re.findall(regex, input)
    dates = sorted(dates, key=lambda x: int(re.split(r"[-/]", x)[-1]))

    seen = set()
    unique_dates = []

    for date in dates:
        if date not in seen:
            seen.add(date)
            unique_dates.append(date)

    dob = unique_dates[0]
    doi = [unique_dates[1]] if len(unique_dates) > 1 else []
    validity = []

    year = int(re.split(r"[-/]", unique_dates[1])[-1]) if len(unique_dates) > 1 else -1
    validity_duration = 8

    i = 2
    while i < len(unique_dates):
        curr_year = int(re.split(r"[-/]", unique_dates[i])[-1])
        if curr_year - year <= validity_duration:
            doi.append(unique_dates[i])
        else:
            break
        i += 1

    while i < len(unique_dates):
        validity.append(unique_dates[i])
        i += 1

    return dob, doi, validity


def clean_input(match):
    """
    Cleans a list of potential names extracted from the text.

    This function iterates through the provided list of names (potentially containing newlines) and performs the following:
        1. Splits each name entry by newline characters.
        2. Creates a new list to store cleaned names.
        3. Adds each individual name chunk (after removing leading/trailing whitespaces) to the cleaned list.

    Args:
        match: A list of potential names (strings) extracted using regular expressions.

    Returns:
        list: A list containing the cleaned names (without newlines and extra spaces).
    """
    cleaned = []

    for name in match:
        split_name = name.split("\n")
        for chunk in split_name:
            cleaned.append(chunk)

    return cleaned


def extract_all_names(input):
    """
    Extracts all names from the provided text using regular expressions and applies stopword removal.

    This function performs the following steps:
        1. Uses a regular expression `\b[A-Z\s]+\b` to match sequences of uppercase letters and spaces (potential names).
        2. Applies `clean_input` to remove newlines and extra spaces from the extracted names.
        3. Defines a list of stopwords commonly found in driving licenses (e.g., "INDIA", "LICENCE").
        4. Filters the cleaned names, removing those containing stopwords or having a length less than 3 characters.
        5. Returns the list of extracted full names and surnames (assuming the first name is the full name).

    Args:
        input: The text extracted from the driving license image.

    Returns:
        list: A list containing the extracted full names and surnames (if found), otherwise an empty list.
    """
    regex = r"\b[A-Z\s]+\b"
    match = re.findall(regex, input)

    names = []
    cleaned = clean_input(match)

    stopwords = [
        "INDIA",
        "OF",
        "UNION",
        "PRADESH",
        "TRANSPORT",
        "DRIVING",
        "LICENCE",
        "FORM",
        "MCWG",
        "LMV",
        "TRANS",
        "ANDHRA",
        "UTTAR",
        "MAHARASHTRA",
        "GUJARAT",
        "TAMIL",
        "NADU",
        "WEST",
        "BENGAL",
        "KERELA",
        "KARNATAKA",
        "DRIVE",
        "AUTHORIZATION",
        "FOLLOWING",
        "CLASS",
        "DOI",
    ]

    names = [
        name.strip()
        for name in cleaned
        if not any(word in name for word in stopwords) and len(name.strip()) > 3
    ]

    return names


def extract_address_regex(input):
    """
    Extracts the address from driving license text using regular expressions targeting specific patterns.

    This function attempts to match various address formats commonly found in driving licenses using a combined regular expression. It searches for patterns like "Address:", "Add", or "ADDRESS -" followed by the actual address details.

    Args:
        input: The text extracted from the driving license image.

    Returns:
        str: The extracted address if found, otherwise an empty string.
    """
    regex_list = [
        r"Address\s*:\s*\n*(.*?)(?=\n\n|\Z)",
        r"Add\b\s*(.*?)(?=\bPIN|$)",
        r"Address\b\s*(.*?)(?=\bPIN|$)",
        r"Address\s*:\s*((?:(?!(?:Valid Till)).*(?:\n|$))+)",
        r"ADDRESS - (.+?)(?= (?:\b\d{6}\b|$))",
        r"Address\s*:\s*((?:(?!(?:Valid Till)).*(?:\n|$))+)",
    ]
    regex = "|".join(regex_list)

    matches = re.findall(regex, input, re.DOTALL)

    address = ""
    found = 0
    for match in matches:
        for group in match:
            if group:
                address = group.strip()
                found = 1
                break
        if found:
            break

    return address


def extract_address(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)

    if "Add" not in text and "ADD" not in text:
        return ""
    rgb = image.convert("RGB")
    with tempfile.TemporaryDirectory() as tempdir:
        tempfile_path = f"{tempdir}/{str(uuid.uuid4())}.jpg"
        rgb.save(tempfile_path)

        image = cv2.imread(tempfile_path)

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        config = r"--oem 3 --psm 6"
        boxes_data = pytesseract.image_to_data(gray_image, config=config)

        boxes = boxes_data.splitlines()
        boxes = [b.split() for b in boxes]

        left, top = -1, -1
        for box in boxes[1:]:
            if len(box) == 12:
                if "Add" in box[11] or "ADD" in box[11]:
                    left = int(box[6])
                    top = int(box[7])

        if left == -1:
            return extract_address_regex(text)

        h, w = gray_image.shape

        right = min(left + int(0.4 * w), w)
        bottom = min(top + int(0.18 * h), h)

        roi = gray_image[top:bottom, left:right]
        address = pytesseract.image_to_string(roi, config=config)

        split_address = address.split(" ")
        split_address.remove(split_address[0])

        address = " ".join(split_address)

        return address


def extract_auth_allowed(input):
    auth_types, auth_allowed = [
        "MCWG",
        "M.CYL.",
        "LMV-NT",
        "LMV",
        "TRANS",
        "INVCRG",
    ], []

    for auth in auth_types:
        if auth in input:
            auth_allowed.append(auth)

    return auth_allowed


def expired(input):
    try:
        date = parse(input, dayfirst=True)
        curr = datetime.now()
        diff = date - curr
        if diff.days >= 0:
            return False
        return True
    except:
        return False


def extract_driving_license_details(image_path):
    """
    Extracts various details from a driving license image using OCR and text processing.

    This function performs the following steps:
        1. Opens the image using Pillow (PIL Fork).
        2. Extracts text from the image using Tesseract OCR.
        3. Extracts names (full name and surname) using `extract_all_names`.
        4. Extracts dates (DoB, DoI, validity) using `extract_all_dates`.
        5. Extracts driving license number using `extract_driving_licence_number`.
        6. Extracts authorized vehicle categories using `extract_auth_allowed`.
        7. Extracts address using a combination of text processing and image analysis:
            - If "Add" or "ADD" is not found in the text, falls back to `extract_address_regex`.
            - Otherwise, attempts to locate the address section in the image based on bounding box information and re-extracts the address using Tesseract with a focused region of interest (ROI).
        8. Checks the validity of the last date in `validity` and sets a remark if expired.

    Args:
        image_path: The path to the driving license image file.

    Returns:
        dict: A dictionary containing the extracted driving license details (full name, DoB, DoI, validity, license number, authorizations, address, and remark).
    """
    image = Image.open(image_path)

    extracted_text = pytesseract.image_to_string(image)

    names = extract_all_names(extracted_text)
    full_name = names[0] if len(names) > 0 else ""
    swd_name = names[1] if len(names) > 1 else ""

    dob, issue_date, validity = extract_all_dates(extracted_text)

    driving_licence_number = extract_driving_licence_number(extracted_text)

    authorizations = extract_auth_allowed(extracted_text)

    address = extract_address(image_path)

    remark = ""
    if len(validity) and expired(validity[-1]):
        remark = "The driving licence has been expired."

    return {
        "Full Name": full_name,
        "S/W/D": swd_name,
        "Date of Birth": dob,
        "Issue Date": issue_date,
        "Validity": validity,
        "Driving Licence Number": driving_licence_number,
        "Authorizations": authorizations,
        "Address": address,
        "Remark": remark,
    }


def driving_licence(image_path):
    return extract_driving_license_details(image_path)
