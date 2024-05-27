import re
import cv2
import pytesseract
from PIL import Image
from datetime import datetime
from dateutil.parser import parse
import tempfile
import uuid


def extract_driving_licence_number(input):
    regex = r"[A-Z]{2}[-\s]\d{13}|[A-Z]{2}[0-9]{2}\s[0-9]{11}"
    match = re.search(regex, input)
    driving_licence_number = match.group(0) if match else ""

    return driving_licence_number


def extract_all_dates(input):
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
    cleaned = []

    for name in match:
        split_name = name.split("\n")
        for chunk in split_name:
            cleaned.append(chunk)

    return cleaned


def extract_all_names(input):
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
