import re
import pytesseract
import imghdr
from PIL import Image


def clean_input(match):
    cleaned = []

    for name in match:
        split_name = name.split("\n")
        for chunk in split_name:
            cleaned.append(chunk)

    return cleaned


def extract_all_names(input):
    regex = r"\n[A-Z\s]+\b"
    match = re.findall(regex, input)

    names = []
    cleaned = clean_input(match)

    stopwords = ["INDIA", "OF", "TAX", "GOVT", "DEPARTMENT", "INCOME"]

    names = [
        name.strip()
        for name in cleaned
        if not any(word in name for word in stopwords) and len(name.strip()) > 3
    ]

    return names


def extract_pan(input):
    regex = r"[A-Z]{5}[0-9]{4}[A-Z]"
    match = re.search(regex, input)
    pan_number = match.group(0) if match else ""

    return pan_number


def extract_dob(input):
    """
    # This regex pattern matches dates in dd/mm/yyyy,
    dd-mm-yyyy, dd.mm.yyyy, and dd/mm/yy formats.
    # It accommodates dates separated by slashes (/),
    hyphens (-), or dots (.), and years in both four-digit
    and two-digit formats.
    """

    regex = r"\b(\d{2}[/\-.]\d{2}[/\-.](?:\d{4}|\d{2}))\b"
    match = re.search(regex, input)
    dob = match.group(0) if match else ""

    return dob


def extract_pan_details(image_path):
    image = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(image)

    format = imghdr.what(image_path)
    if format != "jpeg":
        image.save("image.jpg", "JPEG")

        converted_image = Image.open("image.jpg")
        converted_image_text = pytesseract.image_to_string(converted_image)

        extracted_text += converted_image_text

    names = extract_all_names(extracted_text)
    full_name = names[0] if len(names) > 0 else ""
    parents_name = names[1] if len(names) > 1 else ""
    dob = extract_dob(extracted_text)
    pan_number = extract_pan(extracted_text)

    return {
        "Full Name": full_name,
        "Parent's Name": parents_name,
        "Date of Birth": dob,
        "PAN Number": pan_number,
    }


def pan(image_path):
    return extract_pan_details(image_path)
