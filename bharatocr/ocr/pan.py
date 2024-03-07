import re
import pytesseract
from PIL import Image
from datetime import datetime


def extract_names(input):

    full_name, fathers_name = "", ""

    if "Name" in input:
        name_regex = r"Name[\s:]+([A-Za-z\s]+)(?:\n|$)"
        fathers_name_regex = r"Father's\s*Name.*?\n([^0-9\n]+)"

        name_match = re.search(name_regex, input, re.IGNORECASE)
        fathers_name_match = re.search(fathers_name_regex, input, re.IGNORECASE)

        full_name = name_match.group(1).strip() if name_match else ""
        fathers_name = fathers_name_match.group(1).strip() if fathers_name_match else ""
    else:
        names = []
        lines = input.split("\n")

        for line in lines:
            if "INCOME TAX DEPARTMENT" not in line:
                match = re.search(r"^[A-Z\s]+$", line)
                if match:
                    name = match.group().strip()
                    names.append(name)

        # print(names)
        if len(names) >= 1:
            full_name = names[0]
        if len(names) >= 2:
            fathers_name = names[1]

    return full_name, fathers_name


def extract_pan(input):

    regex = r"[A-Z]{5}[0-9]{4}[A-Z]"
    match = re.search(regex, input)
    pan_number = match.group(0) if match else ""

    return pan_number


def extract_dob(input):

    regex = r"\b(\d{2}/\d{2}/\d{4})\b"
    match = re.search(regex, input)
    dob = match.group(0) if match else ""

    return dob


def extract_pan_details(image_path):

    image = Image.open(image_path)

    extracted_text = pytesseract.image_to_string(image)
    # print(extracted_text)

    full_name, fathers_name = extract_names(extracted_text)
    dob = extract_dob(extracted_text)
    pan_number = extract_pan(extracted_text)

    return {
        "Full Name": full_name,
        "Father's Name": fathers_name,
        "Date of Birth": dob,
        "PAN Number": pan_number,
    }


image_path = "test_image2.jpg"


def pan(image_path):
    return extract_pan_details(image_path)


# pan_details = extract_pan_details(image_path)
