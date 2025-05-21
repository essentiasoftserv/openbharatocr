import re
import imghdr
import pytesseract
from PIL import Image
import cv2
import numpy as np


def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(src=image, ddepth=-1, kernel=kernel)


def binarize_otsu(gray_image):
    _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.bitwise_not(binary)


def enhanced_preprocess(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpened = sharpen_image(gray)
    binary = binarize_otsu(sharpened)
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return cleaned


def get_text_from_image(image):
    custom_config = r"--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789./- "
    return pytesseract.image_to_string(image, config=custom_config)


def get_text_with_confidence(image):
    custom_config = r"--oem 3 --psm 6"
    data = pytesseract.image_to_data(
        image, config=custom_config, output_type=pytesseract.Output.DICT
    )
    texts = [
        data["text"][i] for i in range(len(data["text"])) if int(data["conf"][i]) > 60
    ]
    return " ".join(texts)


def get_text_from_image(image):
    config = r"--oem 3 --psm 4 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/:- "
    return pytesseract.image_to_string(image, config=config)


def get_text_with_confidence(image):
    config = r"--oem 3 --psm 4"
    data = pytesseract.image_to_data(
        image, config=config, output_type=pytesseract.Output.DICT
    )
    texts = [
        data["text"][i] for i in range(len(data["text"])) if int(data["conf"][i]) > 60
    ]
    return " ".join(texts)


def extract_pan(input_text):
    regex = r"\b([A-Z]{5}[0-9]{4}[A-Z])\b"
    match = re.search(regex, input_text)
    return match.group(1) if match else ""


def extract_dob(text):
    dob_regex = re.compile(
        r"\b(0[1-9]|[12][0-9]|3[01])[/\-.](0[1-9]|1[0-2])[/\-.](19|20)\d{2}\b"
    )
    match = dob_regex.search(text)
    if match:
        return match.group(0)
    else:
        return ""


def clean_input(lines):
    stopwords = [
        "INDIA",
        "OF",
        "TAX",
        "GOVT",
        "DEPARTMENT",
        "INCOME",
        "CARD",
        "PERMANENT",
        "ACCOUNT",
    ]
    cleaned_names = []
    for name in lines:
        name = name.strip()
        if not any(word in name.upper() for word in stopwords) and len(name) > 3:
            cleaned_names.append(name)
    return cleaned_names


def extract_full_name(text):
    lines = text.splitlines()
    full_name = ""
    full_name_pattern = re.compile(r"(?:Name|Na?me|Ta?Name|Na?ta/?Name)", re.IGNORECASE)

    for i, line in enumerate(lines):
        if full_name_pattern.search(line):
            for j in range(i + 1, min(i + 4, len(lines))):
                candidate = lines[j].strip()
                if candidate and re.match(r"^[A-Z\s]+$", candidate):
                    full_name = candidate.strip()
                    return full_name
    return full_name


def extract_parent_name(text):
    lines = text.splitlines()
    parent_name = ""
    parent_name_pattern = re.compile(r"(Father|Parents|Dad|FathersName)", re.IGNORECASE)

    for i, line in enumerate(lines):
        if parent_name_pattern.search(line):
            for j in range(i + 1, min(i + 4, len(lines))):
                candidate = lines[j].strip()
                if candidate and re.match(r"^[A-Z\s]+$", candidate):
                    parent_name = candidate.strip()
                    return parent_name
    return parent_name


def extract_pan_details(image_path):
    image = Image.open(image_path)
    format = imghdr.what(image_path)
    if format != "jpeg":
        image.save("temp_image.jpg", "JPEG")
        image = Image.open("temp_image.jpg")

    extracted_text = get_text_from_image(image)

    return {
        "Full Name": extract_full_name(extracted_text),
        "Parent's Name": extract_parent_name(extracted_text),
        "Date of Birth": extract_dob(extracted_text),
        "PAN Number": extract_pan(extracted_text),
    }


def extract_pan_details_version2(image_path):
    preprocessed_image = enhanced_preprocess(image_path)
    extracted_text = get_text_with_confidence(preprocessed_image)

    return {
        "Full Name": extract_full_name(extracted_text),
        "Parent's Name": extract_parent_name(extracted_text),
        "Date of Birth": extract_dob(extracted_text),
        "PAN Number": extract_pan(extracted_text),
    }


def pan(image_path):
    result = extract_pan_details(image_path)

    if not all(
        [
            result["Full Name"],
            result["Parent's Name"],
            result["Date of Birth"],
            result["PAN Number"],
        ]
    ):
        result_v2 = extract_pan_details_version2(image_path)
        result["Full Name"] = result["Full Name"] or result_v2["Full Name"]
        result["Parent's Name"] = result["Parent's Name"] or result_v2["Parent's Name"]
        result["Date of Birth"] = result["Date of Birth"] or result_v2["Date of Birth"]
        result["PAN Number"] = result["PAN Number"] or result_v2["PAN Number"]

    return result


if __name__ == "__main__":
    image_path = "/home/rishabh/openbharatocr/openbharatocr/ocr/PAN1.jpeg"
    pan_details = pan(image_path)
    print(pan_details)
