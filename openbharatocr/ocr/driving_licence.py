import re
import cv2
import pytesseract
from PIL import Image
import numpy as np
from dateutil.parser import parse


def preprocess_for_sketch(image_path):
    """
    Preprocesses an image to convert it into a black and white sketch-like look
    with black text on a white background for improved text extraction.

    Args:
        image_path (str): The path to the image.

    Returns:
        numpy.ndarray: The preprocessed image with black text on a white background.
    """
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to smooth the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use adaptive thresholding to highlight text regions with a clear white background
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )

    # Use bitwise inversion to make text black and background white
    inverted_image = cv2.bitwise_not(binary)

    # Detect edges using Canny edge detector
    edges = cv2.Canny(inverted_image, threshold1=50, threshold2=150)

    # Combine edges with the original grayscale to enhance text
    sketch = cv2.addWeighted(inverted_image, 0.8, edges, 0.2, 0)

    # Apply morphological operations to enhance text regions further
    kernel = np.ones((1, 1), np.uint8)
    processed_image = cv2.morphologyEx(sketch, cv2.MORPH_CLOSE, kernel, iterations=1)

    return processed_image


def extract_driving_licence_number(input_text):
    regex = r"[A-Z]{2}[-\s]\d{13}|[A-Z]{2}[0-9]{2}\s[0-9]{11}"
    match = re.search(regex, input_text)
    driving_licence_number = match.group(0) if match else ""

    return driving_licence_number


def extract_all_dates(input_text):
    regex = r"\b\d{2}[/-]\d{2}[/-]\d{4}\b"
    dates = re.findall(regex, input_text)
    dates = sorted(dates, key=lambda x: int(re.split(r"[-/]", x)[-1]))

    seen = set()
    unique_dates = []

    for date in dates:
        if date not in seen:
            seen.add(date)
            unique_dates.append(date)

    dob = unique_dates[0] if unique_dates else ""
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


def extract_all_names(input_text):
    """
    Extracts the name from the input text using keywords and patterns,
    and filters out any names that contain stopwords.

    Args:
        input_text (str): The OCR extracted text.

    Returns:
        str: The extracted name or an empty string if not found.
    """
    # Common keywords that precede the name in the text
    name_keywords = [
        "Name",
        "NAME",
        "Name:",
        "NAME:",
        "Holder",
        "HOLDER",
        "Name of Holder",
        "NAME OF HOLDER",
    ]

    # Stopwords that are likely to indicate non-name parts
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

    # Split the input text into lines for easier processing
    lines = input_text.splitlines()
    name_line = ""

    # Iterate over each line to find a potential name using keywords
    for line in lines:
        for keyword in name_keywords:
            if keyword in line:
                # Assuming name follows the keyword
                name_line = line.split(keyword)[
                    -1
                ].strip()  # Get the text after the keyword
                if name_line:
                    break
        if name_line:
            break

    # If a potential name line is found, further process to clean and format it
    if name_line:
        # Removing any trailing numbers or non-alphabet characters
        name_line = re.sub(r"[^A-Za-z\s]", "", name_line).strip()

        # Check if the extracted name contains any stopwords
        if any(stopword in name_line.upper() for stopword in stopwords):
            return ""  # Return an empty string if any stopword is found in the name

        return name_line

    # Return empty string if no name found
    return ""


def extract_address_regex(input_text):
    regex_list = [
        r"Address\s*:\s*\n*(.*?)(?=\n\n|\Z)",
        r"Add\b\s*(.*?)(?=\bPIN|$)",
        r"Address\b\s*(.*?)(?=\bPIN|$)",
        r"Address\s*:\s*((?:(?!(?:Valid Till)).*(?:\n|$))+)",
        r"ADDRESS - (.+?)(?= (?:\b\d{6}\b|$))",
        r"Address\s*:\s*((?:(?!(?:Valid Till)).*(?:\n|$))+)",
    ]
    regex = "|".join(regex_list)

    matches = re.findall(regex, input_text, re.DOTALL)

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
    image = np.array(rgb)

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

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


def extract_auth_allowed(input_text):
    authorizations = [
        "MCWG",
        "MCWOG",
        "LMV",
        "LMV-NT",
        "HMV",
        "HMV-NT",
        "TRANS",
        "TRANS-NT",
        "MGV",
        "HPMV",
        "HTV",
        "TRAILER",
    ]
    allowed = []
    for auth in authorizations:
        if auth in input_text:
            allowed.append(auth)

    return allowed


def extract_dl_info(image_path, sketch=False):
    if sketch:
        image = preprocess_for_sketch(image_path)
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
    else:
        image_pil = Image.open(image_path)

    input_text = pytesseract.image_to_string(image_pil)

    dob, doi, validity = extract_all_dates(input_text)
    names = extract_all_names(input_text)
    address = extract_address(image_path)
    allowed = extract_auth_allowed(input_text)
    licence_number = extract_driving_licence_number(input_text)

    return {
        "Driving Licence Number": licence_number,
        "Date of Birth": dob,
        "Date of Issue": doi,
        "Valid Till": validity,
        "Name": names,
        "Address": address,
        "Authorization Allowed": allowed,
    }


def compare_results(result1, result2):
    final_result = {}

    for key in result1:
        value1 = result1[key]
        value2 = result2[key]

        if not value1 and value2:
            final_result[key] = value2
        elif value1 and not value2:
            final_result[key] = value1
        elif value1 and value2:
            # Compare the length of the results if both are non-empty
            final_result[key] = value1 if len(value1) >= len(value2) else value2
        else:
            final_result[key] = None

    return final_result


def extract_driving_license_data(image_path):
    # Extracting data without preprocessing
    normal_result = extract_dl_info(image_path, sketch=False)

    # Extracting data with sketch-like preprocessing
    sketch_result = extract_dl_info(image_path, sketch=True)

    # Comparing results to get the most accurate information
    final_result = compare_results(normal_result, sketch_result)

    return final_result


def driving_licence(image_path):
    return extract_driving_license_data(image_path)
