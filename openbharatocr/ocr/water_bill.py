import re
import pytesseract
from PIL import Image
from datetime import datetime
import cv2


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


def extract_name(input):
    """Extracts the customer name from the text using regular expressions.

    Args:
        input (str): The text to extract the name from.

    Returns:
        str: The extracted customer name (or empty string if not found).
    """
    regex = r"Name:\s*(.*?)(?:\.\s|(?=\n))"
    match = re.search(regex, input)
    name = match.group(1).strip() if match else ""

    if name == "":
        regex = r"(?:Mr\sMrs\s*[:\s]?\s*)(.*?)(?:\bConsumer\b|['/]|$)"
        match = re.search(regex, input, re.IGNORECASE)
        name = match.group(1).strip() if match else ""

    return name


def extract_bill_amount(input):
    """Extracts the bill amount from the text using a regular expression.

    Args:
        input (str): The text to extract the bill amount from.

    Returns:
        str: The extracted bill amount (or empty string if not found).
    """
    regex = r"Bill Amount \(Rs\.\)\s*:? (\d+)"
    match = re.search(regex, input, re.IGNORECASE)
    bill_amount = match.group(1).strip() if match else ""
    return bill_amount


def extract_meter_number(input):
    """
    Extracts the meter number from the water bill text using a regular expression.

    This function assumes the meter number is labeled as "Meter No." followed by
    an optional colon or period and expects digits (0-9) or "NA" to represent the number.

    Args:
        input (str): The text to extract the meter number from (typically the OCR output from a water bill image).

    Returns:
        str: The extracted meter number (or an empty string if not found).
    """
    regex = r"Meter No\.\s*:\s*(\d+|NA)"
    match = re.search(regex, input, re.IGNORECASE)
    meter_number = match.group(1).strip() if match else ""
    return meter_number


def extract_all_dates(input):
    """
    Extracts all dates in the format DD-MMM-YYYY from the given text using a regular expression.

    This function assumes dates are in the format DD-MMM-YYYY (e.g., 15-Jan-2024).
    It extracts all matching occurrences, parses them as datetime objects,
    sorts them chronologically, and returns them as formatted strings (DD-MM-YYYY).

    Args:
        input (str): The text to extract dates from.

    Returns:
        list: A list of extracted dates in YYYY-MM-DD format (sorted chronologically),
              or an empty list if no dates are found.
    """
    regex = r"\b(\d{1,2}-[A-Z]{3}-\d{4})\b"
    dates = re.findall(regex, input)
    formatted_dates = []
    for date in dates:
        try:
            formatted_date = datetime.strptime(date, "%d-%b-%Y")
            formatted_dates.append(formatted_date)
        except ValueError:
            continue
    sorted_dates = sorted(formatted_dates)
    sorted_dates_str = [date.strftime("%d-%m-%Y") for date in sorted_dates]
    return sorted_dates_str


def extract_phone(input):
    regex = r"Mobile No\.\s*:\s*(\d+)"
    match = re.search(regex, input)
    phone = match.group(1).strip() if match else ""
    return phone


def extract_address(input):
    regex = r"Address\s*:\s*(.*?)(?=\s*[A-Z][a-zA-Z\s]*:|$)"
    match = re.search(regex, input, re.DOTALL | re.IGNORECASE)
    address = match.group(1).strip() if match else ""
    return address


def extract_mr_code(input):
    regex = r"Zone/MR\s*Code:\s*([A-Z0-9/]+\s*[A-Z0-9/]*)"
    match = re.search(regex, input, re.IGNORECASE)
    mr_code = match.group(1).strip() if match else ""
    return mr_code


def extract_area_code(input):
    regex = r"Area Code\s*:\s*([\w/-]+)"
    match = re.search(regex, input, re.IGNORECASE)
    area_code = match.group(1).strip() if match else ""
    return area_code


def extract_bill_number(input):
    regex = r"Bill No\.\s*(?::\s*)?(\d+)"
    match = re.search(regex, input)
    bill_number = match.group(1).strip() if match else ""
    return bill_number


def extract_govt_body(input):
    regex = r"Delhi Jal Board"
    match = re.search(regex, input, re.IGNORECASE)
    govt_body = match.group(0).strip() if match else "Unknown"
    return govt_body


def extract_bill_date(input):
    regex = r"Bill Date\s*:? (\d{2}-[A-Z]{3}-\d{4})"
    match = re.search(regex, input, re.IGNORECASE)
    bill_date = match.group(1).strip() if match else ""
    return bill_date


def extract_bill_due_date(input):
    regex = r"Bill Due Date\s*:? (\d{2}-[A-Z]{3}-\d{4})"
    match = re.search(regex, input, re.IGNORECASE)
    bill_due_date = match.group(1).strip() if match else ""
    return bill_due_date


def extract_water_bill_details(image_path):
    """Extracts water bill details from an image using OCR and regular expressions.

    This function performs the following steps:

    1. Opens the image using Pillow.
    2. Extracts text using Tesseract (assuming the text is in a supported language).
    3. Extracts various water bill details using specific regular expressions.

    Args:
        image_path (str): The path to the water bill image.

    Returns:
        dict: A dictionary containing extracted water bill information
              (e.g., "Name", "Bill Amount", "Bill Date", etc.).
    """
    image = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(image)

    name = extract_name(extracted_text)
    dates = extract_all_dates(extracted_text)
    bill_date = extract_bill_date(extracted_text)
    due_date = extract_bill_due_date(extracted_text)
    address = extract_address(extracted_text)
    phone = extract_phone(extracted_text)
    bill_amount = extract_bill_amount(extracted_text)
    mr_code = extract_mr_code(extracted_text)
    area_code = extract_area_code(extracted_text)
    meter_number = extract_meter_number(extracted_text)
    bill_number = extract_bill_number(extracted_text)
    govt_body = extract_govt_body(extracted_text)

    return {
        "Name": name,
        "Phone Number": phone,
        "Bill Amount": bill_amount,
        "Bill Date": bill_date,
        "Due Date": due_date,
        "Address": address,
        "Zone/MR Code": mr_code,
        "Area Code": area_code,
        "Meter Number": meter_number,
        "Bill Number": bill_number,
        "Source/Govt Body Name": govt_body,
    }


def water_bill(image_path):
    """Extracts water bill details from an image.

    This function is a wrapper for `extract_water_bill_details`.

    Args:
        image_path (str): The path to the water bill image.

    Returns:
        dict: A dictionary containing extracted water bill information.
    """
    return extract_water_bill_details(image_path)
