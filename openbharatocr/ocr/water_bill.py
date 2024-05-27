import re
import pytesseract
from PIL import Image
from datetime import datetime


def preprocess_for_bold_text(image):
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
    regex = r"Name:\s*(.*?)(?:\.\s|(?=\n))"
    match = re.search(regex, input)
    name = match.group(1).strip() if match else ""

    if name == "":
        regex = r"(?:Mr\sMrs\s*[:\s]?\s*)(.*?)(?:\bConsumer\b|['/]|$)"
        match = re.search(regex, input, re.IGNORECASE)
        name = match.group(1).strip() if match else ""

    return name


def extract_bill_amount(input):
    regex = r"Bill Amount \(Rs\.\)\s*:? (\d+)"
    match = re.search(regex, input, re.IGNORECASE)
    bill_amount = match.group(1).strip() if match else ""
    return bill_amount


def extract_meter_number(input):
    regex = r"Meter No\.\s*:\s*(\d+|NA)"
    match = re.search(regex, input, re.IGNORECASE)
    meter_number = match.group(1).strip() if match else ""
    return meter_number


def extract_all_dates(input):
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
    return extract_water_bill_details(image_path)
