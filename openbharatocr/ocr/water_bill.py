import re
import pytesseract
from PIL import Image
from datetime import datetime


def extract_name(input):

    regex = r"(?:Name\s*[:\s]?\s*)(.*?)(?:\bConsumer\b|['/]|$)"
    match = re.search(regex, input, re.IGNORECASE)
    name = match.group(1).strip() if match else ""

    if name == "":
        regex = r"(?:Mr\sMrs\s*[:\s]?\s*)(.*?)(?:\bConsumer\b|['/]|$)"
        match = re.search(regex, input, re.IGNORECASE)
        name = match.group(1).strip() if match else ""

    return name


def extract_all_dates(input):

    regex = r"\b(\d{1,2}[/\-.](?:\d{2}|\d{4}|\w{3})[/\-.]\d{2,4})\b"
    dates = re.findall(regex, input)
    formatted_dates = []
    for date in dates:
        try:
            formatted_date = datetime.strptime(date, "%d/%m/%Y")
        except ValueError:
            try:
                formatted_date = datetime.strptime(date, "%d-%m-%Y")
            except ValueError:
                formatted_date = datetime.strptime(date, "%d-%b-%Y")
        formatted_dates.append(formatted_date)

    sorted_dates = sorted(formatted_dates)
    sorted_dates_str = [date.strftime("%d/%m/%Y") for date in sorted_dates]

    return sorted_dates_str


def extract_phone(input):

    regex = r"[6789]\d{9}"
    match = re.search(regex, input)
    phone = match.group(0) if match else ""

    return phone


def extract_address(input):

    regex = r"Address:\s*(.*?)(?=\b[A-Z][a-zA-Z\s]*:|\b[A-Z][a-zA-Z\s]*$)"
    match = re.search(regex, input, re.DOTALL | re.IGNORECASE)
    address = match.group(1).strip() if match else ""

    return address


def extract_meter_type(input):
    regex = r"Meter\s*Type\s*.DJB.Pvt.:\s*([A-Z]+)\s"
    match = re.search(regex, input, re.IGNORECASE)
    meter_type = match.group(1).strip() if match else ""

    return meter_type


def extract_bill_amount(input):

    regex = r"total\sdue\sdate\s*(\d+)\n"
    match = re.search(regex, input, re.IGNORECASE)
    bill_amount = match.group(1).strip() if match else ""

    if bill_amount == "":
        regex = r".Rs.\s*(\d+)\n"
        matches = re.findall(regex, input, re.IGNORECASE)
        bill_amount = matches[-1] if len(matches) > 0 else ""

    return bill_amount


def extract_water_bill_details(image_path):

    image = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(image)

    name = extract_name(extracted_text)

    dates = extract_all_dates(extracted_text)
    bill_date = dates[0] if len(dates) > 0 else ""
    due_date = dates[1] if len(dates) > 1 else ""

    address = extract_address(extracted_text)
    phone = extract_phone(extracted_text)

    meter_type = extract_meter_type(extracted_text)
    bill_amount = extract_bill_amount(extracted_text)

    return {
        "Name": name,
        "Phone Number": phone,
        "Bill Amount": bill_amount,
        "Bill Date": bill_date,
        "Due Date": due_date,
        "Address": address,
        "Meter Type": meter_type,
    }


def water_bill(image_path):
    return extract_water_bill_details(image_path)
