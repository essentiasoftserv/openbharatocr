import re
import pytesseract
from PIL import Image


def extract_names(input):
    regex_swd = r"dual\sOwner\)?\s*:?\s*([A-Z.]+\s[A-Z.]+\s[A-Z.]+)"
    match = re.search(regex_swd, input, re.IGNORECASE)
    swd = match.group(1) if match else ""

    regex_name = r"NAME\s*:?\s*([A-Z]+\s[A-Z]+)"
    match_name = re.search(regex_name, input, re.IGNORECASE)
    name = match_name.group(1) if match_name else ""

    if swd == "":
        regex_swd = r"NAME\s*:?\s*([A-Z]+)\s"
        match_swd = re.findall(regex_swd, input, re.IGNORECASE)
        swd = match_swd[1] if len(match_swd) > 1 else ""

    if swd == "":
        regex_swd = r"OF\s*:?\s*[S/O01]*\s*([A-Z]+\s[A-Z]+)"
        match = re.search(regex_swd, input, re.IGNORECASE)
        swd = match.group(1) if match else ""

    return name, swd


def extract_reg_number(input):
    regex = r"(?=.*\d)[A-Z0-9]{10}"
    match = re.search(regex, input)
    reg_number = match.group(0) if match else ""

    return reg_number


def extract_chasis(input):
    regex = r"[A-Z0-9]{17,18}"
    match = re.search(regex, input)
    chasis = match.group(0) if match else ""

    return chasis


def extract_fuel_type(input):
    regex = r"Fuel(?:\s+Type)?\s*[\s:\.]\s*([A-Z/]+)\s"
    match = re.search(regex, input, re.IGNORECASE)
    fuel_type = match.group(1) if match else ""
    return fuel_type


def extract_vehicle_class(input):
    regex = r"(?:Veh.c.e\sClass|Veh\sCl)\s*[\s:]\s*([A-Z0-9/()-]+)\s([A-Z0-9/()-]+)\s"
    match = re.search(regex, input, re.IGNORECASE)
    vehicle_class = match.group(1) if match else ""
    return vehicle_class


def extract_manufacturer(input):
    regex = r"MFR\s*:\s*([A-Z\s]+)\n"
    match = re.search(regex, input, re.IGNORECASE)
    manufacturer = match.group(1) if match else ""
    return manufacturer


def extract_tax_info(input):
    regex = r"Tax\sUp\s{0,1}to\s*:\s*([A-Z]+)\s"
    match = re.search(regex, input, re.IGNORECASE)
    tax_up_to = match.group(1) if match else ""
    return tax_up_to


def extract_model(input):
    regex = r"Mode.\s*[\s:]\s*([A-Z0-9/+()-.]+(?:\s+[^\w\n]*[A-Z0-9/+()-.]+){0,3})\s"
    match = re.search(regex, input, re.IGNORECASE)
    model = match.group(1) if match else ""
    return model


def extract_all_dates(input_text):
    regex = r"\b(\d{1,2}[/\-.](?:\d{2}|\d{4}|\w{3})[/\-.]\d{2,4})\b"
    dates = re.findall(regex, input_text)
    sorted_dates = sorted(
        dates, key=lambda date: int(date.split("/")[-1].split("-")[-1])
    )

    return sorted_dates


def extract_address(input):
    regex = r"Address:?\s*((?:.|\n)*?\d{6})"
    match = re.search(regex, input, re.IGNORECASE)
    address = match.group(1) if match else ""

    return address


def extract_vehicle_registration_details(image_path):
    image = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(image)

    names = extract_names(extracted_text)
    name, swd = names[0], names[1]

    dates = extract_all_dates(extracted_text)
    expiry_date = dates[-1] if len(dates) > 0 else ""
    registration_date = dates[0] if len(dates) > 1 else ""

    chasis_number = extract_chasis(extracted_text)
    reg_number = extract_reg_number(extracted_text)

    address = extract_address(extracted_text)

    fuel_type = extract_fuel_type(extracted_text)
    vehicle_class = extract_vehicle_class(extracted_text)
    model = extract_model(extracted_text)

    manufacturer = extract_manufacturer(extracted_text)
    tax_up_to = extract_tax_info(extracted_text)

    return {
        "Registration Number": reg_number,
        "Chasis Number": chasis_number,
        "Full Name": name,
        "S/W/D of": swd,
        "Address": address,
        "Fuel Type": fuel_type,
        "Vehicle Class": vehicle_class,
        "Vehicle Model": model,
        "Manufacturer": manufacturer,
        "Registration Date": registration_date,
        "Expiry Date": expiry_date,
        "Tax Upto": tax_up_to,
    }


def vehicle_registration(image_path):
    return extract_vehicle_registration_details(image_path)
