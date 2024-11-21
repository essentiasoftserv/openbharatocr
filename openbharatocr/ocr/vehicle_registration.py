import re
import pytesseract
from PIL import Image


def extract_names(input_text):
    """
    Extracts owner name and son/wife/daughter of (SWD) information from the given text using regular expressions.

    This function attempts to extract names in three ways, prioritizing formats
    containing "Dual Owner", "NAME", and "S/O W/D" patterns.

    Args:
        input_text (str): The text to extract names from.

    Returns:
        tuple: A tuple containing the extracted full name (string)
               and son/wife/daughter of information (SWD, string),
               or empty strings if not found.
    """
    # Regular expression for extracting SWD information
    regex_swd = r"dual\sOwner\)?\s*:?\s*\n([A-Z.]+\s[A-Z.]+\s[A-Z.]+)"
    match = re.search(regex_swd, input_text, re.IGNORECASE)
    swd = match.group(1) if match else ""

    # Regular expression for extracting NAME information
    regex_name = r"NAME\s*:?\s*([A-Z]+\s[A-Z]+\s[A-Z.]+|[A-Z]+\s[A-Z.]+)"
    match_name = re.search(regex_name, input_text, re.IGNORECASE)
    if match_name:
        name_parts = match_name.group(1).split()
        
        # Define a set of unwanted suffixes and non-last-name words
       unwanted_suffixes = {"Son", "Daughter", "Wife", "verncie", "Other"}
        
        # Filter out unwanted suffixes if they appear at the end of the name
        while name_parts and name_parts[-1] in unwanted_suffixes:
            name_parts.pop()

        # Ensure the final name has only first, middle, and last name components
        if len(name_parts) > 3:
            name_parts = name_parts[:3]
            
        name = " ".join(name_parts)
    else:
        name = ""
    
    # Additional regex checks for SWD if not found
    if not swd:
        regex_swd = r"NAME\s*:?\s*([A-Z]+\s[A-Z.]+\s[A-Z.]+|[A-Z]+\s[A-Z.]+)"
        match_swd = re.findall(regex_swd, input_text, re.IGNORECASE)
        if len(match_swd) > 1:
            swd_parts = match_swd[1].split()
            
            # Remove any unwanted suffixes if present
            while swd_parts and swd_parts[-1] in unwanted_suffixes:
                swd_parts.pop()
            
            # Keep only the first three parts as the valid SWD information
            swd = " ".join(swd_parts[:3])
    # Final fallback for SWD extraction
    if not swd:
        regex_swd = r"OF\s*:?\s*[S/O01]*\s*\n([A-Z]+\s[A-Z]+)"
        match = re.search(regex_swd, input_text, re.IGNORECASE)
        swd = match.group(1) if match else ""
    
    return name, swd


def extract_reg_number(input):
    """
    Extracts the vehicle registration number from the given text using a regular expression.

    This function searches for a pattern containing at least one digit followed by
    10 alphanumeric characters.

    Args:
        input (str): The text to extract the registration number from.

    Returns:
        str: The extracted registration number, or an empty string if not found.
    """
    regex = r"(?=.*\d)[A-Z0-9]{10}"
    match = re.search(regex, input)
    reg_number = match.group(0) if match else ""

    return reg_number


def extract_chasis(input):
    """
    Extracts the chasis number from the given text using a regular expression.

    This function searches for a pattern containing 17 or 18 alphanumeric characters.

    Args:
        input (str): The text to extract the chasis number from.

    Returns:
        str: The extracted chasis number, or an empty string if not found.
    """
    regex = r"(?i)\b([A-Z0-9]{17})\b"
    match = re.search(regex, input)
    chasis = match.group(0) if match else ""

    return chasis


def extract_fuel_type(input):
    """
    Extracts the fuel type from the given text using a regular expression.

    This function searches for patterns containing "Fuel Type" or "Fuel" followed by
    a colon or period, and then extracts the following text containing letters and slashes.

    Args:
        input (str): The text to extract the fuel type from.

    Returns:
        str: The extracted fuel type, or an empty string if not found.
    """
    # Regular expression to match specific fuel types directly
    regex = r"\b(diesel|petrol|electric)\b"
    match = re.search(regex, input_text, re.IGNORECASE)
    
    # If a match is found, return the fuel type with standardized capitalization
    fuel_type = match.group(1).capitalize() if match else ""
    return fuel_type

def extract_vehicle_class(input):
    """
    Extracts the vehicle class from the given text using a regular expression.

    This function searches for patterns containing "Veh.Class" or "Veh Cl" followed by
    a colon or period, and then extracts two words separated by spaces or special characters.

    Args:
        input (str): The text to extract the vehicle class from.

    Returns:
        str: The extracted vehicle class (two words combined), or an empty string if not found.
    """
    regex = r"(?:Veh.c.e\sClass|Veh\sCl)\s*[\s:]\s*([A-Z0-9/()-]+)\s([A-Z0-9/()-]+)\s"
    match = re.search(regex, input, re.IGNORECASE)
    vehicle_class = match.group(1) if match else ""
    return vehicle_class


def extract_manufacturer(input):
    """
    Extracts the vehicle manufacturer from the given text using a regular expression.

    This function searches for a pattern containing "MFR" followed by a colon and extracts
    the following text containing letters and spaces.

    Args:
        input (str): The text to extract the manufacturer from.

    Returns:
        str: The extracted manufacturer, or an empty string if not found.
    """
    regex = r"MFR\s*:\s*([A-Z\s]+)\n"
    match = re.search(regex, input, re.IGNORECASE)
    manufacturer = match.group(1) if match else ""
    return manufacturer


def extract_tax_info(input):
    """
    Extracts tax information (up to which month/year) from the given text using a regular expression.

    This function searches for a pattern containing "Tax Up To" followed by a colon or space,
    and then extracts the following word (assuming it represents the month/year).

    Args:
        input (str): The text to extract the tax information from.

    Returns:
        str: The extracted tax information (month/year), or an empty string if not found.
    """
    regex = r"Tax\sUp\s{0,1}to\s*:\s*([A-Z]+)\s"
    match = re.search(regex, input, re.IGNORECASE)
    tax_up_to = match.group(1) if match else ""
    return tax_up_to


def extract_model(input):
    """
    Extracts the vehicle model from the given text using a regular expression.

    This function searches for a pattern containing "Model" followed by a colon or space,
    and then extracts the following text containing letters, numbers, forward slashes,
    hyphens, parentheses, periods, and spaces (up to 4 words).

    Args:
        input (str): The text to extract the model from.

    Returns:
        str: The extracted vehicle model, or an empty string if not found.
    """
    regex = r"Mode.\s*[\s:]\s*([A-Z0-9/+()-.]+(?:\s+[^\w\n]*[A-Z0-9/+()-.]+){0,3})\s"
    match = re.search(regex, input, re.IGNORECASE)
    model = match.group(1) if match else ""
    return model


def extract_all_dates(input_text):
    """
    Extracts all dates from the given text using a regular expression.

    This function searches for patterns in formats like DD/MM/YYYY, DD-MM-YYYY, or
    MMM/YYYY, and sorts the extracted dates chronologically.

    Args:
        input_text (str): The text to extract dates from.

    Returns:
        list: A list of extracted dates sorted in ascending order (strings).
    """
    regex = r"\b(\d{1,2}[/\-.](?:\d{2}|\d{4}|\w{3})[/\-.]\d{2,4})\b"
    dates = re.findall(regex, input_text)
    sorted_dates = sorted(
        dates, key=lambda date: int(date.split("/")[-1].split("-")[-1])
    )

    return sorted_dates


def extract_address(input):
    """
    Extracts the address from the given text using a regular expression.

    This function searches for patterns containing "Address" (optional colon)
    followed by any characters and spaces, prioritizing lines ending with a postal code (6 digits).

    Args:
        input (str): The text to extract the address from.

    Returns:
        str: The extracted address, or an empty string if not found.
    """
    regex = r"Address:?\s*((?:.|\n)*?\d{6})"
    match = re.search(regex, input, re.IGNORECASE)
    if match:
        address = match.group(1).replace('\n', ' ')
    
        # Remove unwanted phrases
        unwanted_phrases = ["Emission Norms", "Not Available"]
        for phrase in unwanted_phrases:
            address = re.sub(r'\b' + re.escape(phrase) + r'\b', '', address)
    
        # Remove extra spaces introduced by removals
        address = re.sub(r'\s+', ' ', address).strip()
    else:
        address = ""
    #address = match.group(1).replace('\n', ' ') if match else ""

    return address


def extract_vehicle_registration_details(image_path):
    """
    Extracts vehicle registration details from an image using a combination of OCR and text processing.

    This function performs the following steps:

    1. Reads the image using Pillow.
    2. Extracts text using Tesseract (assuming the text is in a supported language).
    3. Extracts owner name and son/wife/daughter of (SWD) information using regular expressions.
    4. Extracts all dates using a regular expression and sorts them chronologically.
    5. Extracts vehicle details like registration number, chasis number, fuel type,
       vehicle class, and model using regular expressions.
    6. Extracts manufacturer and tax information using regular expressions.
    7. Extracts address using a regular expression prioritizing lines ending with a postal code.

    Args:
        image_path (str): The path to the vehicle registration image.

    Returns:
        dict: A dictionary containing extracted vehicle registration details with keys like
              "Registration Number", "Chasis Number", "Full Name", etc.
    """
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
    """
    Extracts vehicle registration details from an image using the extract_vehicle_registration_details function.

    Args:
        image_path (str): The path to the vehicle registration image.

    Returns:
        dict: A dictionary containing extracted vehicle registration details.
    """
    return extract_vehicle_registration_details(image_path)
