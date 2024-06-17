import easyocr
import re

def extract_name(text):
    regex = re.compile(r"Name\s*([A-Za-z\s]+)(?=\s*(?:Sex|Date\s*Birth))", re.IGNORECASE)
    match = re.search(regex, text)
    if match:
        return match.group(1).strip()
    return None

def extract_address_of_birth_place(text):
    regex = re.compile(r"Place\s+of\s+Birth\s*:?[\s\n]*([A-Z\s,.-]+)(?=\s+Name\s+(?:Mcther|Mother))", re.IGNORECASE)
    match = re.search(regex, text)
    if match:
        return match.group(1).strip()
    return None

def extract_dob(text):
    regex = re.compile(r"Date\s+Birth\s+(\d{2}/\d{2}/\d{4})", re.IGNORECASE)
    match = re.search(regex, text)
    if match:
        return match.group(1).strip()
    return None

def extract_mother_name(text):
    regex = re.compile(r"Name\s+(?:Mcther|Mother)\s+([\w]+)", re.IGNORECASE)
    match = re.search(regex, text)
    if match:
        return match.group(1).strip()
    return None

def extract_father_name(text):
    regex = re.compile(r"Name\s+the\s+Father\s+([\w]+)", re.IGNORECASE)
    match = re.search(regex, text)
    if match:
        return match.group(1).strip()
    return None

def extract_registration_number(text):
    regex = re.compile(r"Registration\s+No\s*([\w]+)", re.IGNORECASE)
    match = re.search(regex, text)
    if match:
        return match.group(1).strip()
    return None

def extract_registration_date(text):
    regex = re.compile(r"Date\s+of\s+Registration\s+(\d{2}/\d{2}/\d{4})", re.IGNORECASE)
    match = re.search(regex, text)
    if match:
        return match.group(1).strip()
    return None

def parse_birth_certificate(image_path):
    reader = easyocr.Reader(['en'])  
    result = reader.readtext(image_path, detail=0)  
    
    extracted_text = ' '.join(result)

    birth_certificate_info = {
        "Name": extract_name(extracted_text),
        "Address of Birth Place": extract_address_of_birth_place(extracted_text),
        "Date of Birth": extract_dob(extracted_text),
        "Mother's Name": extract_mother_name(extracted_text),
        "Father's Name": extract_father_name(extracted_text),
        "Registration Number": extract_registration_number(extracted_text),
        "Registration Date": extract_registration_date(extracted_text),
    }

    return birth_certificate_info