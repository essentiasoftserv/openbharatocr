import cv2
import numpy as np
from datetime import datetime
import pytesseract
import re
from fuzzywuzzy import process

# Comprehensive list of bank names
POPULAR_BANKS = [
    "State Bank of India", "Punjab National Bank", "HDFC Bank", "ICICI Bank", "Axis Bank", "Bank of Baroda", 
    "Canara Bank", "Union Bank of India", "Kotak Mahindra Bank", "IndusInd Bank", "RBL Bank", "Jammu & Kashmir Bank",
    "Karnataka Bank", "Karur Vysya Bank", "Punjab & Sind Bank", "South Indian Bank", "City Union Bank", 
    "Tamilnad Mercantile Bank", "DCB Bank", "Bank of Maharashtra", "Indian Overseas Bank", "Federal Bank", 
    "Bandhan Bank", "Central Bank of India", "IDFC FIRST Bank", "Yes Bank", "IDBI Bank", "Indian Bank", 
    "Bank of India", "Union Bank of India"
]


def enhance_contrast(image):
    # Convert the input BGR image to LAB color space to isolate luminance.
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split the LAB image into its three channels: L (luminance), A, and B.
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the luminance channel for localized contrast enhancement.
    # clipLimit controls contrast clipping; tileGridSize defines the grid size for histogram equalization.
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge the enhanced luminance channel back with the original A and B channels.
    limg = cv2.merge((cl, a, b))

    # Convert the LAB image back to BGR color space for further use.
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_image

def preprocess_image(image_path, dpi=400):
    # Read the image
    image = cv2.imread(image_path)

    # Step 1: Enhance contrast
    enhanced_image = enhance_contrast(image)

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)

    # Step 3: Rescale the image for higher DPI (adjust dimensions to DPI)
    scale_factor = dpi / 72  # Typical screen DPI is 72, so scale to 300 DPI
    new_size = (int(gray.shape[1] * scale_factor), int(gray.shape[0] * scale_factor))
    scaled_image = cv2.resize(gray, new_size, interpolation=cv2.INTER_CUBIC)

    # Step 4: Remove noise using GaussianBlur
    blurred = cv2.GaussianBlur(scaled_image, (5, 5), 0)

    # Step 5: Thresholding (binarization) - Convert text to black and background to white
    _, binary_image = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 6: Morphological operations (opening) to clean up the image
    kernel = np.ones((1, 1), np.uint8)
    opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    return opened_image


# Function to clean OCR-extracted text for more reliable data extraction
def clean_text(input_text):
    # Correction map to handle common OCR mistakes
    corrections = {
        "Iqumber": "Number",
        "Nr":"Mr",
        # Add more common corrections here
    }

    for wrong, correct in corrections.items():
        input_text = re.sub(re.escape(wrong), correct, input_text, flags=re.IGNORECASE)

    # Clean non-alphanumeric characters except for common separators
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s\/\-\:\.]', ' ', input_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Reduce multiple spaces to one space
    return cleaned_text.strip()


def extract_cif_no(input_text):
    match = re.search(r"CIF(?:\s*No\.?|No| Number |#)[:\s]*([\d\-]+)", input_text, re.IGNORECASE)
    return match.group(1).replace("-", "").strip() if match else None

def extract_name(input_text):
    # This regex looks for the word "Name" followed by a colon or spaces and captures subsequent characters
    # excluding titles and unnecessary prefixes. It assumes the name follows "Name:" and is a sequence of words
    # with possible additional descriptors like "S/D/W/H/o:"
    match = re.search(r"Name[:\s]*((?:Mr\.?|S\/D\/W\/H\/o:?\s*)?([A-Za-z\s]+))", input_text, re.IGNORECASE)
    return match.group(2).strip() if match else None

def extract_account_no(input_text):
    match = re.search(r"Account(?: No\.?| Number)[:\s]+(\d{9,14})", input_text, re.IGNORECASE)
    return match.group(1) if match else None

def extract_address(cleaned_text):
    # Normalize text for consistent processing
    cleaned_text = cleaned_text.replace("\n", " ").strip()

    # Start by locating the address section
    match = re.search(r"Address[:\s]*([\w\s,./\-]*)", cleaned_text, re.IGNORECASE)

    if match:
        # Extract raw address-like section
        raw_address = match.group(1)

        # Split into parts for line-by-line processing
        lines = re.split(r"[\n:.]", raw_address)

        # Keywords to identify interruptions
        interruptions = {"phone", "email", "branch", "code", "cif", "date", "nom", "account", "mop", "type", "name"}

        # Clean lines by skipping interrupted parts
        valid_address_parts = []
        for line in lines:
            line_lower = line.lower().strip()
            # Skip lines that start with interruption keywords
            if not any(line_lower.startswith(keyword) for keyword in interruptions):
                valid_address_parts.append(line.strip())

        # Join cleaned address parts
        return " ".join(valid_address_parts)

    # If no match found
    return None

def extract_phone(input_text):
    match = re.search(r"Phone(?: No)?\.?\s*(\d{10})", input_text, re.IGNORECASE)
    return match.group(1) if match else None

def extract_branch_name(input_text):
    match = re.search(r"Branch[:\s]+([A-Za-z\s]+)", input_text, re.IGNORECASE)
    return match.group(1).strip() if match else None

def extract_generic_bank_name(input_text):
    # Check explicitly for "SBI", "sbi", or "STATE BANK"
    if re.search(r"\b(SBI|sbi|STATE BANK)\b", input_text, re.IGNORECASE):
        return "STATE BANK OF INDIA"

    # Generic regex for other bank names
    match = re.search(r"\b([A-Za-z\s]+Bank(?:\s*Ltd|\s*Co)?)\b", input_text, re.IGNORECASE)
    return match.group(1).strip() if match else None

def extract_open_date(text):
    date_patterns = [
        r'\b(\d{1,2}/\d{1,2}/\d{4})\b',  # DD/MM/YYYY
        r'\b(\d{1,2}-\d{1,2}-\d{4})\b',  # DD-MM-YYYY
        r'\b(\d{4}-\d{1,2}-\d{1,2})\b',  # YYYY-MM-DD
    ]
    for pattern in date_patterns:
        date_match = re.search(pattern, text)
        if date_match:
            date_str = date_match.group(1)
            try:
                if "-" in date_str and len(date_str) == 10:
                    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                elif "/" in date_str:
                    date_obj = datetime.strptime(date_str, "%d/%m/%Y")
                return date_obj.strftime("%d/%m/%Y")
            except ValueError:
                continue
    return None


def match_with_popular_banks(extracted_bank_name):
    if extracted_bank_name:
        matched_bank = process.extractOne(extracted_bank_name, POPULAR_BANKS)
        if matched_bank and matched_bank[1] > 80:
            return matched_bank[0]
    return None


# Function to sort parsed passbook data by date
def sort_passbooks_by_date(passbooks):
    return sorted(
        passbooks,
        key=lambda x: x.get("date_of_issue", datetime.min),
        reverse=False
    )


# Function to process and extract information from the passbook
def parse_passbook(image_path):
    preprocessed_image = preprocess_image(image_path,dpi=400)
    extracted_text = pytesseract.image_to_string(preprocessed_image, config="--psm 6 --oem 3")

    # Clean the extracted text
    cleaned_text = clean_text(extracted_text)
    print("Cleaned Extracted Text:", cleaned_text)

    raw_bank_name = extract_generic_bank_name(cleaned_text)
    matched_bank_name = match_with_popular_banks(raw_bank_name)

    # Extract key information from the cleaned text
    passbook_info = {
        "cif_no": extract_cif_no(cleaned_text),
        "name": extract_name(cleaned_text),
        "account_no": extract_account_no(cleaned_text),
        "address": extract_address(cleaned_text),
        "phone": extract_phone(cleaned_text),
        "branch_name": extract_branch_name(cleaned_text),
        "bank_name": matched_bank_name if matched_bank_name else raw_bank_name,
        "date_of_issue": extract_open_date(cleaned_text),
    }

    return passbook_info