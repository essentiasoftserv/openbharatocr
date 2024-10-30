import cv2
import pytesseract
import re
import numpy as np
from datetime import datetime

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return morph

def load_yolo_model(weights_path, config_path):
    net = cv2.dnn.readNet(weights_path, config_path)
    print("YOLO model loaded successfully.")
    return net

def extract_water_bill_details(image_path, net):
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError("Image not found or unable to load.")

    # Preprocess the image
    roi = preprocess_image(original_image)

    # Use Tesseract to extract text
    extracted_text = pytesseract.image_to_string(roi)
    print("Extracted Text from Image:\n", extracted_text)

    # Improved regex patterns for extracting details
    name_regex = r"(?i)(?:name|customer name|order name|consumer name|account name|owner name)\s*:\s*([A-Z\s]+)"
    address_regex = r"(?i)(address\s*:?\s*([^\n]+))"
    date_regex = r"(?i)(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
    amount_regex = r"(?i)(?:charges amount\s*:?\s*\$?(\d+\.?\d*)|\$?(\d+\.?\d*))"
    account_regex = r"(?i)(?:seq no|meter no|mt no|account no|number|no)[\s:]*([A-Za-z0-9\-_]{7,})"

    # Extract details using regex
    name_match = re.search(name_regex, extracted_text)
    address_match = re.search(address_regex, extracted_text)
    dates = re.findall(date_regex, extracted_text)
    amounts = re.findall(amount_regex, extracted_text)
    accounts = re.findall(account_regex, extracted_text)

    # Extracted details with fallback
    name = name_match.group(1).strip() if name_match else "Name not found"
    address = address_match.group(2).strip() if address_match else "Address not found"

    # Parse and sort dates
    parsed_dates = []
    for date_str in dates:
        try:
            parsed_date = datetime.strptime(date_str, "%d/%m/%Y")
        except ValueError:
            try:
                parsed_date = datetime.strptime(date_str, "%m/%d/%Y")
            except ValueError:
                continue
        parsed_dates.append(parsed_date)

    # Sort dates to determine reading and due dates
    if parsed_dates:
        parsed_dates.sort()
        reading_date = parsed_dates[0].strftime("%d/%m/%Y")
        due_date = parsed_dates[-1].strftime("%d/%m/%Y")
    else:
        reading_date = "Reading Date not found"
        due_date = "Due Date not found"

    # Extract highest amount, preferring those after "charges amount" if available
    valid_amounts = [float(amount[0] or amount[1]) for amount in amounts if any(amount)]
    if valid_amounts:
        net_amount = max(valid_amounts)
    else:
        net_amount = "Amount not found"

    # Extract account number if close to specified headings
    account_number = "Account number not found"
    for account in accounts:
        if len(account) >= 7:
            account_number = account
            break

    # Print extracted details
    print("Extracted Name:", name)
    print("Extracted Address:", address)
    print("Reading Date:", reading_date)
    print("Due Date:", due_date)
    print("Net Amount:", net_amount)
    print("Account Number:", account_number)

    return {
        'Name': name,
        'Address': address,
        'Reading Date': reading_date,
        'Due Date': due_date,
        'Net Amount': net_amount,
        'Account Number': account_number
    }

# Run extraction
#image_path = "path to image "
#yolo_weights_path = 'path to yolov4.weights'
#yolo_cfg_path = '/path to yolov4.cfg'

net = load_yolo_model(yolo_weights_path, yolo_cfg_path)
