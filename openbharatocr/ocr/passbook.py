import cv2
import easyocr
import re


def extract_name(input):
    """
    Extracts the customer name from the given text using a regular expression.

    Args:
        input (str): The text to extract the name from.

    Returns:
        str: The extracted customer name, or None if not found.
    """
    regex = re.compile(r"Customer Name\s+([A-Z\s]+)")
    match = re.search(regex, input)
    if match:
        return match.group(1).strip()
    return None


def extract_open_date(input):
    """
    Extracts the account open date from the given text using a regular expression.

    Args:
        input (str): The text to extract the open date from.

    Returns:
        str: The extracted account open date in DD MMM YYYY format, or None if not found.
    """
    regex = re.compile(r"Open Date\s*(\d{1,2} \w{3} \d{4})")
    match = re.search(regex, input)
    if match:
        return match.group(1)
    return None


def extract_bank_name(input):
    """
    Extracts the bank name from the given text using a regular expression.

    This function searches for patterns containing "Bank", "Bank Ltd",
    "Bank Limited", or "Credit Union" considering case-insensitivity
    and matches across multiple lines.

    Args:
        input (str): The text to extract the bank name from.

    Returns:
        str: The extracted bank name, or None if not found.
    """
    regex = re.compile(
        r"\b[A-Za-z\s&]+(?:BANK|BANK LTD|BANK LIMITED|CREDIT UNION)\b", re.MULTILINE
    )
    match = re.search(regex, input)
    if match:
        return match.group(0).strip()
    else:
        return None


def extract_phone(input):
    """
    Extracts the phone number from the given text using a regular expression.

    This function searches for patterns starting with "Mobile No" and extracts
    the following digits, considering case-insensitivity.

    Args:
        input (str): The text to extract the phone number from.

    Returns:
        str: The extracted phone number, or None if not found.
    """
    regex = re.compile(r"Mobile No\s*(\d+)", re.IGNORECASE)
    match = re.search(regex, input)
    if match:
        return match.group(1).strip()
    return None


def extract_branch_name(input):
    """
    Extracts the branch name from the given text using a regular expression.

    This function searches for patterns starting with "Branch Name" and extracts
    the following text, considering case-insensitivity.

    Args:
        input (str): The text to extract the branch name from.

    Returns:
        str: The extracted branch name, or None if not found.
    """
    regex = re.compile(r"Branch Name\s*([A-Za-z\d\s-]+)", re.IGNORECASE)
    match = re.search(regex, input)
    if match:
        return match.group(1).strip()
    return None


def extract_nomination_name(input):
    """
    Extracts the nomination name from the given text using a regular expression.

    This function searches for patterns containing "Nominee" or "Nomination"
    followed by two capitalized words.

    Args:
        input (str): The text to extract the nomination name from.

    Returns:
        str: The extracted nomination name (full name), or None if not found.
    """
    regex = re.compile(r"Nomina(?:non|tion)\s+([A-Z][a-z]+\s[A-Z][a-z]+)")
    match = re.search(regex, input)
    if match:
        return match.group(1).strip()
    return None


def extract_email(input):
    """
    Extracts the email address from the given text using a regular expression.

    This function searches for email addresses in the format of username@domain.com,
    where username can contain letters, numbers, periods, underscores, plus signs,
    and hyphens, and domain can contain letters, numbers, periods, and hyphens.

    Args:
        input (str): The text to extract the email address from.

    Returns:
        str: The extracted email address, or None if not found.
    """

    regex = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    match = re.search(regex, input)
    if match:
        return match.group(0)
    return None


def extract_account_no(input):
    """
    Extracts the account number from the given text using a regular expression.

    This function searches for patterns containing "Account Number:" followed by
    9 to 12 digits, considering case-insensitivity.

    Args:
        input (str): The text to extract the account number from.

    Returns:
        str: The extracted account number, or None if not found.
    """
    regex = re.compile(r"Account Number:\s*(\d{9,12})", re.IGNORECASE)
    match = re.search(regex, input)
    if match:
        return match.group(1)
    return None


def extract_cif_no(input):
    """
    Extracts the CIF number from the given text using a regular expression.

    This function searches for patterns containing "CIF" (case-insensitive),
    optionally followed by "No" or ".", and then extracts the following digits.

    Args:
        input (str): The text to extract the CIF number from.

    Returns:
        str: The extracted CIF number, or None if not found.
    """
    regex = re.compile(r"CIF(?: No)?\.?\s*(\d+)", re.IGNORECASE)
    match = re.search(regex, input)
    if match:
        return match.group(1)
    return None


def extract_address(input):
    """
    Extracts the address from the given text using a regular expression.

    This function attempts to extract addresses using a list of patterns that
    commonly represent addresses. The patterns include house numbers, street names,
    city/town names, and postal codes.

    Args:
        input (str): The text to extract the address from.

    Returns:
        str: The extracted address, or None if no matching pattern is found.
    """
    regex = [
        r"\d+\s[A-Za-z\s,]+(?:Road|Street|Avenue|Boulevard|Lane|Drive|Court|Place|Square|Plaza|Terrace|Trail|Parkway|Circle)\s*,?\s*(?:\d{5}|\d{5}-\d{4})?",
        r"\d+\s[A-Za-z\s,]+(?:Road|Street|Avenue|Boulevard|Lane|Drive|Court|Place|Square|Plaza|Terrace|Trail|Parkway|Circle)",
        r"\d{1,5}\s[A-Za-z\s]+,\s*[A-Za-z\s]+,\s*[A-Za-z\s]+,\s*\d{5}",
        r"\d{1,5}\s[A-Za-z\s]+,\s*[A-Za-z\s]+,\s*[A-Za-z\s]+",
        r"\d{1,5}\s[A-Za-z\s]+,\s*[A-Za-z\s]+",
        r"[A-Za-z\s]+,\s*[A-Za-z\s]+,\s*[A-Za-z\s]+,\s*\d{5}",
        r"[A-Za-z\s]+,\s*[A-Za-z\s]+,\s*\d{5}",
        r"[A-Za-z\s]+,\s*\d{5}",
    ]
    for pattern in regex:
        match = re.search(pattern, input)
        if match:
            return match.group(0).strip()
    return None


def parse_passbook_frontpage(image_path):
    """
    Parses a passbook front page image to extract various customer and account information.

    This function uses EasyOCR to read text from the image and then employs regular expressions
    to extract specific details like name, account number, address, phone number, etc.

    Args:
        image_path (str): The path to the passbook front page image.

    Returns:
        dict: A dictionary containing the extracted passbook information.
    """
    reader = easyocr.Reader(["en"])

    image = cv2.imread(image_path)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    results = reader.readtext(gray_image)

    extracted_text = " ".join([text for _, text, _ in results])

    passbook_info = {
        "cif_no": extract_cif_no(extracted_text),
        "name": extract_name(extracted_text),
        "account_no": extract_account_no(extracted_text),
        "address": extract_address(extracted_text),
        "phone": extract_phone(extracted_text),
        "email": extract_email(extracted_text),
        "nomination_name": extract_nomination_name(extracted_text),
        "branch_name": extract_branch_name(extracted_text),
        "bank_name": extract_bank_name(extracted_text),
        "date_of_issue": extract_open_date(extracted_text),
    }

    return passbook_info
