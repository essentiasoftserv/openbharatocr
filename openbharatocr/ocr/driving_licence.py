import re
from PIL import Image
from pytesseract import image_to_string


def extract_driving_licence_number(text: str) -> str:
    """
    Extract driving licence number from text.
    Format may be DL-1420110012345 or DL14 20110012345
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string")

    pattern = r"\b[A-Z]{2}[- ]?\d{2}[- ]?\d{11}\b|\b[A-Z]{2}\d{2}\s\d{11}\b"
    match = re.search(pattern, text)
    return match.group(0) if match else None


def extract_all_dates(text: str):
    """
    Extract all dates from text and classify into
    DOB, DOI (issue dates), Validity (expiry).
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string")

    pattern = r"\b\d{2}/\d{2}/\d{4}\b"
    matches = re.findall(pattern, text)

    dob, doi, validity = None, [], []
    if matches:
        dob = matches[0]
        if len(matches) > 1:
            doi = matches[1:-1]
        if len(matches) > 1:
            validity = [matches[-1]]

    return dob, doi, validity


def clean_input(matches: list) -> list:
    """
    Clean names list by splitting on newline.
    """
    if not isinstance(matches, list):
        raise TypeError("Input must be a list")

    cleaned = []
    for m in matches:
        parts = str(m).split("\n")
        cleaned.extend([p.strip() for p in parts if p.strip()])
    return cleaned


def extract_all_names(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("Input must be a string")

    stopwords = {"INDIA", "TRANSPORT", "LICENCE"}

    # Remove prefixes like 'Name:', 'NAME -', etc.
    text = re.sub(r"(?i)\bname[:\-]?", "", text).strip()

    words = text.split()

    # if any stopword is present, reject
    if any(sw in words for sw in stopwords):
        return ""

    return " ".join(words[:2]) if len(words) >= 2 else ""


def extract_address_regex(text: str) -> str:
    """
    Extract address using regex if present.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string")

    pattern1 = r"Address\s*:\s*(.*)"
    pattern2 = r"ADDRESS\s*-\s*(.*)"

    match1 = re.search(pattern1, text, flags=re.IGNORECASE | re.DOTALL)
    if match1:
        addr = match1.group(1).split("\n\n")[0].strip()
        return addr

    match2 = re.search(pattern2, text, flags=re.IGNORECASE)
    if match2:
        addr = match2.group(1)
        # stop at 6 digit pincode or Other text
        addr = re.split(r"\d{6}|Other", addr)[0].strip()
        return addr

    return ""


def extract_address(image_path: str) -> str:
    """
    Extract address from image using OCR.
    """
    try:
        img = Image.open(image_path)
        text = image_to_string(img)
        return extract_address_regex(text)
    except FileNotFoundError:
        raise
    except Exception as e:
        raise e


def extract_auth_allowed(text: str) -> list:
    if not isinstance(text, str):
        raise TypeError("Input must be a string")

    pattern = r"(MCWG|LMV|TRANS|M\.CYL\.)"
    return re.findall(pattern, text)


def driving_licence(image_path: str) -> dict:
    """
    Master function: Extracts all details from a Driving Licence image.
    """
    from PIL import Image
    from pytesseract import image_to_string

    img = Image.open(image_path)
    text = image_to_string(img)

    dob, doi, validity = extract_all_dates(text)

    details = {
        "licence_number": extract_driving_licence_number(text),
        "dates": {"dob": dob, "doi": doi, "validity": validity},
        "name": extract_all_names(text),
        "address": extract_address_regex(text),
        "auth_types": extract_auth_allowed(text),
        "raw_text": text,
    }
    return details

    # def driving_licence(image_path: str) -> dict:
    #     """
    #     Master function: Extracts all details from a Driving Licence image
    #     Returns dict with number, dates, name, address, and auth types.
    #     """
    #     # OCR
    #     img = Image.open(image_path)
    #     text = image_to_string(img)

    #     details = {
    #         "licence_number": extract_driving_licence_number(text),
    #         "dates": {
    #             "dob": None,
    #             "doi": [],
    #             "validity": []
    #         },
    #         "name": extract_all_names(text),
    #         "address": extract_address_regex(text),
    #         "auth_types": extract_auth_allowed(text),
    #         "raw_text": text
    #     }

    #     dob, doi, validity = extract_all_dates(text)
    #     details["dates"]["dob"] = dob
    #     details["dates"]["doi"] = doi
    #     details["dates"]["validity"] = validity

    #     return details

    # # Optional CLI run
    # if __name__ == "__main__":
    sample = "sample_driving_licence.jpg"  # replace with real file
    try:
        result = driving_licence(sample)
        for k, v in result.items():
            print(f"{k}: {v}")
    except Exception as e:
        print("Error:", e)
