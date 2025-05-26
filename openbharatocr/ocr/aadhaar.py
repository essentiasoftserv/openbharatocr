import cv2
import pytesseract
from PIL import Image
import re
import numpy as np


def preprocess_image_light(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return gray


def crop_back_top_left_area(image):
    h, w = image.shape[:2]
    x_start, y_start = 0, 0
    x_end = int(w * 0.6)
    y_end = int(h * 0.5)
    return image[y_start:y_end, x_start:x_end]


def ocr_image_with_config(image, config="--psm 6"):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    text = pytesseract.image_to_string(image, config=config)
    print("----- OCR TEXT START -----")
    print(text)
    print("------ OCR TEXT END ------\n")
    return text


def extract_name(text):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    garbage_keywords = [
        "Government",
        "Of",
        "India",
        "Unique",
        "Identification",
        "Authority",
        "DOB",
        "Gender",
    ]

    for line in lines:
        if (
            re.match(r"^[A-Za-z\s]+$", line)
            and len(line.split()) >= 2
            and not any(word in line for word in garbage_keywords)
            and len(line) <= 40
        ):
            return line
    return ""


def extract_dob(text):
    match = re.search(
        r"(?:DOB|Date of Birth|D\.O\.B|D O B|D\.O\.B\.)[:\s]*([0-9]{2}/[0-9]{2}/[0-9]{4})",
        text,
        re.I,
    )
    if match:
        return match.group(1)
    match = re.search(r"([0-9]{2}/[0-9]{2}/[0-9]{4})", text)
    if match:
        return match.group(1)
    return ""


def extract_gender(text):
    match = re.search(r"\b(Male|Female|Transgender)\b", text, re.I)
    if match:
        return match.group(1).capitalize()
    return ""


def extract_aadhaar_number(text):
    text = text.replace("\n", " ")
    match = re.search(r"(\d{4}\s*\d{4}\s*\d{4})", text)
    if match:
        return match.group(1).replace(" ", "")
    match = re.search(r"\b(\d{12})\b", text)
    if match:
        return match.group(1)
    return ""


def extract_fathers_name_and_address(text):
    father_name = ""
    address = ""

    text = text.replace("S/ o", "S/o").replace("S/ O", "S/o").replace("s/ o", "s/o")

    match = re.search(
        r"(?:S/o|S/O|Son of|W/o|D/o|W/O|D/O)[:\s]*([A-Z][a-zA-Z\s]{3,40})", text, re.I
    )
    if match:
        fn_candidate = match.group(1).strip()
        if len(fn_candidate.split()) >= 2:
            father_name = fn_candidate

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    garbage_patterns = [
        r"mera\s+aadhaar",
        r"www\.",
        r"uidai",
        r"https?://",
        r"email",
        r"@",
        r"contact\s+us",
    ]
    address_lines = []
    addr_start = False

    for line in lines:
        line_lower = line.lower()
        if father_name and father_name in line:
            addr_start = True
            continue
        if "address" in line_lower:
            addr_start = True
            continue

        if addr_start:
            if any(re.search(p, line_lower) for p in garbage_patterns):
                continue
            if re.match(r"^\d{4}\s*\d{4}\s*\d{4}$", line.replace(" ", "")):
                continue
            if len(re.findall(r"[A-Za-z]", line)) < max(3, len(line) * 0.5):
                continue
            address_lines.append(line)
            if re.search(r"\b\d{6}\b", line):  # Indian PIN code
                break

    address = ", ".join(address_lines).strip()

    # fallback if no good address
    if not address:
        fallback_lines = [
            l
            for l in lines
            if re.search(r"\d", l)
            or re.search(r"(village|city|state|pin|postcode|district|town)", l, re.I)
        ]
        address = ", ".join(fallback_lines).strip()

    return father_name, address


def extract_front_aadhaar_details(image_path):
    print(f"Processing front image: {image_path}")
    image = cv2.imread(image_path)
    text = ocr_image_with_config(image)
    return {
        "Full Name": extract_name(text),
        "Date of Birth": extract_dob(text),
        "Gender": extract_gender(text),
        "Aadhaar Number": extract_aadhaar_number(text),
    }


def extract_front_aadhaar_details_preprocessed(image_path):
    print(f"Processing front image with preprocessing: {image_path}")
    gray = preprocess_image_light(image_path)
    text = ocr_image_with_config(gray)
    return {
        "Full Name": extract_name(text),
        "Date of Birth": extract_dob(text),
        "Gender": extract_gender(text),
        "Aadhaar Number": extract_aadhaar_number(text),
    }


def extract_back_aadhaar_details(image_path):
    print(f"Processing back image: {image_path}")
    image = cv2.imread(image_path)
    text = ocr_image_with_config(image)
    return dict(
        zip(["Father's Name", "Address"], extract_fathers_name_and_address(text))
    )


def extract_back_aadhaar_details_preprocessed(image_path):
    print(f"Processing back image with preprocessing: {image_path}")
    gray = preprocess_image_light(image_path)
    text = ocr_image_with_config(gray)
    return dict(
        zip(["Father's Name", "Address"], extract_fathers_name_and_address(text))
    )


def extract_back_aadhaar_details_roi(image_path):
    print(f"Processing back image with top-left ROI cropping: {image_path}")
    image = cv2.imread(image_path)
    gray = preprocess_image_light(image_path)
    cropped = crop_back_top_left_area(gray)
    text = ocr_image_with_config(cropped)
    return dict(
        zip(["Father's Name", "Address"], extract_fathers_name_and_address(text))
    )


def merge_results(res1, res2):
    merged = {}
    for key in res1.keys():
        v1 = res1.get(key, "")
        v2 = res2.get(key, "")
        merged[key] = v1 if len(v1) >= len(v2) else v2
    return merged


def extract_full_aadhaar_with_roi(front_image_path, back_image_path):
    front1 = extract_front_aadhaar_details(front_image_path)
    front2 = extract_front_aadhaar_details_preprocessed(front_image_path)
    front = merge_results(front1, front2)

    back1 = extract_back_aadhaar_details(back_image_path)
    back2 = extract_back_aadhaar_details_preprocessed(back_image_path)
    back3 = extract_back_aadhaar_details_roi(back_image_path)
    back = {}
    for k in back1:
        candidates = [back1.get(k, ""), back2.get(k, ""), back3.get(k, "")]
        back[k] = max(candidates, key=len)

    return {**front, **back}


if __name__ == "__main__":
    front_img_path = "path/to/front/aadhaar.jpg"
    back_img_path = "path/to/back/aadhaar.jpg"

    data = extract_full_aadhaar_with_roi(front_img_path, back_img_path)
    print("\nFinal Extracted Aadhaar Details:")
    for k, v in data.items():
        print(f"{k}: {v if v else 'Not Found'}")
