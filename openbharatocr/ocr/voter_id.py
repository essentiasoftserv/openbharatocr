import re
import cv2
import pytesseract
from PIL import Image
import numpy as np
import os
import tempfile
import uuid

# Download the models from links and set in the environment
YOLO_CFG = os.environ.get(
    "YOLO_CFG", "yolov3_custom.cfg"
)  # https://drive.google.com/file/d/1SEst2lVoFDOgUVLZ5kje9GTb2tHRA8U-/view?usp=sharing
YOLO_WEIGHT = os.environ.get(
    "YOLO_WEIGHT", "yolov3_custom_6000.weights"
)  # https://drive.google.com/file/d/1cGGstycfogmO6O7ToB2DAEXOgTWVgINh/view?usp=drive_link


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


def extract_voter_details_yolo(image_path):
    image = Image.open(image_path)
    net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHT)
    classes = ["elector", "relation", "voterid"]

    rgb = image.convert("RGB")
    with tempfile.TemporaryDirectory() as tempdir:
        tempfile_path = f"{tempdir}/{str(uuid.uuid4())}.jpg"
        rgb.save(tempfile_path)

        img = cv2.imread(tempfile_path)
        if img is None:
            print("Error: Unable to read the input image.")
            exit()

        height, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(
            img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False
        )

        net.setInput(blob)
        output_layers_name = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_name)

        boxes = []
        confidences = []
        class_ids = []
        detected_texts = {}

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]

                x = max(0, x)
                y = max(0, y)
                w = min(width - x, w)
                h = min(height - y, h)

                label = str(classes[class_ids[i]])
                crop_img = img[y : y + h, x : x + w]
                if crop_img.size == 0:
                    continue
                text = pytesseract.image_to_string(crop_img, config="--psm 6")
                detected_texts[label] = text.strip()

        return detected_texts


def extract_voter_id(input):
    regex = r".{0,3}[0-9]{7}"
    match = re.search(regex, input)
    voter_id = match.group(0) if match else ""

    return voter_id


def extract_names(input):
    regex = r"Name\s*[:=+]?\s*(.*)"
    matches = re.findall(regex, input, re.IGNORECASE)
    names = [match.strip() for match in matches] if matches else []

    return names


def extract_lines_with_uppercase_words(input):
    lines_with_uppercase_words = []
    pattern = r"\b[A-Z]+(?:\s+[A-Z]+)*\b"
    for line in input.split("\n"):
        if re.search(pattern, line):
            uppercase_words = re.findall(pattern, line)
            for word in uppercase_words:
                lines_with_uppercase_words.append(word)
    return lines_with_uppercase_words


def extract_gender(input):
    if "Female" in input or "FEMALE" in input:
        return "Female"
    elif "Male" in input or "MALE" in input:
        return "Male"
    else:
        return ""


def extract_date(input):
    regex = r"\b([0-9X]{2}[/\-.][0-9X]{2}[/\-.](?:\d{4}|\d{2}))\b"
    match = re.search(regex, input)
    dob = match.group(0) if match else ""

    return dob


def extract_address(input):
    regex = r"Address\s*:?\s*[A-Za-z0-9:,-.\n\s\/]+[0-9]{6}"
    match = re.search(regex, input)
    address = match.group(0) if match else ""

    if not match:
        regex = r"[A-Za-z0-9:,-.\n\s\/]+[0-9]{6}"
        match = re.search(regex, input)
        address = match.group(0) if match else ""

    return address


def extract_voterid_details_front(image_path):
    image = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(image)

    voter_id = extract_voter_id(extracted_text)

    names = extract_names(extracted_text)
    electors_name = names[0] if len(names) > 0 else ""
    fathers_name = names[1] if len(names) > 1 else ""

    gender = extract_gender(extracted_text)

    dob = extract_date(extracted_text)
    rgb = image.convert("RGB")
    with tempfile.TemporaryDirectory() as tempdir:
        tempfile_path = f"{tempdir}/{str(uuid.uuid4())}.jpg"
        rgb.save(tempfile_path)

        image = cv2.imread(tempfile_path)
        preprocessed = preprocess_for_bold_text(image)
        cv2.imwrite("preprocessed_image.jpg", preprocessed)

        image = Image.open("preprocessed_image.jpg")
        clean_text = pytesseract.image_to_string(image)

        if electors_name == "":
            names = extract_lines_with_uppercase_words(clean_text)
            electors_name = names[-2] if len(names) > 1 else ""
            fathers_name = names[-1] if len(names) > 0 else ""

        if dob == "":
            dob = extract_date(clean_text)

        if voter_id == "":
            voter_id = extract_voter_id(clean_text)

        if gender == "":
            gender = extract_gender(clean_text)

        return {
            "Voter ID": voter_id,
            "Elector's Name": electors_name,
            "Father's Name": fathers_name,
            "Gender": gender,
            "Date of Birth": dob,
        }


def extract_voterid_details_back(image_path):
    image = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(image)

    address = extract_address(extracted_text)
    doi = extract_date(extracted_text)

    return {"Address": address, "Date of Issue": doi}


def voter_id_front(front_path):
    image = Image.open(front_path)
    extracted_text = pytesseract.image_to_string(image)

    if (
        "Date" in extracted_text
        or "Age" in extracted_text
        or "Sex" in extracted_text
        or "Gender" in extracted_text
    ):
        return extract_voterid_details_front(front_path)
    else:
        return extract_voter_details_yolo(front_path)


def voter_id_back(back_path):
    back_details = extract_voterid_details_back(back_path)
    return back_details
