import re
import cv2
import pytesseract
from PIL import Image
import numpy as np
import os
import tempfile
import uuid

YOLO_CFG = os.environ.get("YOLO_CFG", "yolov3_custom.cfg")
YOLO_WEIGHT = os.environ.get("YOLO_WEIGHT", "yolov3_custom_6000.weights")

pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"


def preprocess_for_bold_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    blurred = cv2.GaussianBlur(denoised, (3, 3), 0)
    enhanced = cv2.addWeighted(blurred, 2.5, blurred, -1.0, 0)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sharpened = cv2.filter2D(
        binary, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    )
    return sharpened


def extract_text_from_image(image):
    return pytesseract.image_to_string(image, config="--psm 6").strip()


def extract_voter_details_yolo(image_path):
    image = Image.open(image_path)
    net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHT)
    classes = ["elector", "relation", "voterid"]
    rgb = image.convert("RGB")

    with tempfile.TemporaryDirectory() as tempdir:
        temp_path = os.path.join(tempdir, f"{uuid.uuid4()}.jpg")
        rgb.save(temp_path)
        img = cv2.imread(temp_path)
        height, width = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_outputs = net.forward(net.getUnconnectedOutLayersNames())

        boxes, confidences, class_ids = [], [], []
        results = {}

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    cx, cy, w, h = (
                        detection[:4] * [width, height, width, height]
                    ).astype(int)
                    x, y = cx - w // 2, cy - h // 2
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in indices.flatten():
            x, y, w, h = boxes[i]
            x, y, w, h = max(0, x), max(0, y), min(w, width - x), min(h, height - y)
            roi = img[y : y + h, x : x + w]
            if roi.size == 0:
                continue
            label = classes[class_ids[i]]
            results[label] = extract_text_from_image(roi)

        return results


def extract_voter_id(text):
    match = re.search(r"([A-Z]{2,4}[0-9]{6,8})", text)
    return match.group(1).strip() if match else ""


def extract_names(text):
    candidates = re.findall(r"\b[A-Z][a-z]+\b", text)

    # Common non-name words to ignore
    blacklist = {
        "The",
        "And",
        "For",
        "In",
        "On",
        "At",
        "Of",
        "By",
        "With",
        "A",
        "An",
        "This",
        "That",
        "These",
        "Those",
    }

    # Filter out non-names
    names = [word for word in candidates if word not in blacklist]

    return names


def extract_lines_with_uppercase_words(text):
    lines = []
    for line in text.split("\n"):
        words = re.findall(r"\b[A-Z]{2,}(?:\s+[A-Z]{2,})*", line)
        lines.extend(words)
    return lines


def extract_gender(text):
    if "female" in text.lower():
        return "Female"
    elif "male" in text.lower():
        return "Male"
    return ""


def extract_date(text):
    match = re.search(r"\b\d{2}[-/.]\d{2}[-/.]\d{2,4}\b", text)
    return match.group(0) if match else ""


def extract_address(text):
    match = re.search(
        r"(?:Address\s*[:\-]?\s*)?([A-Za-z0-9,.\-\/\s\n]+?\d{6})", text, re.IGNORECASE
    )
    return match.group(1).strip() if match else ""


def extract_voterid_details_front(image_path):
    image = Image.open(image_path)
    text = extract_text_from_image(image)
    voter_id = extract_voter_id(text)
    names = extract_names(text)
    gender = extract_gender(text)
    dob = extract_date(text)

    if not voter_id or not names or not dob:
        img = preprocess_for_bold_text(cv2.imread(image_path))
        Image.fromarray(img).save("preprocessed_temp.jpg")
        text = extract_text_from_image(Image.open("preprocessed_temp.jpg"))
        if not voter_id:
            voter_id = extract_voter_id(text)
        if not names:
            candidates = extract_lines_with_uppercase_words(text)
            if len(candidates) >= 2:
                names = [candidates[-2], candidates[-1]]
        if not dob:
            dob = extract_date(text)
        if not gender:
            gender = extract_gender(text)

    return {
        "Voter ID": voter_id,
        "Elector's Name": names[0] if len(names) > 0 else "",
        "Father's Name": names[1] if len(names) > 1 else "",
        "Gender": gender,
        "Date of Birth": dob,
    }


def extract_voterid_details_back(image_path):
    text = extract_text_from_image(Image.open(image_path))
    return {
        "Address": extract_address(text),
        "Date of Issue": extract_date(text),
    }


def voter_id_front(image_path):
    text = extract_text_from_image(Image.open(image_path))
    if any(keyword in text.lower() for keyword in ["date", "age", "gender", "sex"]):
        return extract_voterid_details_front(image_path)
    return extract_voter_details_yolo(image_path)


def voter_id_back(image_path):
    return extract_voterid_details_back(image_path)


if __name__ == "__main__":
    front_image_path = "path_to_front_image.jpg"
    back_image_path = "path_to_back_image.jpg"

    front_details = voter_id_front(front_image_path)
    print("Front Details:", front_details)

    back_details = voter_id_back(back_image_path)
    print("Back Details:", back_details)
