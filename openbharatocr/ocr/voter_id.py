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
    """
    Preprocesses an image to enhance bold text for improved OCR extraction.

    This function performs the following steps:

    1. Converts the image to grayscale.
    2. Applies morphological opening (erosion followed by dilation) with a rectangular kernel
       to reduce noise, especially around bold text.
    3. Increases contrast using weighted addition to make bold text stand out more.
    4. Applies binarization with Otsu's thresholding to separate foreground (text) from background.
    5. Applies sharpening using a Laplacian filter to further enhance edges of bold text.

    Args:
        image (numpy.ndarray): The image to preprocess.

    Returns:
        numpy.ndarray: The preprocessed image with enhanced bold text.
    """
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
    """
    Extracts voter information from a voter ID image using YOLO object detection and OCR.

    This function performs the following steps:

    1. Reads the image using Pillow.
    2. Loads a pre-trained YOLO model for object detection. (YOLO_CFG and YOLO_WEIGHT are assumed to be defined elsewhere)
    3. Defines classes to be detected: "elector" (elector's name), "relation" (father's name), "voterid".
    4. Converts the image to RGB format (assuming YOLO expects RGB).
    5. Uses a temporary directory to store a temporary image file for processing.
    6. Reads the temporary image using OpenCV.
    7. Performs object detection using YOLO, identifying bounding boxes and class labels for detected objects.
    8. Initializes empty dictionaries for boxes, confidences, class IDs, and detected texts.
    9. Iterates through each detected object's information:
       - Extracts bounding box coordinates (x, y, width, height).
       - Calculates absolute coordinates based on the image size.
       - Crops the image based on the bounding box to isolate the detected region.
       - Skips empty crops to avoid errors.
       - Extracts text from the cropped image using Tesseract with configuration for single block processing.
       - Stores the extracted text along with the corresponding class label in the detected_texts dictionary.
    10. Returns the dictionary containing detected texts categorized by their labels (e.g., "elector": "John Doe").

    Args:
        image_path (str): The path to the voter ID image.

    Returns:
        dict: A dictionary containing voter information extracted using OCR, categorized by labels (e.g., "elector": "John Doe").
    """
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
    """
    Extracts the voter ID number from the given text using a regular expression.

    This function searches for a pattern containing 0 to 3 optional characters followed by 7 digits.

    Args:
        input (str): The text to extract the voter ID from.

    Returns:
        str: The extracted voter ID, or an empty string if not found.
    """

    regex = r".{0,3}[0-9]{7}"
    match = re.search(regex, input)
    voter_id = match.group(0) if match else ""

    return voter_id


def extract_names(input):
    """
    Extracts names from the given text using a regular expression.

    This function searches for the word "Name" followed by an optional colon, equal sign, or plus sign, and then captures any following characters.
    It extracts all occurrences and returns a list, handling potential multiple names.

    Args:
        input (str): The text to extract names from.

    Returns:
        list: A list of extracted names (strings), or an empty list if not found.
    """
    regex = r"Name\s*[:=+]?\s*(.*)"
    matches = re.findall(regex, input, re.IGNORECASE)
    names = [match.strip() for match in matches] if matches else []

    return names


def extract_lines_with_uppercase_words(input):
    """
    Extracts lines containing sequences of uppercase words from the given text.

    This function iterates through lines in the input text:
    1. Uses a regular expression to search for lines containing one or more uppercase words separated by spaces.
    2. If a match is found, extracts all uppercase words using the same regular expression and appends them to a list.

    Args:
        input (str): The text to extract lines with uppercase words from.

    Returns:
        list: A list of extracted lines containing sequences of uppercase words (strings),
              or an empty list if none are found.
    """
    lines_with_uppercase_words = []
    pattern = r"\b[A-Z]+(?:\s+[A-Z]+)*\b"
    for line in input.split("\n"):
        if re.search(pattern, line):
            uppercase_words = re.findall(pattern, line)
            for word in uppercase_words:
                lines_with_uppercase_words.append(word)
    return lines_with_uppercase_words


def extract_gender(input):
    """
    Extracts the gender from the given text using case-insensitive matching.

    This function searches for the presence of "Female" or "Male" (or their uppercase equivalents) in the input text.
    It returns "Female" if a match for "Female" is found, "Male" if a match for "Male" is found, otherwise returns an empty string.

    Args:
        input (str): The text to extract the gender from.

    Returns:
        str: The extracted gender ("Female" or "Male"), or an empty string if not found.
    """

    if "Female" in input or "FEMALE" in input:
        return "Female"
    elif "Male" in input or "MALE" in input:
        return "Male"
    else:
        return ""


def extract_date(input):
    """
    Extracts the date of birth from the given text using a regular expression.

    This function searches for a pattern containing two digits followed by a separator (slash, hyphen, or dot),
    another two digits followed by a separator, and then either four or two digits representing the year.
    The entire pattern must be surrounded by word boundaries.

    Args:
        input (str): The text to extract the date of birth from.

    Returns:
        str: The extracted date of birth (in format DD/MM/YYYY or DD-MM-YYYY), or an empty string if not found.
    """
    regex = r"\b([0-9X]{2}[/\-.][0-9X]{2}[/\-.](?:\d{4}|\d{2}))\b"
    match = re.search(regex, input)
    dob = match.group(0) if match else ""

    return dob


def extract_address(input):
    """
    Extracts the address from the given text using a regular expression with two approaches.

    This function prioritizes lines containing "Address" followed by an optional colon and any characters/spaces,
    ending with a postal code (6 digits). If not found, it attempts to extract any line containing
    an address-like pattern (alphanumeric characters, punctuation, spaces) ending with a postal code.

    Args:
        input (str): The text to extract the address from.

    Returns:
        str: The extracted address (including postal code), or an empty string if not found.
    """
    regex = r"Address\s*:?\s*[A-Za-z0-9:,-.\n\s\/]+[0-9]{6}"
    match = re.search(regex, input)
    address = match.group(0) if match else ""

    if not match:
        regex = r"[A-Za-z0-9:,-.\n\s\/]+[0-9]{6}"
        match = re.search(regex, input)
        address = match.group(0) if match else ""

    return address


def extract_voterid_details_front(image_path):
    """
    Extracts voter information from the front side of a voter ID image using OCR with fallback for non-standard layouts.

    This function performs the following steps:

    1. Opens the image using Pillow.
    2. Extracts text using Tesseract (assuming the text is in a supported language).
    3. Extracts voter ID, names (elector's and father's), gender, and date of birth using regular expressions.
    4. Converts the image to RGB format.
    5. Creates a temporary file to store a preprocessed image.
    6. Reads the image using OpenCV.
    7. Applies pre-processing to enhance bold text for better OCR.
    8. Extracts text again from the preprocessed image.
    9. If elector's name is not found using the initial extraction:
       - Extracts lines containing sequences of uppercase words, potentially containing names.
       - Assigns the last two words (assuming the second-last is the father's name) to elector's name and father's name.
    10. Similar logic is applied to extract the date of birth and voter ID if not found initially.
    11. Extracts gender using string matching.
    12. Returns a dictionary containing extracted voter information.

    Args:
        image_path (str): The path to the front side of the voter ID image.

    Returns:
        dict: A dictionary containing extracted voter information
              (e.g., "Voter ID", "Elector's Name", "Father's Name", "Gender", "Date of Birth").
    """
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
    """
    Extracts address and date of issue from the back side of a voter ID image using OCR.

    This function performs the following steps:

    1. Opens the image using Pillow.
    2. Extracts text using Tesseract (assuming the text is in a supported language).
    3. Extracts address and date of issue using regular expressions.
    4. Returns a dictionary containing extracted information.

    Args:
        image_path (str): The path to the back side of the voter ID image.

    Returns:
        dict: A dictionary containing extracted information
              (e.g., "Address", "Date of Issue").
    """
    image = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(image)

    address = extract_address(extracted_text)
    doi = extract_date(extracted_text)

    return {"Address": address, "Date of Issue": doi}


def voter_id_front(front_path):
    """
    Extracts voter information from the front side of a voter ID image using an adaptive approach.

    This function first performs basic OCR to see if the layout includes keywords
    like "Date", "Age", "Sex", or "Gender". If these keywords are found, it assumes
    a standard layout and uses the `extract_voterid_details_front` function for extraction.
    Otherwise, it employs the `extract_voter_details_yolo` function, which might be
    more suitable for non-standard layouts that may require object detection.

    Args:
        front_path (str): The path to the front side of the voter ID image.

    Returns:
        dict: A dictionary containing extracted voter information.
    """
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
    """
    Extracts address and date of issue from the back side of a voter ID image.

    This function calls the `extract_voterid_details_back` function to process the
    back side image and extract relevant information.

    Args:
        back_path (str): The path to the back side of the voter ID image.

    Returns:
        dict: A dictionary containing extracted information
              (e.g., "Address", "Date of Issue").
    """
    back_details = extract_voterid_details_back(back_path)
    return back_details
