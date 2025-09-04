# import cv2
# import pytesseract
# import re
# import numpy as np
# from pytesseract import Output
# from PIL import Image


# def preprocess_image(image_path):
#     """
#     Preprocesses the image to enhance text for OCR.

#     Args:
#         image_path (str): The path to the image.

#     Returns:
#         numpy.ndarray: The preprocessed image.
#     """
#     # Read the image
#     image = cv2.imread(image_path)

#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Apply Gaussian blur to smooth the image
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#     # Apply adaptive thresholding to create a binary image
#     binary = cv2.adaptiveThreshold(
#         blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
#     )

#     # Invert colors for better OCR performance
#     inverted_image = cv2.bitwise_not(binary)

#     return inverted_image


# import re


# def extract_name(text):
#     """
#     Extracts the recipient's name from the given text.

#     This function uses a regular expression to search for patterns commonly found in degree certificates that indicate the recipient's name.

#     Args:
#     text: The text extracted from the degree certificate image.

#     Returns:
#     The extracted recipient's name as a string, or None if no name is found.
#     """
#     patterns = [
#         r"conferred on",
#         r"conferred upon",
#         r"awarded to",
#         r"certify that",
#         r"certifies that",
#         r"testify that",
#         r"known that",
#         r"admits",
#         r"granted",
#     ]

#     # Create a regex pattern by joining all patterns with an optional whitespace and capturing the name
#     name_pattern = r"(?:{})\s+([A-Z][a-zA-Z' -]+(?:\s[A-Z][a-zA-Z' -]+)*)".format(
#         "|".join(patterns)
#     )

#     # Compile the regex with case insensitivity
#     regex = re.compile(name_pattern, re.IGNORECASE)

#     # Search for the pattern in the input text
#     match = regex.search(text)

#     if match:
#         return match.group(1).strip()

#     return None


# def extract_degree_name(input):
#     """
#     Extracts the degree name from the given text.

#     This function uses a regular expression to match common degree abbreviations (e.g., B.A., Ph.D.) and full names (e.g., Bachelor of Science) found in degree certificates.

#     Args:
#     text: The text extracted from the degree certificate image.

#     Returns:
#     The extracted degree name as a string, or None if no degree name is found.
#     """
#     regex = re.compile(
#         r"\b(?:Bachelor|Bachelors|Master|Doctor|Associate|B\.A\.|B\.Sc\.|M\.A\.|M\.Sc\.|Ph\.D\.|M\.B\.A\.|B\.E\.|B\.Tech|M\.E\.|M\.Tech|B\.Com|M\.Com|B\.Ed|M\.Ed|B\.Pharm|M\.Pharm|B\.Arch|M\.Arch|LL\.B|LL\.M|D\.Phil|D\.Lit|BFA|MFA|MRes|MSt)\s*(?:of\s*[A-Za-z]+)?\b",
#         re.IGNORECASE,
#     )
#     match = re.search(regex, input)
#     if match:
#         return match.group(0).strip()
#     return None


# def extract_institution_name(input):
#     """
#     Extracts the institution name (university, college, etc.) from the given text.

#     This function uses a regular expression to match various formats of institution names that might be present in degree certificates. It covers names like "Massachusetts Institute of Technology" or "University of California, Berkeley".

#     Args:
#     text: The text extracted from the degree certificate image.

#     Returns:
#     The extracted institution name as a string, or None if no institution name is found.
#     """
#     regex = re.compile(
#         r"\b(?:College of [A-Za-z\s]+|[A-Z][a-z]*\sInstitute of [A-Za-z]+|(?:UNIVERSITY OF [A-Za-z]+|[w A-Za-z]*\s(University|Aniversity)?))",
#         re.IGNORECASE,
#     )
#     match = re.search(regex, input)
#     if match:
#         return match.group(0).strip()
#     return None


# def extract_year_of_passing(input):
#     """

#     This function uses a regular expression to search for common patterns indicating the year of passing in degree certificates, such as "year of passing" or "in the year".

#     Args:
#     text: The text extracted from the degree certificate image.

#     Returns:
#     The extracted year of passing as a string, or None if no year of passing is found.
#     """
#     regex = re.compile(
#         r"\b(nineteen (hundred|hundred and) (one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|twenty[- ]one|twenty[- ]two|twenty[- ]three|twenty[- ]four|twenty[- ]five|twenty[- ]six|twenty[- ]seven|twenty[- ]eight|twenty[- ]nine|thirty|forty|fifty|sixty|seventy|eighty|ninety)([- ](one|two|three|four|five|six|seven|eight|nine))?|\d{4}|(two|too|tfoo|tw)\s*(thousand|thousand and)\s*(one|two|three|four|five|six|seven|eight|nine|ten|tex|eleven|twelve|thirteen|fourteen|fifteen|fiventy|sixteen|seventeen|eighteen|nineteen|twenty|twenty[- ]one|twenty[- ]two|twenty[- ]three|twenty[- ]four|twenty[- ]five|twenty[- ]six|twenty[- ]seven|twenty[- ]eight|twenty[- ]nine))\b",
#         re.IGNORECASE,
#     )
#     match = re.search(regex, input)
#     if match:
#         return match.group(1)
#     return None


# def check_image_quality(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#     variance_of_laplacian = cv2.Laplacian(image, cv2.CV_64F).var()

#     mean_brightness = image.mean()

#     return variance_of_laplacian > 50 and mean_brightness > 50


# def preprocess_image(image_path):
#     gray = cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY)

#     gray = cv2.equalizeHist(gray)

#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))

#     processed = cv2.dilate(gray, kernel, iterations=1)

#     processed = cv2.erode(processed, kernel, iterations=1)

#     sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

#     sharpened = cv2.filter2D(processed, -1, sharpen_kernel)

#     return sharpened


# def parse_degree_certificate(image_path):
#     """
#     Parses information from a degree certificate image.

#     This function takes the path to a degree certificate image and attempts to extract the following information using regular expressions and Tesseract OCR:

#         * Recipient's Name
#         * Degree Name
#         * University Name
#         * Year of Passing

#     Args:
#         image_path (str): The path to the degree certificate image file.

#     Returns:
#         dict: A dictionary containing the extracted information with keys "Name", "Degree Name", "University Name", and "Year of Passing". The values can be None if the corresponding information is not found in the image.
#     """
#     if not check_image_quality(image_path):
#         return "Image quality is too low to process."

#     image = cv2.imread(image_path)
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     preprocessed_image = preprocess_image(image_path)
#     extracted_text = pytesseract.image_to_string(gray_image, output_type=Output.STRING)

#     degree_info = {
#         "Name": extract_name(extracted_text),
#         "Degree Name": extract_degree_name(extracted_text),
#         "University Name": extract_institution_name(extracted_text),
#         "Year of Passing": extract_year_of_passing(extracted_text),
#     }

#     return degree_info


# def degree(image_path):
#     """
#     Convenience function to parse degree certificate information.

#     This function simply calls `parse_degree_certificate` and returns the resulting dictionary.

#     Args:
#         image_path (str): The path to the degree certificate image file.

#     Returns:
#         dict: A dictionary containing the extracted information from the degree certificate (same as the output of `parse_degree_certificate`).
#     """
#     return parse_degree_certificate(image_path)


# if __name__ == "__main__":
#     parse_degree_certificate("/home/manasvi/projects/openbharatocr/faltu/degree/50.jpg")
























# # # ####################### with multi lang and paddleocr##################
# import re
# import cv2
# import numpy as np
# from paddleocr import PaddleOCR
# from typing import Optional, Dict, Union

# # Initialize PaddleOCR once (English only, adjust lang as needed)
# print("[DEBUG] Initializing PaddleOCR model...")
# ocr_model = PaddleOCR(use_angle_cls=True, lang="en")
# print("[DEBUG] PaddleOCR initialized.")


# def preprocess_image(image_path):
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"Could not read image at {image_path}")

#     # Upscale small images
#     height, width = image.shape[:2]
#     if max(height, width) < 1000:
#         scale = 1000 / max(height, width)
#         image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = cv2.equalizeHist(gray)

#     # Minimal sharpening only
#     sharpen_kernel = np.array([[-1, -1, -1],
#                                [-1,  9, -1],
#                                [-1, -1, -1]])
#     sharpened = cv2.filter2D(gray, -1, sharpen_kernel)

#     # Convert back to 3 channels for PaddleOCR
#     return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


# def check_image_quality(image_path: str, threshold: float = 100.0) -> bool:
#     print(f"[DEBUG] Checking image quality for: {image_path}")
#     image = cv2.imread(image_path)
#     if image is None:
#         print("[ERROR] Could not read image for quality check.")
#         return False
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     variance = cv2.Laplacian(gray, cv2.CV_64F).var()
#     print(f"[DEBUG] Variance of Laplacian: {variance}")
#     is_clear = variance > threshold
#     print(f"[DEBUG] Image is {'clear' if is_clear else 'blurry'} based on threshold {threshold}")
#     return is_clear


# def extract_name(text: str) -> Optional[str]:
#     print("[DEBUG] Extracting name from text...")
#     patterns = [
#         r"conferred on",
#         r"conferred upon",
#         r"awarded to",
#         r"certify that",
#         r"certifies that",
#         r"testify that",
#         r"known that",
#         r"admits",
#         r"granted",
#     ]
#     name_pattern = r"(?:{})\s+([A-Z][a-zA-Z' -]+(?:\s[A-Z][a-zA-Z' -]+)*)".format("|".join(patterns))
#     match = re.search(name_pattern, text, re.IGNORECASE)
#     if match:
#         print(f"[DEBUG] Name found: {match.group(1).strip()}")
#         return match.group(1).strip()
#     print("[DEBUG] Name not found.")
#     return None


# def extract_degree_name(text: str) -> Optional[str]:
#     print("[DEBUG] Extracting degree name from text...")
#     regex = re.compile(
#         r"\b(?:Bachelor|Bachelors|Master|Doctor|Associate|B\.A\.|B\.Sc\.|M\.A\.|M\.Sc\.|Ph\.D\.|M\.B\.A\.|B\.E\.|B\.Tech|M\.E\.|M\.Tech|B\.Com|M\.Com|B\.Ed|M\.Ed|B\.Pharm|M\.Pharm|B\.Arch|M\.Arch|LL\.B|LL\.M|D\.Phil|D\.Lit|BFA|MFA|MRes|MSt)\s*(?:of\s*[A-Za-z]+)?\b",
#         re.IGNORECASE
#     )
#     match = regex.search(text)
#     if match:
#         print(f"[DEBUG] Degree name found: {match.group(0).strip()}")
#         return match.group(0).strip()
#     print("[DEBUG] Degree name not found.")
#     return None


# def extract_institution_name(text: str) -> Optional[str]:
#     print("[DEBUG] Extracting institution name from text...")
#     regex = re.compile(
#         r"\b(?:College of [A-Za-z\s]+|[A-Z][a-z]*\sInstitute of [A-Za-z]+|(?:UNIVERSITY OF [A-Za-z]+|[w A-Za-z]*\s(?:University|Aniversity)?))",
#         re.IGNORECASE
#     )
#     match = regex.search(text)
#     if match:
#         print(f"[DEBUG] Institution found: {match.group(0).strip()}")
#         return match.group(0).strip()
#     print("[DEBUG] Institution not found.")
#     return None


# def extract_year_of_passing(text: str) -> Optional[str]:
#     print("[DEBUG] Extracting year of passing from text...")
#     regex = re.compile(
#         r"\b(\d{4}|nineteen\s(?:hundred|hundred and)?\s*\w+|two\s*thousand\s*\w*)\b",
#         re.IGNORECASE
#     )
#     match = regex.search(text)
#     if match:
#         print(f"[DEBUG] Year of passing found: {match.group(1)}")
#         return match.group(1)
#     print("[DEBUG] Year of passing not found.")
#     return None


# def parse_degree_certificate(image_path: str) -> Union[str, Dict[str, Optional[str]]]:
#     print(f"[DEBUG] Parsing degree certificate: {image_path}")
#     if not check_image_quality(image_path):
#         print("[ERROR] Image quality too low.")
#         return "Image quality is too low to process."

#     preprocessed_image = preprocess_image(image_path)
#     print("[DEBUG] Preprocessing complete. Running OCR...")

#     results = ocr_model.predict(preprocessed_image)
#     print("[DEBUG] OCR complete.")
#     print(f"[DEBUG] Raw OCR results: {results}")

#     extracted_text = " ".join([line[1][0] for page in results for line in page])

#     print(f"[DEBUG] Extracted text: {extracted_text}")

#     degree_info = {
#         "Name": extract_name(extracted_text),
#         "Degree Name": extract_degree_name(extracted_text),
#         "University Name": extract_institution_name(extracted_text),
#         "Year of Passing": extract_year_of_passing(extracted_text),
#     }
#     print(f"[DEBUG] Extracted degree info: {degree_info}")

#     return degree_info


# def degree(image_path: str) -> Union[str, Dict[str, Optional[str]]]:
#     print(f"[DEBUG] Starting degree extraction for: {image_path}")
#     return parse_degree_certificate(image_path)


# if __name__ == "__main__":
#     image_path = "/home/manasvi/projects/openbharatocr/faltu/degree/50.jpg"
#     result = degree(image_path)
#     print("[DEBUG] Final result:")
#     print(result)






























# ################## diffrent approch ######################
# # import cv2
# # import numpy as np
# # import re
# # from PIL import Image, ImageEnhance, ImageFilter
# # import requests
# # import base64
# # import json
# # import os
# # from io import BytesIO
# # import matplotlib.pyplot as plt

# # # For offline models
# # try:
# #     import torch
# #     import torchvision.transforms as transforms
# #     from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# #     TROCR_AVAILABLE = True
# # except ImportError:
# #     TROCR_AVAILABLE = False

# # try:
# #     import keras_ocr
# #     KERAS_OCR_AVAILABLE = True
# # except ImportError:
# #     KERAS_OCR_AVAILABLE = False

# # try:
# #     from doctr.io import DocumentFile
# #     from doctr.models import ocr_predictor
# #     DOCTR_AVAILABLE = True
# # except ImportError:
# #     DOCTR_AVAILABLE = False

# # try:
# #     import easyocr
# #     EASYOCR_AVAILABLE = True
# # except ImportError:
# #     EASYOCR_AVAILABLE = False

# # class AdvancedDegreeParser:
# #     def __init__(self):
# #         print("Initializing OCR models...")
# #         self.ocr_engines = {}
        
# #         # Initialize available OCR engines
# #         self._init_easyocr()
# #         self._init_keras_ocr()
# #         self._init_trocr()
# #         self._init_doctr()
        
# #         print(f"Available OCR engines: {list(self.ocr_engines.keys())}")
    
# #     def _init_easyocr(self):
# #         """Initialize EasyOCR"""
# #         if EASYOCR_AVAILABLE:
# #             try:
# #                 self.ocr_engines['easyocr'] = easyocr.Reader(['en'], gpu=False)
# #                 print("✓ EasyOCR initialized")
# #             except Exception as e:
# #                 print(f"✗ EasyOCR failed to initialize: {e}")
    
# #     def _init_keras_ocr(self):
# #         """Initialize Keras OCR"""
# #         if KERAS_OCR_AVAILABLE:
# #             try:
# #                 self.ocr_engines['keras_ocr'] = keras_ocr.pipeline.Pipeline()
# #                 print("✓ Keras OCR initialized")
# #             except Exception as e:
# #                 print(f"✗ Keras OCR failed to initialize: {e}")
    
# #     def _init_trocr(self):
# #         """Initialize TrOCR (Transformer OCR)"""
# #         if TROCR_AVAILABLE:
# #             try:
# #                 self.trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
# #                 self.trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
# #                 self.ocr_engines['trocr'] = True
# #                 print("✓ TrOCR initialized")
# #             except Exception as e:
# #                 print(f"✗ TrOCR failed to initialize: {e}")
    
# #     def _init_doctr(self):
# #         """Initialize DocTR"""
# #         if DOCTR_AVAILABLE:
# #             try:
# #                 self.ocr_engines['doctr'] = ocr_predictor(pretrained=True)
# #                 print("✓ DocTR initialized")
# #             except Exception as e:
# #                 print(f"✗ DocTR failed to initialize: {e}")
    
# #     def preprocess_image_advanced(self, image_path):
# #         """
# #         Advanced image preprocessing with multiple techniques
# #         """
# #         image = cv2.imread(image_path)
# #         if image is None:
# #             raise ValueError(f"Could not read image from {image_path}")
        
# #         processed_images = {}
        
# #         # Method 1: Standard grayscale with denoising
# #         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# #         denoised = cv2.fastNlMeansDenoising(gray)
# #         processed_images['denoised'] = denoised
        
# #         # Method 2: CLAHE + Gaussian blur
# #         clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
# #         clahe_img = clahe.apply(gray)
# #         blurred = cv2.GaussianBlur(clahe_img, (3,3), 0)
# #         processed_images['clahe_blur'] = blurred
        
# #         # Method 3: Morphological operations
# #         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
# #         morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
# #         processed_images['morph'] = morph
        
# #         # Method 4: Bilateral filter + adaptive threshold
# #         bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
# #         adaptive = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# #         processed_images['bilateral_adaptive'] = adaptive
        
# #         # Method 5: Unsharp masking
# #         gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
# #         unsharp = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
# #         processed_images['unsharp'] = unsharp
        
# #         # Method 6: Edge enhancement
# #         kernel_edge = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
# #         edge_enhanced = cv2.filter2D(gray, -1, kernel_edge)
# #         processed_images['edge_enhanced'] = edge_enhanced
        
# #         return processed_images, image
    
# #     def extract_text_easyocr(self, image):
# #         """Extract text using EasyOCR"""
# #         if 'easyocr' not in self.ocr_engines:
# #             return ""
        
# #         try:
# #             results = self.ocr_engines['easyocr'].readtext(image, detail=0, paragraph=True)
# #             return ' '.join(results)
# #         except Exception as e:
# #             print(f"EasyOCR extraction failed: {e}")
# #             return ""
    
# #     def extract_text_keras_ocr(self, image):
# #         """Extract text using Keras OCR"""
# #         if 'keras_ocr' not in self.ocr_engines:
# #             return ""
        
# #         try:
# #             # Convert to RGB if needed
# #             if len(image.shape) == 3:
# #                 image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# #             else:
# #                 image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
# #             prediction_groups = self.ocr_engines['keras_ocr'].recognize([image_rgb])
# #             extracted_text = ' '.join([text for text, box in prediction_groups[0]])
# #             return extracted_text
# #         except Exception as e:
# #             print(f"Keras OCR extraction failed: {e}")
# #             return ""
    
# #     def extract_text_trocr(self, image):
# #         """Extract text using TrOCR"""
# #         if 'trocr' not in self.ocr_engines:
# #             return ""
        
# #         try:
# #             # Convert to PIL Image
# #             if len(image.shape) == 3:
# #                 pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# #             else:
# #                 pil_image = Image.fromarray(image).convert('RGB')
            
# #             # Process image
# #             pixel_values = self.trocr_processor(images=pil_image, return_tensors="pt").pixel_values
# #             generated_ids = self.trocr_model.generate(pixel_values)
# #             generated_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
# #             return generated_text
# #         except Exception as e:
# #             print(f"TrOCR extraction failed: {e}")
# #             return ""
    
# #     def extract_text_doctr(self, image):
# #         """Extract text using DocTR"""
# #         if 'doctr' not in self.ocr_engines:
# #             return ""
        
# #         try:
# #             # Save image temporarily
# #             temp_path = "temp_image.jpg"
# #             cv2.imwrite(temp_path, image)
            
# #             # Process with DocTR
# #             doc = DocumentFile.from_images([temp_path])
# #             result = self.ocr_engines['doctr'](doc)
            
# #             # Extract text
# #             extracted_text = ""
# #             for page in result.pages:
# #                 for block in page.blocks:
# #                     for line in block.lines:
# #                         for word in line.words:
# #                             extracted_text += word.value + " "
# #                         extracted_text += "\n"
            
# #             # Clean up
# #             if os.path.exists(temp_path):
# #                 os.remove(temp_path)
            
# #             return extracted_text
# #         except Exception as e:
# #             print(f"DocTR extraction failed: {e}")
# #             return ""
    
# #     def extract_text_google_vision(self, image_path):
# #         """
# #         Extract text using Google Cloud Vision API (if API key available)
# #         """
# #         try:
# #             # This requires Google Cloud Vision API key
# #             # For demo purposes, this is a placeholder
# #             # You would need to set up Google Cloud credentials
            
# #             from google.cloud import vision
# #             client = vision.ImageAnnotatorClient()
            
# #             with open(image_path, 'rb') as image_file:
# #                 content = image_file.read()
            
# #             image = vision.Image(content=content)
# #             response = client.text_detection(image=image)
# #             texts = response.text_annotations
            
# #             if texts:
# #                 return texts[0].description
# #             return ""
            
# #         except Exception as e:
# #             print(f"Google Vision API not available or failed: {e}")
# #             return ""
    
# #     def extract_text_azure_ocr(self, image_path):
# #         """
# #         Extract text using Azure Computer Vision API (if API key available)
# #         """
# #         try:
# #             # This requires Azure Cognitive Services API key
# #             # Placeholder implementation
            
# #             import requests
            
# #             # You would need to set these
# #             endpoint = "YOUR_AZURE_ENDPOINT"
# #             subscription_key = "YOUR_SUBSCRIPTION_KEY"
            
# #             ocr_url = endpoint + "vision/v3.2/ocr"
            
# #             with open(image_path, 'rb') as image_file:
# #                 image_data = image_file.read()
            
# #             headers = {
# #                 'Ocp-Apim-Subscription-Key': subscription_key,
# #                 'Content-Type': 'application/octet-stream'
# #             }
            
# #             params = {'language': 'unk', 'detectOrientation': 'true'}
            
# #             response = requests.post(ocr_url, headers=headers, params=params, data=image_data)
# #             response.raise_for_status()
            
# #             analysis = response.json()
            
# #             extracted_text = ""
# #             for region in analysis['regions']:
# #                 for line in region['lines']:
# #                     for word in line['words']:
# #                         extracted_text += word['text'] + " "
# #                     extracted_text += "\n"
            
# #             return extracted_text
            
# #         except Exception as e:
# #             print(f"Azure OCR not available or failed: {e}")
# #             return ""
    
# #     def extract_all_texts(self, image_path):
# #         """
# #         Extract text using all available methods
# #         """
# #         processed_images, original = self.preprocess_image_advanced(image_path)
# #         all_extractions = {}
        
# #         print("\nExtracting text using multiple OCR engines...")
        
# #         # Test each preprocessing method with each OCR engine
# #         for preprocess_name, processed_img in processed_images.items():
# #             print(f"\nTesting preprocessing method: {preprocess_name}")
            
# #             # EasyOCR
# #             if 'easyocr' in self.ocr_engines:
# #                 text = self.extract_text_easyocr(processed_img)
# #                 all_extractions[f'easyocr_{preprocess_name}'] = text
# #                 print(f"  EasyOCR: {len(text)} characters extracted")
            
# #             # Keras OCR
# #             if 'keras_ocr' in self.ocr_engines:
# #                 text = self.extract_text_keras_ocr(processed_img)
# #                 all_extractions[f'keras_ocr_{preprocess_name}'] = text
# #                 print(f"  Keras OCR: {len(text)} characters extracted")
            
# #             # TrOCR
# #             if 'trocr' in self.ocr_engines:
# #                 text = self.extract_text_trocr(processed_img)
# #                 all_extractions[f'trocr_{preprocess_name}'] = text
# #                 print(f"  TrOCR: {len(text)} characters extracted")
            
# #             # DocTR
# #             if 'doctr' in self.ocr_engines:
# #                 text = self.extract_text_doctr(processed_img)
# #                 all_extractions[f'doctr_{preprocess_name}'] = text
# #                 print(f"  DocTR: {len(text)} characters extracted")
        
# #         # Also try cloud APIs if available
# #         google_text = self.extract_text_google_vision(image_path)
# #         if google_text:
# #             all_extractions['google_vision'] = google_text
        
# #         azure_text = self.extract_text_azure_ocr(image_path)
# #         if azure_text:
# #             all_extractions['azure_ocr'] = azure_text
        
# #         return all_extractions
    
# #     def extract_name_advanced(self, text):
# #         """
# #         Advanced name extraction with multiple patterns
# #         """
# #         patterns = [
# #             r"(?:conferred\s+(?:on|upon)|awarded\s+to|granted\s+to)\s+([A-Z][a-zA-Z'\s.-]+(?:\s[A-Z][a-zA-Z'\s.-]+)*)",
# #             r"(?:This\s+is\s+to\s+certify\s+that)\s+([A-Z][a-zA-Z'\s.-]+(?:\s[A-Z][a-zA-Z'\s.-]+)*)",
# #             r"(?:certify\s+that|certifies\s+that)\s+([A-Z][a-zA-Z'\s.-]+(?:\s[A-Z][a-zA-Z'\s.-]+)*)",
# #             r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*(?:has\s+successfully\s+completed|has\s+completed)",
# #             r"(?:Mr\.|Ms\.|Mrs\.|Dr\.|Miss)\s+([A-Z][a-zA-Z'\s.-]+(?:\s[A-Z][a-zA-Z'\s.-]+)*)",
# #             r"(?:presented\s+to|diploma\s+to)\s+([A-Z][a-zA-Z'\s.-]+(?:\s[A-Z][a-zA-Z'\s.-]+)*)",
# #             # More flexible pattern
# #             r"\b([A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)\b(?=\s+(?:has|is|was|for))",
# #         ]
        
# #         for pattern in patterns:
# #             matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
# #             for match in matches:
# #                 name = match.group(1).strip()
# #                 # Clean up
# #                 name = re.sub(r'\s+', ' ', name)
# #                 # Validate name (should be reasonable length and not contain numbers)
# #                 if 3 <= len(name) <= 50 and not any(char.isdigit() for char in name):
# #                     # Additional validation to avoid false positives
# #                     if not any(word.lower() in name.lower() for word in ['university', 'institute', 'college', 'degree', 'bachelor', 'master', 'doctor']):
# #                         return name
        
# #         return None
    
# #     def extract_degree_advanced(self, text):
# #         """
# #         Advanced degree extraction
# #         """
# #         patterns = [
# #             r"\b(?:Master\s+of\s+Business\s+Administration|MBA|M\.B\.A\.)\b",
# #             r"\b(?:Bachelor\s+of\s+[A-Za-z\s&]+|Master\s+of\s+[A-Za-z\s&]+|Doctor\s+of\s+[A-Za-z\s&]+)\b",
# #             r"\b(?:B\.?(?:A|Sc|E|Tech|Com|Ed|Pharm|Arch|FA)|M\.?(?:A|Sc|E|Tech|Com|Ed|Pharm|Arch|FA|BA|Res|St|Phil)|Ph\.?D|D\.?(?:Phil|Lit)|LL\.?[BM])\b",
# #             r"\b(?:Diploma\s+in\s+[A-Za-z\s&]+|Certificate\s+in\s+[A-Za-z\s&]+)\b",
# #             # Post Graduate programs
# #             r"\b(?:Post\s+Graduate\s+(?:Diploma|Certificate|Programme)\s+in\s+[A-Za-z\s&]+|PGDM|PGP)\b"
# #         ]
        
# #         for pattern in patterns:
# #             match = re.search(pattern, text, re.IGNORECASE)
# #             if match:
# #                 degree = match.group(0).strip()
# #                 return degree
        
# #         return None
    
# #     def extract_institution_advanced(self, text):
# #         """
# #         Advanced institution extraction
# #         """
# #         patterns = [
# #             r"\b(?:Indian\s+Institute\s+of\s+Management|IIM)\s*(?:[,-]?\s*([A-Za-z]+))?\b",
# #             r"\b(?:Indian\s+Institute\s+of\s+Technology|IIT)\s*(?:[,-]?\s*([A-Za-z]+))?\b",
# #             r"\b(?:[A-Za-z\s]+\s+(?:University|Institute|College|School))\b",
# #             r"\b(?:BITS|ISB|XLRI|IISC|NIT)\s*(?:[A-Za-z]+)?\b",
# #             # General pattern for educational institutions
# #             r"\b(?:[A-Z][a-zA-Z]*\s+)*(?:University|Institute|College|School|Academy)(?:\s+of\s+[A-Za-z\s]+)?\b"
# #         ]
        
# #         for pattern in patterns:
# #             match = re.search(pattern, text, re.IGNORECASE)
# #             if match:
# #                 institution = match.group(0).strip()
# #                 if len(institution) > 3:
# #                     return institution
        
# #         return None
    
# #     def extract_year_advanced(self, text):
# #         """
# #         Advanced year extraction
# #         """
# #         patterns = [
# #             r"\b(20[0-2][0-9])\b",
# #             r"\b(19[89][0-9])\b",
# #             r"(?:year|in|during|class\s+of|batch\s+of|graduated\s+in)\s*[:-]?\s*(20[0-2][0-9])",
# #             r"(?:session|academic\s+year)\s*[:-]?\s*(20[0-2][0-9])",
# #             # Date patterns
# #             r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}[,.]?\s+(20[0-2][0-9])\b"
# #         ]
        
# #         for pattern in patterns:
# #             match = re.search(pattern, text, re.IGNORECASE)
# #             if match:
# #                 year = match.group(1)
# #                 if year and 1950 <= int(year) <= 2030:
# #                     return year
        
# #         return None
    
# #     def parse_certificate(self, image_path):
# #         """
# #         Main parsing function
# #         """
# #         print("Starting advanced certificate parsing...")
        
# #         # Extract text using all methods
# #         all_texts = self.extract_all_texts(image_path)
        
# #         # Combine all texts for comprehensive analysis
# #         combined_text = " ".join(all_texts.values())
        
# #         # Try to extract information
# #         results = {
# #             "Name": None,
# #             "Degree Name": None,
# #             "University Name": None,
# #             "Year of Passing": None
# #         }
        
# #         print("\n" + "="*60)
# #         print("ANALYZING EXTRACTED TEXTS...")
# #         print("="*60)
        
# #         # Show preview of each extraction
# #         for method, text in all_texts.items():
# #             if text and len(text.strip()) > 10:
# #                 print(f"\n--- {method.upper()} ---")
# #                 preview = text.strip()[:200] + "..." if len(text) > 200 else text.strip()
# #                 print(preview)
                
# #                 # Try extraction from this text
# #                 name = self.extract_name_advanced(text)
# #                 degree = self.extract_degree_advanced(text)
# #                 institution = self.extract_institution_advanced(text)
# #                 year = self.extract_year_advanced(text)
                
# #                 # Use first successful extraction
# #                 if name and not results["Name"]:
# #                     results["Name"] = name
# #                 if degree and not results["Degree Name"]:
# #                     results["Degree Name"] = degree
# #                 if institution and not results["University Name"]:
# #                     results["University Name"] = institution
# #                 if year and not results["Year of Passing"]:
# #                     results["Year of Passing"] = year
        
# #         # Try combined text as backup
# #         if not all(results.values()):
# #             print("\n--- ANALYZING COMBINED TEXT ---")
# #             if not results["Name"]:
# #                 results["Name"] = self.extract_name_advanced(combined_text)
# #             if not results["Degree Name"]:
# #                 results["Degree Name"] = self.extract_degree_advanced(combined_text)
# #             if not results["University Name"]:
# #                 results["University Name"] = self.extract_institution_advanced(combined_text)
# #             if not results["Year of Passing"]:
# #                 results["Year of Passing"] = self.extract_year_advanced(combined_text)
        
# #         return results

# # # Main function
# # def parse_certificate_advanced(image_path):
# #     """
# #     Advanced certificate parsing function
# #     """
# #     parser = AdvancedDegreeParser()
# #     return parser.parse_certificate(image_path)

# # # Example usage with error handling
# # if __name__ == "__main__":
# #     image_path = "/home/manasvi/projects/openbharatocr/faltu/degree/50.jpg" # Replace with your image path
    
# #     try:
# #         print("Advanced Certificate Parser")
# #         print("=" * 50)
        
# #         results = parse_certificate_advanced(image_path)
        
# #         print("\n" + "="*60)
# #         print("FINAL EXTRACTED INFORMATION:")
# #         print("="*60)
        
# #         for key, value in results.items():
# #             status = "✓" if value else "✗"
# #             print(f"{status} {key}: {value if value else 'Not found'}")
        
# #         print("="*60)
        
# #     except Exception as e:
# #         print(f"Error: {e}")
# #         print("\nPlease ensure you have installed the required packages:")
# #         print("pip install torch torchvision transformers")
# #         print("pip install keras-ocr")
# #         print("pip install python-doctr[torch]")
# #         print("pip install easyocr")
# #         print("pip install opencv-python pillow numpy matplotlib")

# # # Installation guide
# # """
# # INSTALLATION COMMANDS:

# # # Basic packages
# # pip install opencv-python pillow numpy matplotlib

# # # OCR Engines (install what you can):
# # pip install easyocr
# # pip install keras-ocr
# # pip install python-doctr[torch]

# # # For TrOCR (Transformer OCR)
# # pip install torch torchvision
# # pip install transformers

# # # Optional cloud APIs (require API keys):
# # pip install google-cloud-vision
# # pip install azure-cognitiveservices-vision-computervision

# # # If you get errors, try:
# # pip install --upgrade pip
# # conda install pytorch torchvision -c pytorch (for conda users)
# # """













# #########################final try please work ####################################
# # import cv2
# # import numpy as np
# # import re
# # from PIL import Image, ImageEnhance, ImageFilter, ImageOps
# # import matplotlib.pyplot as plt
# # import os

# # # Multiple OCR approach - install what's available
# # OCR_ENGINES = {}

# # # Try to import different OCR libraries
# # try:
# #     import easyocr
# #     OCR_ENGINES['easyocr'] = easyocr.Reader(['en'], gpu=False)
# #     print("✓ EasyOCR loaded")
# # except ImportError:
# #     print("✗ EasyOCR not available - install with: pip install easyocr")

# # try:
# #     import pytesseract
# #     OCR_ENGINES['tesseract'] = pytesseract
# #     print("✓ Tesseract loaded")
# # except ImportError:
# #     print("✗ Tesseract not available - install with: pip install pytesseract")

# # try:
# #     from paddleocr import PaddleOCR
# #     OCR_ENGINES['paddle'] = PaddleOCR(use_angle_cls=True, lang='en')
# #     print("✓ PaddleOCR loaded")
# # except ImportError:
# #     print("✗ PaddleOCR not available - install with: pip install paddlepaddle paddleocr")

# # class CertificateParser:
# #     def __init__(self):
# #         self.available_engines = list(OCR_ENGINES.keys())
# #         print(f"Available OCR engines: {self.available_engines}")
    
# #     def enhance_image_for_ocr(self, image_path):
# #         """
# #         Multiple image enhancement techniques
# #         """
# #         # Read the original image
# #         original = cv2.imread(image_path)
# #         if original is None:
# #             raise ValueError(f"Could not read image: {image_path}")
        
# #         # Convert to PIL for advanced enhancements
# #         pil_image = Image.fromarray(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        
# #         enhanced_images = {}
        
# #         # Method 1: Basic enhancement
# #         enhancer = ImageEnhance.Contrast(pil_image)
# #         contrast_enhanced = enhancer.enhance(1.8)
# #         enhancer = ImageEnhance.Sharpness(contrast_enhanced)
# #         sharp_enhanced = enhancer.enhance(2.0)
# #         enhanced_images['enhanced'] = cv2.cvtColor(np.array(sharp_enhanced), cv2.COLOR_RGB2BGR)
        
# #         # Method 2: Grayscale with adaptive threshold
# #         gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        
# #         # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
# #         clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
# #         clahe_img = clahe.apply(gray)
        
# #         # Denoise
# #         denoised = cv2.fastNlMeansDenoising(clahe_img)
        
# #         # Adaptive threshold
# #         adaptive_thresh = cv2.adaptiveThreshold(
# #             denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
# #         )
# #         enhanced_images['adaptive'] = adaptive_thresh
        
# #         # Method 3: Morphological processing
# #         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
# #         morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
# #         enhanced_images['morph'] = morph
        
# #         # Method 4: High contrast binary
# #         _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# #         enhanced_images['binary'] = binary
        
# #         # Method 5: Edge enhancement
# #         kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
# #         sharpened = cv2.filter2D(gray, -1, kernel_sharp)
# #         enhanced_images['edge_enhanced'] = sharpened
        
# #         return enhanced_images, original
    
# #     def extract_with_easyocr(self, image):
# #         """Extract text using EasyOCR"""
# #         if 'easyocr' not in OCR_ENGINES:
# #             return ""
        
# #         try:
# #             results = OCR_ENGINES['easyocr'].readtext(image, detail=0, paragraph=True)
# #             return ' '.join(results)
# #         except Exception as e:
# #             print(f"EasyOCR failed: {e}")
# #             return ""
    
# #     def extract_with_tesseract(self, image):
# #         """Extract text using Tesseract"""
# #         if 'tesseract' not in OCR_ENGINES:
# #             return ""
        
# #         try:
# #             # Different Tesseract configurations
# #             configs = [
# #                 '--oem 3 --psm 6',  # Assume uniform block of text
# #                 '--oem 3 --psm 4',  # Assume single column of varying sizes
# #                 '--oem 3 --psm 3',  # Fully automatic page segmentation
# #                 '--oem 3 --psm 8',  # Single word
# #                 '--oem 3 --psm 7',  # Single text line
# #             ]
            
# #             best_text = ""
# #             for config in configs:
# #                 try:
# #                     text = OCR_ENGINES['tesseract'].image_to_string(image, config=config)
# #                     if len(text) > len(best_text):
# #                         best_text = text
# #                 except:
# #                     continue
            
# #             return best_text
# #         except Exception as e:
# #             print(f"Tesseract failed: {e}")
# #             return ""
    
# #     def extract_with_paddle(self, image):
# #         """Extract text using PaddleOCR"""
# #         if 'paddle' not in OCR_ENGINES:
# #             return ""
        
# #         try:
# #             results = OCR_ENGINES['paddle'].ocr(image, cls=True)
# #             text_parts = []
            
# #             if results and results[0]:
# #                 for line in results[0]:
# #                     if line and len(line) > 1 and line[1][1] > 0.5:  # Confidence > 0.5
# #                         text_parts.append(line[1][0])
            
# #             return ' '.join(text_parts)
# #         except Exception as e:
# #             print(f"PaddleOCR failed: {e}")
# #             return ""
    
# #     def extract_all_text(self, image_path):
# #         """Extract text using all available methods"""
# #         enhanced_images, original = self.enhance_image_for_ocr(image_path)
        
# #         all_extractions = {}
        
# #         print("\nExtracting text with different methods...")
        
# #         for preprocess_method, processed_img in enhanced_images.items():
# #             print(f"\nProcessing with {preprocess_method} enhancement:")
            
# #             # Try EasyOCR
# #             if 'easyocr' in self.available_engines:
# #                 text = self.extract_with_easyocr(processed_img)
# #                 all_extractions[f'easyocr_{preprocess_method}'] = text
# #                 print(f"  EasyOCR: {len(text)} characters")
            
# #             # Try Tesseract
# #             if 'tesseract' in self.available_engines:
# #                 text = self.extract_with_tesseract(processed_img)
# #                 all_extractions[f'tesseract_{preprocess_method}'] = text
# #                 print(f"  Tesseract: {len(text)} characters")
            
# #             # Try PaddleOCR
# #             if 'paddle' in self.available_engines:
# #                 text = self.extract_with_paddle(processed_img)
# #                 all_extractions[f'paddle_{preprocess_method}'] = text
# #                 print(f"  PaddleOCR: {len(text)} characters")
        
# #         return all_extractions
    
# #     def extract_certificate_info(self, text):
# #         """Extract specific information from certificate text"""
# #         info = {
# #             'Name': None,
# #             'Degree': None,
# #             'Institution': None,
# #             'Year': None
# #         }
        
# #         # Clean the text
# #         text = re.sub(r'\s+', ' ', text.strip())
        
# #         # Extract Name - Multiple patterns for different certificate formats
# #         name_patterns = [
# #             r'(?:certify that|certifies that|conferred upon|awarded to|presented to)\s+([A-Z][a-zA-Z\s.]+?)(?:\s+has|\s+is|\s+for)',
# #             r'([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+has\s+(?:successfully\s+)?completed',
# #             r'(?:Mr\.|Ms\.|Mrs\.|Dr\.)\s+([A-Z][a-zA-Z\s.]+?)(?:\s+has|\s+is)',
# #             r'This\s+is\s+to\s+certify\s+that\s+([A-Z][a-zA-Z\s.]+?)(?:\s+has|\s+is)',
# #         ]
        
# #         for pattern in name_patterns:
# #             match = re.search(pattern, text, re.IGNORECASE)
# #             if match:
# #                 name = match.group(1).strip()
# #                 # Validate name
# #                 if 3 <= len(name) <= 50 and not any(char.isdigit() for char in name):
# #                     info['Name'] = name
# #                     break
        
# #         # Extract Degree
# #         degree_patterns = [
# #             r'Master\s+of\s+Business\s+Administration',
# #             r'M\.?B\.?A\.?',
# #             r'Bachelor\s+of\s+[A-Za-z\s]+',
# #             r'Master\s+of\s+[A-Za-z\s]+',
# #             r'Doctor\s+of\s+[A-Za-z\s]+',
# #             r'B\.?[A-Z][a-z]*|M\.?[A-Z][a-z]*|Ph\.?D',
# #             r'Post\s+Graduate\s+(?:Diploma|Programme|Certificate)',
# #             r'PGDM|PGP'
# #         ]
        
# #         for pattern in degree_patterns:
# #             match = re.search(pattern, text, re.IGNORECASE)
# #             if match:
# #                 info['Degree'] = match.group(0).strip()
# #                 break
        
# #         # Extract Institution
# #         institution_patterns = [
# #             r'Indian\s+Institute\s+of\s+Management[,\s]*([A-Za-z]*)',
# #             r'IIM[,\s]*([A-Za-z]*)',
# #             r'[A-Za-z\s]+University',
# #             r'[A-Za-z\s]+Institute(?:\s+of\s+[A-Za-z\s]+)?',
# #             r'[A-Za-z\s]+College'
# #         ]
        
# #         for pattern in institution_patterns:
# #             match = re.search(pattern, text, re.IGNORECASE)
# #             if match:
# #                 institution = match.group(0).strip()
# #                 if len(institution) > 3:
# #                     info['Institution'] = institution
# #                     break
        
# #         # Extract Year
# #         year_patterns = [
# #             r'\b(20[0-2][0-9])\b',
# #             r'\b(19[89][0-9])\b',
# #             r'(?:year|class|batch)\s+(?:of\s+)?(20[0-2][0-9])',
# #             r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+(20[0-2][0-9])'
# #         ]
        
# #         for pattern in year_patterns:
# #             match = re.search(pattern, text, re.IGNORECASE)
# #             if match:
# #                 year = match.group(1)
# #                 if 1980 <= int(year) <= 2030:
# #                     info['Year'] = year
# #                     break
        
# #         return info
    
# #     def parse_certificate(self, image_path):
# #         """Main function to parse certificate"""
# #         if not self.available_engines:
# #             return "No OCR engines available. Please install at least one OCR library."
        
# #         print(f"Parsing certificate: {image_path}")
        
# #         # Extract all text using different methods
# #         all_extractions = self.extract_all_text(image_path)
        
# #         # Combine all extracted texts
# #         combined_text = ""
# #         best_extraction = ""
# #         max_length = 0
        
# #         print("\n" + "="*60)
# #         print("TEXT EXTRACTION RESULTS:")
# #         print("="*60)
        
# #         for method, text in all_extractions.items():
# #             if text and len(text.strip()) > 20:
# #                 print(f"\n--- {method.upper()} ---")
# #                 preview = text.strip()[:300] + "..." if len(text) > 300 else text.strip()
# #                 print(preview)
                
# #                 combined_text += " " + text
                
# #                 # Keep track of the best extraction (longest meaningful text)
# #                 if len(text) > max_length:
# #                     max_length = len(text)
# #                     best_extraction = text
        
# #         # Extract structured information
# #         print(f"\n{'-'*60}")
# #         print("EXTRACTING STRUCTURED INFORMATION...")
# #         print(f"{'-'*60}")
        
# #         # Try to extract from the best single extraction first
# #         info = self.extract_certificate_info(best_extraction)
        
# #         # If some fields are missing, try the combined text
# #         combined_info = self.extract_certificate_info(combined_text)
# #         for key, value in combined_info.items():
# #             if not info[key] and value:
# #                 info[key] = value
        
# #         return {
# #             'extracted_info': info,
# #             'full_text': combined_text.strip(),
# #             'best_extraction': best_extraction.strip()
# #         }

# # # Easy-to-use function
# # def parse_certificate_image(image_path):
# #     """
# #     Simple function to parse certificate image
# #     """
# #     parser = CertificateParser()
# #     return parser.parse_certificate(image_path)

# # # Example usage
# # if __name__ == "__main__":
# #     # Replace with your image path
# #     image_path = "/home/manasvi/projects/openbharatocr/faltu/degree/50.jpg"  # Update this path
    
# #     try:
# #         results = parse_certificate_image(image_path)
        
# #         print("\n" + "="*70)
# #         print("FINAL RESULTS:")
# #         print("="*70)
        
# #         info = results['extracted_info']
        
# #         print("\n📋 CERTIFICATE INFORMATION:")
# #         print("-" * 40)
# #         for key, value in info.items():
# #             status = "✓" if value else "✗"
# #             print(f"{status} {key:<12}: {value or 'Not found'}")
        
# #         print(f"\n📄 FULL EXTRACTED TEXT:")
# #         print("-" * 40)
# #         print(results['full_text'][:500] + "..." if len(results['full_text']) > 500 else results['full_text'])
        
# #         print("\n" + "="*70)
        
# #     except FileNotFoundError:
# #         print("❌ Image file not found. Please check the file path.")
# #     except Exception as e:
# #         print(f"❌ Error: {e}")
# #         print("\n💡 Make sure you have at least one OCR library installed:")
# #         print("   pip install easyocr")
# #         print("   pip install pytesseract")
# #         print("   pip install paddlepaddle paddleocr")

# # # Installation instructions
# # """
# # QUICK INSTALL (choose one or more):

# # 1. EasyOCR (Recommended - works well with certificates):
# #    pip install easyocr

# # 2. Tesseract:
# #    pip install pytesseract
# #    # Also install Tesseract binary:
# #    # Windows: Download from https://github.com/tesseract-ocr/tesseract
# #    # Mac: brew install tesseract
# #    # Linux: sudo apt-get install tesseract-ocr

# # 3. PaddleOCR:
# #    pip install paddlepaddle paddleocr

# # 4. Basic requirements:
# #    pip install opencv-python pillow numpy matplotlib

# # USAGE:
# # 1. Save your certificate image as 'certificate.jpg'
# # 2. Run: python script_name.py
# # 3. Or use: parse_certificate_image("path/to/your/image.jpg")
# # """





################### only doctr ###############################


# import re
# import cv2
# import numpy as np
# from doctr.io import DocumentFile
# from doctr.models import ocr_predictor
# from typing import Optional, Dict, Union

# # Initialize docTR OCR model (detection + recognition)
# print("[DEBUG] Initializing docTR OCR model...")
# ocr_model = ocr_predictor(det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True)
# print("[DEBUG] docTR initialized.")


# def preprocess_image(image_path):
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"Could not read image at {image_path}")

#     # Upscale small images
#     height, width = image.shape[:2]
#     if max(height, width) < 1000:
#         scale = 1000 / max(height, width)
#         image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = cv2.equalizeHist(gray)

#     # Minimal sharpening
#     sharpen_kernel = np.array([[-1, -1, -1],
#                                [-1,  9, -1],
#                                [-1, -1, -1]])
#     sharpened = cv2.filter2D(gray, -1, sharpen_kernel)

#     # Convert back to 3 channels for docTR
#     return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


# def check_image_quality(image_path: str, threshold: float = 100.0) -> bool:
#     print(f"[DEBUG] Checking image quality for: {image_path}")
#     image = cv2.imread(image_path)
#     if image is None:
#         print("[ERROR] Could not read image for quality check.")
#         return False
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     variance = cv2.Laplacian(gray, cv2.CV_64F).var()
#     print(f"[DEBUG] Variance of Laplacian: {variance}")
#     return variance > threshold


# def extract_name(text: str) -> Optional[str]:
#     print("[DEBUG] Extracting name from text...")
#     patterns = [
#         r"conferred on",
#         r"conferred upon",
#         r"awarded to",
#         r"certify that",
#         r"certifies that",
#         r"testify that",
#         r"known that",
#         r"admits",
#         r"granted",
#     ]
#     name_pattern = r"(?:{})\s+([A-Z][a-zA-Z' -]+(?:\s[A-Z][a-zA-Z' -]+)*)".format("|".join(patterns))
#     match = re.search(name_pattern, text, re.IGNORECASE)
#     return match.group(1).strip() if match else None


# def extract_degree_name(text: str) -> Optional[str]:
#     print("[DEBUG] Extracting degree name from text...")
#     regex = re.compile(
#         r"\b(?:Bachelor|Bachelors|Master|Doctor|Associate|B\.A\.|B\.Sc\.|M\.A\.|M\.Sc\.|Ph\.D\.|M\.B\.A\.|B\.E\.|B\.Tech|M\.E\.|M\.Tech|B\.Com|M\.Com|B\.Ed|M\.Ed|B\.Pharm|M\.Pharm|B\.Arch|M\.Arch|LL\.B|LL\.M|D\.Phil|D\.Lit|BFA|MFA|MRes|MSt)\s*(?:of\s*[A-Za-z]+)?\b",
#         re.IGNORECASE
#     )
#     match = regex.search(text)
#     return match.group(0).strip() if match else None


# def extract_institution_name(text: str) -> Optional[str]:
#     print("[DEBUG] Extracting institution name from text...")
#     regex = re.compile(
#         r"\b(?:College of [A-Za-z\s]+|[A-Z][a-z]*\sInstitute of [A-Za-z]+|(?:UNIVERSITY OF [A-Za-z]+|[A-Za-z\s]+ University))",
#         re.IGNORECASE
#     )
#     match = regex.search(text)
#     return match.group(0).strip() if match else None


# def extract_year_of_passing(text: str) -> Optional[str]:
#     print("[DEBUG] Extracting year of passing from text...")
#     regex = re.compile(
#         r"\b(\d{4}|nineteen\s(?:hundred|hundred and)?\s*\w+|two\s*thousand\s*\w*)\b",
#         re.IGNORECASE
#     )
#     match = regex.search(text)
#     return match.group(1) if match else None


# def parse_degree_certificate(image_path: str) -> Union[str, Dict[str, Optional[str]]]:
#     print(f"[DEBUG] Parsing degree certificate: {image_path}")
#     if not check_image_quality(image_path):
#         return "Image quality is too low to process."

#     preprocessed_image = preprocess_image(image_path)
#     print("[DEBUG] Preprocessing complete. Running OCR...")
#     print(type(preprocessed_image))


#     # Run docTR OCR on numpy array (wrap inside list)
#     doc = DocumentFile.from_images(image_path)   # each element = one page
#     result = ocr_model(doc)

#     # Export text
#     exported = result.export()
#     extracted_text = " ".join(
#         [word["value"] for block in exported["pages"][0]["blocks"]
#          for line in block["lines"]
#          for word in line["words"]]
#     )
#     print(f"[DEBUG] Extracted text: {extracted_text}")

#     degree_info = {
#         "Name": extract_name(extracted_text),
#         "Degree Name": extract_degree_name(extracted_text),
#         "University Name": extract_institution_name(extracted_text),
#         "Year of Passing": extract_year_of_passing(extracted_text),
#     }
#     print(f"[DEBUG] Extracted degree info: {degree_info}")
#     return degree_info



# def degree(image_path: str) -> Union[str, Dict[str, Optional[str]]]:
#     print(f"[DEBUG] Starting degree extraction for: {image_path}")
#     return parse_degree_certificate(image_path)


# if __name__ == "__main__":
#     image_path = "/home/manasvi/projects/openbharatocr/faltu/degree/50.jpg"
#     result = degree(image_path)
#     print("[DEBUG] Final result:")
#     print(result)









############################## doctr with clahe and blur ###############################
# import re
# import cv2
# import numpy as np
# from doctr.io import DocumentFile
# from doctr.models import ocr_predictor
# from typing import Optional, Dict, Union

# # Initialize docTR OCR model (detection + recognition)
# print("[DEBUG] Initializing docTR OCR model...")
# ocr_model = ocr_predictor(det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True)
# print("[DEBUG] docTR initialized.")


# def preprocess_image(image_path: str) -> np.ndarray:
#     """Preprocess image with CLAHE + Gaussian blur + sharpening for docTR OCR"""
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"Could not read image at {image_path}")

#     # Upscale small images
#     height, width = image.shape[:2]
#     if max(height, width) < 1000:
#         scale = 1000 / max(height, width)
#         image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # ✅ CLAHE instead of simple histogram equalization
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     clahe_applied = clahe.apply(gray)

#     # ✅ Apply Gaussian blur to reduce noise
#     blurred = cv2.GaussianBlur(clahe_applied, (3, 3), 0)

#     # Minimal sharpening
#     sharpen_kernel = np.array([[-1, -1, -1],
#                                [-1,  9, -1],
#                                [-1, -1, -1]])
#     sharpened = cv2.filter2D(blurred, -1, sharpen_kernel)

#     # Convert back to 3 channels for docTR
#     return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


# def check_image_quality(image_path: str, threshold: float = 100.0) -> bool:
#     print(f"[DEBUG] Checking image quality for: {image_path}")
#     image = cv2.imread(image_path)
#     if image is None:
#         print("[ERROR] Could not read image for quality check.")
#         return False
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     variance = cv2.Laplacian(gray, cv2.CV_64F).var()
#     print(f"[DEBUG] Variance of Laplacian: {variance}")
#     return variance > threshold


# def extract_name(text: str) -> Optional[str]:
#     print("[DEBUG] Extracting name from text...")
#     patterns = [
#         r"conferred on",
#         r"conferred upon",
#         r"awarded to",
#         r"certify that",
#         r"certifies that",
#         r"testify that",
#         r"known that",
#         r"admits",
#         r"granted",
#     ]
#     name_pattern = r"(?:{})\s+([A-Z][a-zA-Z' -]+(?:\s[A-Z][a-zA-Z' -]+)*)".format("|".join(patterns))
#     match = re.search(name_pattern, text, re.IGNORECASE)
#     return match.group(1).strip() if match else None


# def extract_degree_name(text: str) -> Optional[str]:
#     print("[DEBUG] Extracting degree name from text...")
#     regex = re.compile(
#         r"\b(?:Bachelor|Bachelors|Master|Doctor|Associate|B\.A\.|B\.Sc\.|M\.A\.|M\.Sc\.|Ph\.D\.|M\.B\.A\.|B\.E\.|B\.Tech|M\.E\.|M\.Tech|B\.Com|M\.Com|B\.Ed|M\.Ed|B\.Pharm|M\.Pharm|B\.Arch|M\.Arch|LL\.B|LL\.M|D\.Phil|D\.Lit|BFA|MFA|MRes|MSt)\s*(?:of\s*[A-Za-z]+)?\b",
#         re.IGNORECASE
#     )
#     match = regex.search(text)
#     return match.group(0).strip() if match else None


# def extract_institution_name(text: str) -> Optional[str]:
#     print("[DEBUG] Extracting institution name from text...")
#     regex = re.compile(
#         r"\b(?:College of [A-Za-z\s]+|[A-Z][a-z]*\sInstitute of [A-Za-z]+|(?:UNIVERSITY OF [A-Za-z]+|[A-Za-z\s]+ University))",
#         re.IGNORECASE
#     )
#     match = regex.search(text)
#     return match.group(0).strip() if match else None


# def extract_year_of_passing(text: str) -> Optional[str]:
#     print("[DEBUG] Extracting year of passing from text...")
#     regex = re.compile(
#         r"\b(\d{4}|nineteen\s(?:hundred|hundred and)?\s*\w+|two\s*thousand\s*\w*)\b",
#         re.IGNORECASE
#     )
#     match = regex.search(text)
#     return match.group(1) if match else None


# def parse_degree_certificate(image_path: str) -> Union[str, Dict[str, Optional[str]]]:
#     print(f"[DEBUG] Parsing degree certificate: {image_path}")
#     if not check_image_quality(image_path):
#         return "Image quality is too low to process."

#     preprocessed_image = preprocess_image(image_path)
#     print("[DEBUG] Preprocessing complete. Running OCR...")
#     print(f"[DEBUG] Preprocessed image type: {type(preprocessed_image)} shape: {preprocessed_image.shape}")

#     # ✅ Run docTR on CLAHE+blur processed image
#     doc = DocumentFile.from_images(image_path)
#     result = ocr_model(doc)

#     exported = result.export()
#     extracted_text = " ".join(
#         [word["value"] for block in exported["pages"][0]["blocks"]
#          for line in block["lines"]
#          for word in line["words"]]
#     )
#     print(f"[DEBUG] Extracted text: {extracted_text}")

#     degree_info = {
#         "Name": extract_name(extracted_text),
#         "Degree Name": extract_degree_name(extracted_text),
#         "University Name": extract_institution_name(extracted_text),
#         "Year of Passing": extract_year_of_passing(extracted_text),
#     }
#     print(f"[DEBUG] Extracted degree info: {degree_info}")
#     return degree_info


# def degree(image_path: str) -> Union[str, Dict[str, Optional[str]]]:
#     print(f"[DEBUG] Starting degree extraction for: {image_path}")
#     return parse_degree_certificate(image_path)


# if __name__ == "__main__":
#     image_path = "/home/manasvi/projects/openbharatocr/faltu/degree/50.jpg"
#     result = degree(image_path)
#     print("[DEBUG] Final result:")
#     print(result)







import re
import cv2
import numpy as np
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from typing import Optional, Dict, Union
from datetime import datetime

# Initialize docTR OCR model (detection + recognition)
print("[DEBUG] Initializing docTR OCR model...")
ocr_model = ocr_predictor(det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True)
print("[DEBUG] docTR initialized.")


def preprocess_image(image_path: str) -> np.ndarray:
    """Preprocess image with CLAHE + Gaussian blur + sharpening for docTR OCR"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Upscale small images
    height, width = image.shape[:2]
    if max(height, width) < 1000:
        scale = 1000 / max(height, width)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ✅ CLAHE instead of simple histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_applied = clahe.apply(gray)

    # ✅ Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(clahe_applied, (3, 3), 0)

    # Minimal sharpening
    sharpen_kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
    sharpened = cv2.filter2D(blurred, -1, sharpen_kernel)

    # Convert back to 3 channels for docTR
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


def check_image_quality(image_path: str, threshold: float = 100.0) -> bool:
    print(f"[DEBUG] Checking image quality for: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print("[ERROR] Could not read image for quality check.")
        return False
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"[DEBUG] Variance of Laplacian: {variance}")
    return variance > threshold


def extract_name(text: str) -> Optional[str]:
    """
    Extract the name of the recipient from degree certificate text.
    """
    print("[DEBUG] Extracting name from text...")
    # Add "has" and "is" in front of conferred patterns
    patterns = [
        r"has conferred upon",
        r"has conferred on",
        r"is conferred upon",
        r"is conferred on",
        r"conferred on",
        r"conferred upon",
        r"awarded to",
        r"certify that",
        r"certifies that",
        r"testify that",
        r"known that",
        r"admits",
        r"granted",
    ]

    # Regex to capture name after the phrase
    name_pattern = r"(?:{})\s+([A-Z][a-zA-Z]*(?:\s[A-Z][a-zA-Z]*)*)".format("|".join(patterns))

    match = re.search(name_pattern, text, re.IGNORECASE)
    if match:
        name = match.group(1).strip()
        # Remove trailing noise like 'has', 'is', etc.
        name = re.sub(r"\b(?:has|is|been|the)\b$", "", name, flags=re.IGNORECASE).strip()
        return name
    return None


import re
from typing import Optional

def extract_degree_name(text: str) -> Optional[str]:
    """
    Extracts clean degree names like 'Bachelor of Science', 'Master of Arts', 'Doctor of Philosophy'.
    Removes trailing words like 'degree', 'in History', etc.
    """
    print("[DEBUG] Extracting degree name from text...")

    regex = re.compile(
        r"\b("
        r"(?:Bachelor|Bachelors|Master|Masters|Doctor|Doctorate|Associate|Ph\.?D\.?|M\.?B\.?A\.?|B\.?Tech|M\.?Tech|B\.?E\.|M\.?E\.|B\.?Sc\.?|M\.?Sc\.?|B\.?A\.?|M\.?A\.?|B\.?Com|M\.?Com|B\.?Ed|M\.?Ed|B\.?Pharm|M\.?Pharm|B\.?Arch|M\.?Arch|LL\.?B|LL\.?M|D\.Phil|D\.Lit|BFA|MFA|MRes|MSt)"
        r"(?:\s+of\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)?"
        r")",
        re.IGNORECASE,
    )

    match = regex.search(text)
    if match:
        degree = match.group(1).strip()

        # ✅ Remove trailing words after the degree name
        degree = re.sub(
            r"\b(Degree|Honours?|Program|Course|in\s+[A-Z][a-z]+.*)$",
            "",
            degree,
            flags=re.IGNORECASE,
        ).strip()

        # ✅ Capitalize properly but keep "of" lowercase
        words = degree.split()
        normalized = " ".join(
            w.capitalize() if w.lower() != "of" else "of" for w in words
        )
        return normalized

    return None


def extract_institution_name(text: str) -> Optional[str]:
    print("[DEBUG] Extracting institution name from text...")
    regex = re.compile(
        r"\b(?:College of [A-Za-z\s]+|[A-Z][a-z]*\sInstitute of [A-Za-z]+|(?:UNIVERSITY OF [A-Za-z]+|[A-Za-z\s]+ University))",
        re.IGNORECASE
    )
    match = regex.search(text)
    return match.group(0).strip().title() if match else None


def extract_year_of_passing(text: str) -> Optional[str]:
    print("[DEBUG] Extracting year of passing from text...")
    # Find all years
    years = re.findall(r"\b(19\d{2}|20\d{2}|21\d{2})\b", text)
    if not years:
        return None

    current_year = datetime.now().year
    valid_years = [int(y) for y in years if 1950 <= int(y) <= current_year]

    if not valid_years:
        return None

    # Use the latest year (ignores foundation years like 1956)
    return str(max(valid_years))


def parse_degree_certificate(image_path: str) -> Union[str, Dict[str, Optional[str]]]:
    print(f"[DEBUG] Parsing degree certificate: {image_path}")
    if not check_image_quality(image_path):
        return "Image quality is too low to process."

    preprocessed_image = preprocess_image(image_path)
    print("[DEBUG] Preprocessing complete. Running OCR...")
    print(f"[DEBUG] Preprocessed image type: {type(preprocessed_image)} shape: {preprocessed_image.shape}")

    # ✅ Run docTR OCR
    doc = DocumentFile.from_images(image_path)
    result = ocr_model(doc)

    exported = result.export()
    extracted_text = " ".join(
        [word["value"] for block in exported["pages"][0]["blocks"]
         for line in block["lines"]
         for word in line["words"]]
    )
    print(f"[DEBUG] Extracted text: {extracted_text}")

    degree_info = {
        "Name": extract_name(extracted_text),
        "Degree Name": extract_degree_name(extracted_text),
        "University Name": extract_institution_name(extracted_text),
        "Year of Passing": extract_year_of_passing(extracted_text),
    }
    print(f"[DEBUG] Extracted degree info: {degree_info}")
    return degree_info


def degree(image_path: str) -> Union[str, Dict[str, Optional[str]]]:
    print(f"[DEBUG] Starting degree extraction for: {image_path}")
    return parse_degree_certificate(image_path)


if __name__ == "__main__":
    image_path = "/home/manasvi/projects/openbharatocr/faltu/degree/50.jpg"
    result = degree(image_path)
    print("[DEBUG] Final result:")
    print(result)
