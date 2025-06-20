import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image
import re


class AadhaarOCR:
    def __init__(self, use_angle_cls=True, lang='en'):
        """
        Initialize PaddleOCR with optimized settings for Aadhaar cards
        
        Args:
            use_angle_cls: Enable angle classification for rotated text
            lang: Language code ('en' for English, 'hi' for Hindi, ['en', 'hi'] for both)
        """
        # Try different initialization approaches based on PaddleOCR version
        initialization_attempts = [
            # Most basic - just language
            {'lang': lang},
            # With language only (string format)
            {'lang': lang if isinstance(lang, str) else 'en'},
            # Empty initialization
            {}
        ]
        
        # Add angle classification attempts if the basic ones fail
        if use_angle_cls:
            initialization_attempts.insert(0, {'lang': lang, 'use_textline_orientation': True})
            initialization_attempts.insert(1, {'lang': lang, 'use_angle_cls': True})
        
        self.ocr = None
        last_error = None
        
        for i, params in enumerate(initialization_attempts):
            try:
                print(f"Attempting PaddleOCR initialization {i+1}/{len(initialization_attempts)}: {params}")
                self.ocr = PaddleOCR(**params)
                print(f"✓ PaddleOCR initialized successfully with params: {params}")
                break
            except Exception as e:
                last_error = e
                print(f"✗ Failed with params {params}: {str(e)}")
                continue
        
        if self.ocr is None:
            raise RuntimeError(f"Failed to initialize PaddleOCR with any configuration. Last error: {last_error}")
    
    def preprocess_image_enhanced(self, image_path):
        """Enhanced preprocessing for better OCR accuracy"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Noise reduction
        denoised = cv2.medianBlur(enhanced, 3)
        
        # Sharpening kernel
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # Morphological operations to clean up text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        cleaned = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def crop_regions_of_interest(self, image, region_type='back'):
        """Crop specific regions for better OCR accuracy"""
        h, w = image.shape[:2]
        
        if region_type == 'back_top_left':
            # Top-left area for address and father's name
            x_start, y_start = 0, 0
            x_end = int(w * 0.65)
            y_end = int(h * 0.6)
            return image[y_start:y_end, x_start:x_end]
        
        elif region_type == 'front_details':
            # Lower portion for name, DOB, gender
            x_start, y_start = 0, int(h * 0.3)
            x_end, y_end = w, h
            return image[y_start:y_end, x_start:x_end]
        
        elif region_type == 'aadhaar_number':
            # Bottom area for Aadhaar number
            x_start, y_start = 0, int(h * 0.7)
            x_end, y_end = w, h
            return image[y_start:y_end, x_start:x_end]
        
        return image
    
    def extract_text_with_paddle(self, image, confidence_threshold=0.5):
        """Extract text using PaddleOCR with confidence filtering"""
        result = None
        
        try:
            # Try the newer predict method first
            if hasattr(self.ocr, 'predict'):
                print("Using predict() method...")
                if isinstance(image, str):
                    result = self.ocr.predict(image)
                else:
                    result = self.ocr.predict(image)
            else:
                print("Using ocr() method...")
                if isinstance(image, str):
                    result = self.ocr.ocr(image)
                else:
                    result = self.ocr.ocr(image)
                    
            print(f"Raw OCR result type: {type(result)}")
            print(f"Raw OCR result length: {len(result) if result else 'None'}")
            print(f"Raw OCR result: {result}")
            
        except Exception as e:
            print(f"Error in primary OCR method: {e}")
            # Try alternative approaches
            try:
                print("Trying fallback ocr() method...")
                result = self.ocr.ocr(image)
                print(f"Fallback result: {result}")
            except Exception as e2:
                print(f"Fallback method also failed: {e2}")
                return ""
        
        if result is None:
            print("Result is None")
            return ""
            
        if len(result) == 0:
            print("Result is empty")
            return ""
        
        # Debug the result structure
        print(f"Processing result with {len(result)} items")
        
        texts = []
        try:
            # Handle different result formats
            for i, batch in enumerate(result):
                print(f"Batch {i}: type={type(batch)}, content={batch}")
                
                if batch is None:
                    continue
                    
                if isinstance(batch, list):
                    for j, item in enumerate(batch):
                        print(f"  Item {j}: type={type(item)}, content={item}")
                        
                        if item is None:
                            continue
                            
                        # Handle different item structures
                        if isinstance(item, dict):
                            # New format: dict with 'text' and 'confidence' keys
                            if 'text' in item:
                                text = item['text']
                                confidence = item.get('confidence', 1.0)
                                print(f"    Dict format - Text: '{text}', Confidence: {confidence}")
                                if confidence >= confidence_threshold:
                                    texts.append(text)
                        elif isinstance(item, (list, tuple)):
                            print(f"    List/tuple format with {len(item)} elements")
                            if len(item) >= 2:
                                # Try different structures
                                if isinstance(item[1], (list, tuple)) and len(item[1]) >= 2:
                                    # Format: [bbox, [text, confidence]]
                                    text = str(item[1][0])
                                    confidence = float(item[1][1])
                                    print(f"    Nested format - Text: '{text}', Confidence: {confidence}")
                                    if confidence >= confidence_threshold:
                                        texts.append(text)
                                elif isinstance(item[1], str):
                                    # Format: [bbox, text]
                                    text = item[1]
                                    print(f"    Simple format - Text: '{text}'")
                                    texts.append(text)
                                else:
                                    # Try to extract text from second element
                                    text = str(item[1])
                                    print(f"    Fallback format - Text: '{text}'")
                                    texts.append(text)
                        else:
                            # Single text item
                            text = str(item)
                            print(f"    Direct text: '{text}'")
                            texts.append(text)
                else:
                    # Single item, not a list
                    print(f"Single item: {batch}")
                    if isinstance(batch, str):
                        texts.append(batch)
                    else:
                        texts.append(str(batch))
            
        except Exception as e:
            print(f"Error parsing OCR results: {e}")
            import traceback
            traceback.print_exc()
            
            # Last resort: try to extract any readable content
            try:
                print("Attempting last resort text extraction...")
                if isinstance(result, str):
                    texts = [result]
                elif isinstance(result, list):
                    for item in result:
                        if isinstance(item, str):
                            texts.append(item)
                        elif hasattr(item, '__str__'):
                            texts.append(str(item))
            except:
                print("Last resort extraction failed")
                texts = []
        
        # Clean up texts
        cleaned_texts = []
        for text in texts:
            if text and isinstance(text, str) and text.strip():
                cleaned_texts.append(text.strip())
        
        full_text = '\n'.join(cleaned_texts)
        print("----- PADDLE OCR TEXT START -----")
        print(f"Extracted text with {len(cleaned_texts)} lines")
        if cleaned_texts:
            print(full_text)
        else:
            print("No text extracted!")
        print("------ PADDLE OCR TEXT END ------\n")
        
        return full_text
    
    def extract_name(self, text):
        """Extract name with improved logic"""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        
        # Keywords to avoid in names
        garbage_keywords = [
            "government", "of", "india", "unique", "identification", "authority",
            "dob", "gender", "male", "female", "aadhaar", "address", "pin",
            "www", "uidai", "mera", "card", "number", "date", "birth"
        ]
        
        # Look for valid name patterns
        for i, line in enumerate(lines):
            line_clean = re.sub(r'[^\w\s]', ' ', line).strip()
            line_lower = line_clean.lower()
            
            # Check if line looks like a name
            if (re.match(r'^[A-Za-z\s]+$', line_clean) and 
                len(line_clean.split()) >= 2 and 
                len(line_clean.split()) <= 5 and
                len(line_clean) >= 5 and len(line_clean) <= 50 and
                not any(keyword in line_lower for keyword in garbage_keywords)):
                
                # Additional validation: should not be all caps institutional text
                if not (line_clean.isupper() and len(line_clean.split()) > 3):
                    return line_clean.title()
        
        return ""
    
    def extract_dob(self, text):
        """Extract date of birth with multiple patterns"""
        # Common DOB patterns
        patterns = [
            r"(?:DOB|Date of Birth|D\.O\.B|D O B|D\.O\.B\.)[:\s]*([0-9]{1,2}[\/\-][0-9]{1,2}[\/\-][0-9]{4})",
            r"(?:DOB|Date of Birth|D\.O\.B|D O B|D\.O\.B\.)[:\s]*([0-9]{1,2}[\/\-][0-9]{1,2}[\/\-][0-9]{2})",
            r"\b([0-9]{1,2}[\/\-][0-9]{1,2}[\/\-][0-9]{4})\b",
            r"\b([0-9]{1,2}[\/\-][0-9]{1,2}[\/\-][0-9]{2})\b"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if match:
                dob = match.group(1)
                # Normalize separators
                dob = re.sub(r'[-]', '/', dob)
                return dob
        
        return ""
    
    def extract_gender(self, text):
        """Extract gender with improved pattern matching"""
        # Look for gender keywords
        match = re.search(r'\b(Male|Female|Transgender|M|F|T)\b', text, re.I)
        if match:
            gender = match.group(1).upper()
            if gender == 'M':
                return 'Male'
            elif gender == 'F':
                return 'Female'
            elif gender == 'T':
                return 'Transgender'
            else:
                return gender.capitalize()
        return ""
    
    def extract_aadhaar_number(self, text):
        """Extract Aadhaar number with improved pattern matching"""
        # Remove newlines and normalize spaces
        text_clean = re.sub(r'\s+', ' ', text.replace('\n', ' '))
        
        # Pattern for 12-digit Aadhaar with spaces
        patterns = [
            r'\b(\d{4}\s*\d{4}\s*\d{4})\b',
            r'\b(\d{12})\b',
            r'(\d{4}[\s\-]\d{4}[\s\-]\d{4})',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text_clean)
            for match in matches:
                # Clean and validate
                clean_number = re.sub(r'[\s\-]', '', match)
                if len(clean_number) == 12 and clean_number.isdigit():
                    # Format as XXXX XXXX XXXX
                    return f"{clean_number[:4]} {clean_number[4:8]} {clean_number[8:]}"
        
        return ""
    
    def extract_fathers_name_and_address(self, text):
        """Extract father's name and address with improved logic"""
        father_name = ""
        address = ""
        
        # Normalize S/o patterns
        text = re.sub(r'S\s*/\s*[oO]', 'S/o', text)
        text = re.sub(r'[Ss]\s*/\s*[oO]', 'S/o', text)
        
        # Extract father's name
        father_patterns = [
            r"(?:S/o|S/O|Son of|W/o|D/o|W/O|D/O|Father)[:\s]*([A-Z][a-zA-Z\s]{2,50})",
            r"(?:पिता|Father|S/o)[:\s]*([A-Z][a-zA-Z\s]{2,50})"
        ]
        
        for pattern in father_patterns:
            match = re.search(pattern, text, re.I)
            if match:
                fn_candidate = match.group(1).strip()
                # Validate father's name
                if (len(fn_candidate.split()) >= 2 and 
                    len(fn_candidate) <= 50 and
                    re.match(r'^[A-Za-z\s]+$', fn_candidate)):
                    father_name = fn_candidate.title()
                    break
        
        # Extract address
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Patterns to ignore in address
        ignore_patterns = [
            r'mera\s+aadhaar', r'www\.', r'uidai', r'https?://', r'email', r'@',
            r'contact\s+us', r'government', r'unique.*identification'
        ]
        
        address_lines = []
        address_started = False
        
        for line in lines:
            line_lower = line.lower()
            
            # Skip garbage lines
            if any(re.search(pattern, line_lower) for pattern in ignore_patterns):
                continue
            
            # Skip Aadhaar number lines
            if re.search(r'\d{4}\s*\d{4}\s*\d{4}', line):
                continue
            
            # Start collecting address after finding father's name or address keyword
            if father_name and father_name.lower() in line_lower:
                address_started = True
                continue
            
            if re.search(r'\b(address|पता)\b', line_lower, re.I):
                address_started = True
                continue
            
            # Collect address lines
            if address_started or re.search(r'\d{6}', line):  # PIN code indicates address
                # Validate line has enough alphabetic content
                if len(re.findall(r'[A-Za-z]', line)) >= max(3, len(line) * 0.3):
                    address_lines.append(line)
                    # Stop if we find a PIN code (end of address)
                    if re.search(r'\b\d{6}\b', line):
                        break
        
        if address_lines:
            address = ', '.join(address_lines).strip()
        
        # Fallback address extraction if nothing found
        if not address:
            potential_address_lines = []
            for line in lines:
                if (re.search(r'\d', line) and 
                    len(re.findall(r'[A-Za-z]', line)) >= 3):
                    potential_address_lines.append(line)
            
            if potential_address_lines:
                address = ', '.join(potential_address_lines[:3]).strip()  # Limit to 3 lines
        
        return father_name, address
    
    def extract_front_details(self, image_path):
        """Extract details from front side of Aadhaar"""
        print(f"Processing front image: {image_path}")
        
        # Try multiple approaches and combine results
        results = []
        
        # Approach 1: Direct OCR
        text1 = self.extract_text_with_paddle(image_path)
        results.append({
            "Full Name": self.extract_name(text1),
            "Date of Birth": self.extract_dob(text1),
            "Gender": self.extract_gender(text1),
            "Aadhaar Number": self.extract_aadhaar_number(text1)
        })
        
        # Approach 2: Preprocessed image
        processed_img = self.preprocess_image_enhanced(image_path)
        text2 = self.extract_text_with_paddle(processed_img)
        results.append({
            "Full Name": self.extract_name(text2),
            "Date of Birth": self.extract_dob(text2),
            "Gender": self.extract_gender(text2),
            "Aadhaar Number": self.extract_aadhaar_number(text2)
        })
        
        # Approach 3: Cropped regions
        details_region = self.crop_regions_of_interest(processed_img, 'front_details')
        text3 = self.extract_text_with_paddle(details_region)
        
        number_region = self.crop_regions_of_interest(processed_img, 'aadhaar_number')
        text4 = self.extract_text_with_paddle(number_region)
        
        results.append({
            "Full Name": self.extract_name(text3),
            "Date of Birth": self.extract_dob(text3),
            "Gender": self.extract_gender(text3),
            "Aadhaar Number": self.extract_aadhaar_number(text4)
        })
        
        # Merge results - pick the longest/most complete value for each field
        final_result = {}
        for key in results[0].keys():
            candidates = [r.get(key, "") for r in results]
            # Pick the longest non-empty result
            final_result[key] = max(candidates, key=lambda x: len(x) if x else 0)
        
        return final_result
    
    def extract_back_details(self, image_path):
        """Extract details from back side of Aadhaar"""
        print(f"Processing back image: {image_path}")
        
        results = []
        
        # Approach 1: Direct OCR
        text1 = self.extract_text_with_paddle(image_path)
        father_name1, address1 = self.extract_fathers_name_and_address(text1)
        results.append({"Father's Name": father_name1, "Address": address1})
        
        # Approach 2: Preprocessed image
        processed_img = self.preprocess_image_enhanced(image_path)
        text2 = self.extract_text_with_paddle(processed_img)
        father_name2, address2 = self.extract_fathers_name_and_address(text2)
        results.append({"Father's Name": father_name2, "Address": address2})
        
        # Approach 3: ROI cropping
        roi_img = self.crop_regions_of_interest(processed_img, 'back_top_left')
        text3 = self.extract_text_with_paddle(roi_img)
        father_name3, address3 = self.extract_fathers_name_and_address(text3)
        results.append({"Father's Name": father_name3, "Address": address3})
        
        # Merge results
        final_result = {}
        for key in results[0].keys():
            candidates = [r.get(key, "") for r in results]
            final_result[key] = max(candidates, key=lambda x: len(x) if x else 0)
        
        return final_result
    
    def extract_complete_aadhaar_details(self, front_image_path, back_image_path):
        """Extract complete Aadhaar details from both sides"""
        print("Starting complete Aadhaar extraction with PaddleOCR...")
        
        front_details = self.extract_front_details(front_image_path)
        back_details = self.extract_back_details(back_image_path)
        
        # Combine results
        complete_details = {**front_details, **back_details}
        
        return complete_details


def main():
    """Example usage"""
    # Initialize OCR with English language
    # For better accuracy with Hindi text, use lang=['en', 'hi']
    aadhaar_ocr = AadhaarOCR(use_angle_cls=True, lang='en')
    
    # Paths to your Aadhaar images
    front_img_path = "/home/rishabh/openbharatocr/openbharatocr/ocr/Aadhar_sample/A1.jpeg"
    back_img_path = "/home/rishabh/openbharatocr/openbharatocr/ocr/Aadhar_sample/A2.jpeg"
    
    try:
        # Extract complete details
        details = aadhaar_ocr.extract_complete_aadhaar_details(front_img_path, back_img_path)
        
        print("\n" + "="*50)
        print("FINAL EXTRACTED AADHAAR DETAILS (PaddleOCR)")
        print("="*50)
        
        for key, value in details.items():
            print(f"{key:15}: {value if value else 'Not Found'}")
        
        print("="*50)
        
    except Exception as e:
        print(f"Error processing Aadhaar: {str(e)}")


if __name__ == "__main__":
    main()