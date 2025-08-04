import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image
import re
from datetime import datetime

try:
    from paddlex.inference.pipelines.ocr.result import OCRResult 
except ImportError:
    class OCRResult: 
        def __init__(self):
            self.rec_texts = []
            self.rec_scores = []
        def __str__(self):
            return "Dummy OCRResult object"
        def __repr__(self):
            return self.__str__()


class AadhaarOCR:
    def __init__(self, use_angle_cls=True, lang='en'):
        """
        Initialize PaddleOCR with optimized settings for Aadhaar cards
        
        Args:
            use_angle_cls: Enable angle classification for rotated text
            lang: Language code ('en' for English, 'hi' for Hindi, ['en', 'hi'] for both)
        """
        initialization_attempts = [
            {'lang': lang, 'use_textline_orientation': True}, 
            {'lang': lang, 'use_angle_cls': True},
            {'lang': lang},
            {'lang': lang if isinstance(lang, str) else 'en'},
            {}
        ]
        
        self.ocr = None
        last_error = None
        
        for params in initialization_attempts:
            try:
                self.ocr = PaddleOCR(**params)
                break
            except Exception as e:
                last_error = e
                continue
        
        if self.ocr is None:
            raise RuntimeError(f"Failed to initialize PaddleOCR with any configuration. Last error: {last_error}")
    
    def preprocess_image_enhanced(self, image_path):
        """Enhanced preprocessing for better OCR accuracy"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise and sharpen
        denoised = cv2.medianBlur(enhanced, 3)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # Morphological cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        cleaned = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, kernel)

        return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    
    def crop_regions_of_interest(self, image, region_type='back'):
        """Crop specific regions for better OCR accuracy"""
        h, w = image.shape[:2]
        
        if region_type == 'back_top_left':
            return image[0:int(h * 0.6), 0:int(w * 0.65)]
        elif region_type == 'front_details':
            return image[int(h * 0.3):h, 0:w]
        elif region_type == 'aadhaar_number':
            return image[int(h * 0.7):h, 0:w]
        
        return image 
    
    def extract_text_with_paddle(self, image, confidence_threshold=0.5):
        """
        Extract text using PaddleOCR with confidence filtering.
        Handles various PaddleOCR output structures.
        """
        try:
            if hasattr(self.ocr, 'predict'):
                raw_result = self.ocr.predict(image)
            else:
                raw_result = self.ocr.ocr(image)
        except Exception:
            try:
                raw_result = self.ocr.ocr(image)
            except Exception:
                return ""
        
        if raw_result is None or len(raw_result) == 0:
            return ""
        
        texts = []
        try:
            for item_container in raw_result:
                # Handle dictionary format with rec_texts and rec_scores
                if isinstance(item_container, dict) and 'rec_texts' in item_container and 'rec_scores' in item_container:
                    for text, score in zip(item_container['rec_texts'], item_container['rec_scores']):
                        if score >= confidence_threshold:
                            texts.append(text)

                # Handle OCRResult objects
                elif isinstance(item_container, OCRResult):
                    if hasattr(item_container, 'rec_texts') and hasattr(item_container, 'rec_scores'):
                        for text, score in zip(item_container.rec_texts, item_container.rec_scores):
                            if score >= confidence_threshold:
                                texts.append(text)
                
                # Handle list format (bbox, text, confidence)
                elif isinstance(item_container, list):
                    for bbox_text_conf in item_container:
                        if isinstance(bbox_text_conf, list) and len(bbox_text_conf) >= 2:
                            if isinstance(bbox_text_conf[1], list) and len(bbox_text_conf[1]) >= 2:
                                text = str(bbox_text_conf[1][0])
                                confidence = float(bbox_text_conf[1][1])
                                if confidence >= confidence_threshold:
                                    texts.append(text)
                            elif isinstance(bbox_text_conf[1], str):
                                text = bbox_text_conf[1]
                                confidence = bbox_text_conf[2] if len(bbox_text_conf) > 2 else 1.0
                                if confidence >= confidence_threshold:
                                    texts.append(text)
                
                # Handle string format
                elif isinstance(item_container, str):
                    texts.append(item_container)

        except Exception:
            # Fallback text extraction using regex
            try:
                full_raw_str = str(raw_result)
                potential_lines = re.findall(r"'(.*?)(?:'|\],)", full_raw_str)
                for line in potential_lines:
                    if len(line.strip()) > 3 and not any(k in line.lower() for k in ['bbox', 'score', 'conf']):
                        texts.append(line.strip())
            except:
                texts = []

        # Clean and join texts
        cleaned_texts = [text.strip() for text in texts if text and isinstance(text, str) and text.strip()]
        return '\n'.join(cleaned_texts)
    
    def extract_name(self, text):
        """Extract full name from the OCR text"""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        
        garbage_keywords = [
            "government", "of", "india", "unique", "identification", "authority",
            "dob", "date of birth", "gender", "male", "female", "aadhaar", "address", "pin", "code",
            "www", "uidai", "mera", "card", "number", "id", "year", "birth", "vid", "virtual", "identification",
            "son", "daughter", "wife", "s/o", "d/o", "w/o", "father"
        ]
        
        name_candidate = ""

        # Look for name after "Government of India" or similar phrases
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if "government of india" in line_lower or "unique identification authority of india" in line_lower:
                if i + 1 < len(lines):
                    potential_name_line = lines[i+1].strip()
                    if (re.match(r'^[A-Za-z\s\.\-]+$', potential_name_line) and 
                        len(potential_name_line.split()) >= 2 and 
                        5 <= len(potential_name_line) <= 50 and 
                        not any(keyword in potential_name_line.lower() for keyword in garbage_keywords)):
                        return potential_name_line.title()
            
            # Look for explicit "Name:" field
            elif "name" in line_lower and ":" in line and i + 1 < len(lines):
                match = re.search(r'name[:\s]*([A-Z][a-zA-Z\s\.\-]{2,50})', line, re.I)
                if match:
                    name_candidate = match.group(1).strip()
                    if (re.match(r'^[A-Za-z\s\.\-]+$', name_candidate) and 
                        len(name_candidate.split()) >= 2 and 
                        5 <= len(name_candidate) <= 50 and
                        not any(keyword in name_candidate.lower() for keyword in garbage_keywords)):
                        return name_candidate.title()
                
                potential_name_line = lines[i+1].strip()
                if (re.match(r'^[A-Za-z\s\.\-]+$', potential_name_line) and 
                    len(potential_name_line.split()) >= 2 and 
                    5 <= len(potential_name_line) <= 50 and 
                    not any(keyword in potential_name_line.lower() for keyword in garbage_keywords)):
                    return potential_name_line.title()

        # General name extraction from valid lines
        for line in lines:
            line_clean = re.sub(r'[^\w\s]', ' ', line).strip()
            line_lower = line_clean.lower()
            
            if (re.match(r'^[A-Za-z\s]+$', line_clean) and 
                2 <= len(line_clean.split()) <= 5 and 
                5 <= len(line_clean) <= 50 and 
                not any(keyword in line_lower for keyword in garbage_keywords) and
                not (line_clean.isupper() and len(line_clean.split()) > 3)):
                
                # Prefer CamelCase names
                if re.match(r'^[A-Z][a-z]+(?:[A-Z][a-z]+)+$', line_clean) and len(line_clean) >= 8:
                    return line_clean.title()
                
                if not name_candidate or (len(line_clean) > len(name_candidate) and 'male' not in line_lower and 'female' not in line_lower):
                     name_candidate = line_clean.title()
        
        return name_candidate if name_candidate else ""
    
    def extract_dob(self, text):
        """Extract date of birth with multiple patterns"""
        current_year = datetime.now().year 
        
        patterns = [
            r"(?:DOB|Date of Birth|D\.O\.B|D O B|D\.O\.B\.|Year of Birth)[:\s]*([0-9]{1,2}[\/\-][0-9]{1,2}[\/\-][0-9]{4})\b",
            r"\b([0-9]{1,2}[\/\-][0-9]{1,2}[\/\-][0-9]{4})\b",
            r"(?:DOB|Date of Birth|D\.O\.B|D O B|D\.O\.B\.|Year of Birth)[:\s]*([0-9]{1,2}[\/\-][0-9]{1,2}[\/\-][0-9]{2})\b",
            r"\b([0-9]{1,2}[\/\-][0-9]{1,2}[\/\-][0-9]{2})\b"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if match:
                dob = match.group(1)
                dob = re.sub(r'[-]', '/', dob)
                
                parts = dob.split('/')
                if len(parts) == 3:
                    try:
                        day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
                        if 0 < month <= 12 and 0 < day <= 31:
                            # Handle 2-digit years
                            if len(parts[2]) == 2:
                                year = (2000 + year) if (year <= (current_year % 100)) else (1900 + year)
                            if 1900 <= year <= current_year:
                                return dob
                    except ValueError:
                        pass
        
        return ""
    
    def extract_gender(self, text):
        """Extract gender with improved pattern matching"""
        text_norm = text.replace('T/FEMALE', 'FEMALE').replace('M/MALE', 'MALE').upper()
        
        # Look for full gender words
        if re.search(r'\bFEMALE\b', text_norm):
            return 'Female'
        elif re.search(r'\bMALE\b', text_norm):
            return 'Male'
        elif re.search(r'\bTRANSGENDER\b', text_norm):
            return 'Transgender'
        
        # Look for single letter abbreviations
        match = re.search(r'\b(F|M|T)\b', text_norm)
        if match:
            gender_abbr = match.group(1)
            if gender_abbr == 'M':
                return 'Male'
            elif gender_abbr == 'F':
                return 'Female'
            elif gender_abbr == 'T':
                return 'Transgender'
        
        return ""
    
    def extract_aadhaar_number(self, text):
        """Extract Aadhaar number with validation"""
        text_clean = re.sub(r'\s+', ' ', text.replace('\n', ' '))
        
        patterns = [
            r'\b(\d{4}\s*\d{4}\s*\d{4})\b',
            r'\b(\d{12})\b',
            r'(\d{4}[\s\-]\d{4}[\s\-]\d{4})'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text_clean)
            for match in matches:
                clean_number = re.sub(r'[\s\-]', '', match)
                if len(clean_number) == 12 and clean_number.isdigit():
                    return f"{clean_number[:4]} {clean_number[4:8]} {clean_number[8:]}"
        
        return ""
    
    def extract_relative_name_and_address(self, text):
        """
        Extract relative's name and address.
        Returns (relation_type, relative_name, address)
        """
        relative_name = ""
        relation_type = "Not Found"
        
        # Normalize relation patterns
        temp_text = text
        temp_text = re.sub(r'[Ss]\s*/\s*[oO]', 'S/o', temp_text)
        temp_text = re.sub(r'[Dd]\s*/\s*[oO]', 'D/o', temp_text)
        temp_text = re.sub(r'[Ww]\s*/\s*[oO]', 'W/o', temp_text)

        # Priority patterns for relation extraction
        relative_patterns = {
            "Husband": r"(W/o|W/O|Wife of)[:\s]*([A-Z][a-zA-Z\s\.\-]{2,60})",
            "Father": r"(S/o|S/O|Son of)[:\s]*([A-Z][a-zA-Z\s\.\-]{2,60})",
            "Daughter": r"(D/o|D/O|Daughter of)[:\s]*([A-Z][a-zA-Z\s\.\-]{2,60})"
        }

        matched_string_to_remove = ""
        
        # Try priority patterns first
        for rel_type_key, pattern in relative_patterns.items():
            match = re.search(pattern, temp_text, re.I)
            if match:
                candidate_name = match.group(2).strip() 
                if (len(candidate_name.split()) >= 2 and
                    len(candidate_name) <= 60 and
                    re.match(r'^[A-Za-z\s\.\-]+$', candidate_name) and
                    not any(keyword in candidate_name.lower() for keyword in ["dob", "gender", "aadhaar", "address", "pin", "india", "private", "limited", "mobile", "tel", "email"])):
                    
                    relative_name = candidate_name.title()
                    relation_type = rel_type_key
                    matched_string_to_remove = match.group(0) 
                    break

        # Try generic "Father" pattern
        if not relative_name:
            match = re.search(r"(Father)[:\s]*([A-Z][a-zA-Z\s\.\-]{2,60})", temp_text, re.I)
            if match:
                candidate_name = match.group(2).strip()
                if (len(candidate_name.split()) >= 2 and 
                    len(candidate_name) <= 60 and
                    re.match(r'^[A-Za-z\s\.\-]+$', candidate_name) and
                    not any(keyword in candidate_name.lower() for keyword in ["dob", "gender", "aadhaar", "address", "pin", "india", "private", "limited", "mobile", "tel", "email"])):
                    relative_name = candidate_name.title()
                    relation_type = "Father"
                    matched_string_to_remove = match.group(0) 

        # Try inferring from address lines
        if not relative_name:
            lines = [line.strip() for line in temp_text.split('\n') if line.strip()]
            for line in lines:
                match = re.search(r'^([A-Z][a-zA-Z\s\.\-]{3,60})\s*[,.]?\s*(?:H\.NO\.|HOUSE|VILLAGE|APARTMENT|STREET|ROAD|SECTOR|PHASE)', line, re.I)
                if match:
                    candidate_name = match.group(1).strip()
                    if (len(candidate_name.split()) >= 2 and
                        len(candidate_name) <= 60 and
                        re.match(r'^[A-Za-z\s\.\-]+$', candidate_name) and
                        not any(keyword in candidate_name.lower() for keyword in ["dob", "gender", "aadhaar", "address", "pin", "india", "private", "limited", "mobile", "tel", "email"])):
                        
                        relative_name = candidate_name.title()
                        if not re.search(r'(W/o|W/O|D/o|D/O)', line, re.I):
                            relation_type = "Father"
                        matched_string_to_remove = match.group(0)
                        break
        
        # Clean text for address extraction
        cleaned_text_for_address = text
        if relative_name and matched_string_to_remove:
            escaped_to_remove = re.escape(matched_string_to_remove)
            cleaned_text_for_address = re.sub(escaped_to_remove + r'[:,\s]*', ' ', cleaned_text_for_address, flags=re.I).strip()
            
            # Remove relative name from address
            if relative_name.title() in cleaned_text_for_address:
                cleaned_text_for_address = cleaned_text_for_address.replace(relative_name.title(), '').strip()
            if relative_name.upper() in cleaned_text_for_address:
                cleaned_text_for_address = cleaned_text_for_address.replace(relative_name.upper(), '').strip()
            
            cleaned_text_for_address = re.sub(r'\n\s*\n', '\n', cleaned_text_for_address).strip()
            cleaned_text_for_address = re.sub(r'\s{2,}', ' ', cleaned_text_for_address).strip()

        # Extract address
        lines_for_address = [line.strip() for line in cleaned_text_for_address.split('\n') if line.strip()]
        address_candidates = []
        address_collection_started = False
        
        ignore_patterns = [
            r'mera\s+aadhaar', r'www\.', r'uidai', r'https?://', r'email', r'@',
            r'contact\s+us', r'government', r'unique.*identification', r'vid',
            r'date of birth', r'gender', r'male', r'female', r'aadhaar', r'number',
            r'paddles', r'ocr', r'\d{4}\s*\d{4}\s*\d{4}',
            r'tel', r'mobile', r'uid', r'virtual\s+id'
        ]
        
        address_start_keywords = [r'\baddress\b', r'\bपता\b', r'\bH\.NO\.\b', r'\bHOUSE\b', r'\bFLAT\b', r'\bVILLAGE\b', r'\bCOLONY\b', r'\bSTREET\b', r'\bROAD\b', r'\bSECTOR\b']

        for line in lines_for_address:
            line_lower = line.lower()
            
            # Start collecting after address keywords
            if any(re.search(kw, line_lower) for kw in address_start_keywords):
                address_collection_started = True
            
            # Collect address lines
            if address_collection_started or re.search(r'\b\d{6}\b', line) or re.search(r'\bH\.NO\.\s*\-?\d+', line, re.I):
                is_garbage = any(re.search(pattern, line_lower) for pattern in ignore_patterns)
                has_alphanumeric = len(re.findall(r'[A-Za-z0-9]', line)) >= len(line) * 0.3
                
                if not is_garbage and len(line) > 5 and has_alphanumeric:
                    address_candidates.append(line)

        # Join and clean final address
        final_address = ", ".join(address_candidates).strip()
        final_address = re.sub(r'(?:PIN\s*CODE|PINCODE)[:\s]*', '', final_address, flags=re.I).strip()
        final_address = re.sub(r'[,]{2,}', ',', final_address).strip() 
        final_address = re.sub(r'\s{2,}', ' ', final_address).strip()
        final_address = re.sub(r'^[\s,]+|[\s,]+$', '', final_address) 
        final_address = ", ".join(filter(None, final_address.split(', ')))

        return relation_type, relative_name, final_address
    
    def extract_front_details(self, image_path):
        """Extract details from front side of Aadhaar"""
        results = []
        
        # Process full image
        processed_full_img = self.preprocess_image_enhanced(image_path)
        text1 = self.extract_text_with_paddle(processed_full_img)
        results.append({
            "Full Name": self.extract_name(text1),
            "Date of Birth": self.extract_dob(text1),
            "Gender": self.extract_gender(text1),
            "Aadhaar Number": self.extract_aadhaar_number(text1)
        })
        
        # Process cropped regions
        details_region = self.crop_regions_of_interest(processed_full_img, 'front_details')
        text2_cropped = self.extract_text_with_paddle(details_region)
        
        number_region = self.crop_regions_of_interest(processed_full_img, 'aadhaar_number')
        text3_cropped = self.extract_text_with_paddle(number_region)

        results.append({
            "Full Name": self.extract_name(text2_cropped),
            "Date of Birth": self.extract_dob(text2_cropped),
            "Gender": self.extract_gender(text2_cropped),
            "Aadhaar Number": self.extract_aadhaar_number(text3_cropped)
        })
        
        # Combine results by choosing best candidate for each field
        final_result = {}
        for key in results[0].keys(): 
            candidates = [r.get(key, "") for r in results]
            final_result[key] = max(candidates, key=len) if any(candidates) else "" 
        
        return final_result
    
    def extract_back_details(self, image_path):
        """Extract details from back side of Aadhaar"""
        results = []
        
        # Process full image
        processed_full_img = self.preprocess_image_enhanced(image_path)
        text1 = self.extract_text_with_paddle(processed_full_img)
        relation_type1, relative_name1, address1 = self.extract_relative_name_and_address(text1)
        results.append({"Relation Type": relation_type1, "Relative's Name": relative_name1, "Address": address1})
        
        # Process cropped region
        roi_img = self.crop_regions_of_interest(processed_full_img, 'back_top_left')
        text2_cropped = self.extract_text_with_paddle(roi_img)
        relation_type2, relative_name2, address2 = self.extract_relative_name_and_address(text2_cropped)
        results.append({"Relation Type": relation_type2, "Relative's Name": relative_name2, "Address": address2})
        
        # Combine results
        final_result = {}
        for key in results[0].keys():
            candidates = [r.get(key, "") for r in results]
            final_result[key] = max(candidates, key=len) if any(candidates) else ""
        
        return final_result
    
    def extract_complete_aadhaar_details(self, front_image_path, back_image_path):
        """Extract complete Aadhaar details from both sides"""
        front_details = self.extract_front_details(front_image_path)
        back_details = self.extract_back_details(back_image_path)
        
        return {**front_details, **back_details}


def main():
    """Example usage"""
    aadhaar_ocr = AadhaarOCR(use_angle_cls=True, lang='en')
    
    front_img_path = "/home/rishabh/openbharatocr/openbharatocr/ocr/Aadhar_sample/A1.jpeg" 
    back_img_path = "/home/rishabh/openbharatocr/openbharatocr/ocr/Aadhar_sample/A2.jpeg"
    
    try:
        details = aadhaar_ocr.extract_complete_aadhaar_details(front_img_path, back_img_path)
        
        print("\n" + "="*50)
        print("EXTRACTED AADHAAR DETAILS")
        print("="*50)
        
        for key, value in details.items():
            if key == "Relative's Name":
                relation_type = details.get("Relation Type", "Not Found")
                if relation_type != "Not Found" and value:
                    print(f"{relation_type}'s Name: {value}")
                else:
                    print(f"Father's Name  : Not Found")
            elif key == "Relation Type":
                continue 
            else:
                print(f"{key:15}: {value if value else 'Not Found'}")
        
        print("="*50)
        
    except Exception as e:
        print(f"Error during Aadhaar processing: {str(e)}")


if __name__ == "__main__":
    main()