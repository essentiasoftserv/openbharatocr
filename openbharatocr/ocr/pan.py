import cv2
import numpy as np
import re
from datetime import datetime
from paddleocr import PaddleOCR
import json
from typing import Dict, List, Tuple, Optional

class PANCardExtractor:
    def __init__(self):
        """
        Initialize PAN Card extractor with PaddleOCR
        """
        # Use minimal initialization to avoid parameter conflicts
        self.ocr = PaddleOCR(lang='en')
        
        # PAN card patterns
        self.pan_pattern = r'[A-Z]{5}[0-9]{4}[A-Z]{1}'
        self.date_patterns = [
            r'\d{2}/\d{2}/\d{4}',
            r'\d{2}-\d{2}-\d{4}',
            r'\d{2}\.\d{2}\.\d{4}'
        ]
        
        # Common keywords for field identification
        self.field_keywords = {
            'name': ['name', 'श्री', 'श्रीमती', 'कुमारी'],
            'father_name': ['father', 'पिता', "father's name", 'fathers name'],
            'dob': ['birth', 'जन्म', 'date of birth', 'dob', 'born'],
            'pan': ['permanent account number', 'pan', 'account number']
        }

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for better OCR accuracy
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize if too small (maintain aspect ratio)
        height, width = img_rgb.shape[:2]
        if width < 800:
            scale = 800 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            img_rgb = cv2.resize(img_rgb, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        # Sharpen
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # Convert back to RGB for PaddleOCR
        final_img = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
        
        return final_img

    def extract_text_with_coordinates(self, image: np.ndarray) -> List[Tuple]:
        """
        Extract text with bounding box coordinates using PaddleOCR
        """
        try:
            # Try new predict method first
            results = self.ocr.predict(image)
        except (AttributeError, TypeError):
            # Fallback to older ocr method without cls parameter
            try:
                results = self.ocr.ocr(image)
            except:
                # Last resort - try with different format
                results = self.ocr(image)
        
        extracted_data = []
        
        # Handle the specific format we're getting
        if results and isinstance(results, list) and len(results) > 0:
            result_dict = results[0]  # First page result
            
            # Check if it's the new PaddleX format
            if isinstance(result_dict, dict) and 'rec_texts' in result_dict:
                texts = result_dict.get('rec_texts', [])
                scores = result_dict.get('rec_scores', [])
                polygons = result_dict.get('rec_polys', [])
                
                print("Extracted text from PaddleX format:")
                for i, text in enumerate(texts):
                    confidence = scores[i] if i < len(scores) else 0.8
                    bbox = polygons[i] if i < len(polygons) else []
                    
                    print(f"  - {text} (confidence: {confidence:.2f})")
                    
                    # Only keep high confidence text
                    if confidence > 0.3:  # Lower threshold for better extraction
                        if len(bbox) > 0:
                            # Calculate center coordinates
                            center_x = float(np.mean(bbox[:, 0]))
                            center_y = float(np.mean(bbox[:, 1]))
                        else:
                            center_x = center_y = 0
                        
                        extracted_data.append({
                            'text': text.strip(),
                            'confidence': confidence,
                            'bbox': bbox.tolist() if hasattr(bbox, 'tolist') else bbox,
                            'center_x': center_x,
                            'center_y': center_y
                        })
                
                return extracted_data
            
            # Fallback to original format handling
            elif isinstance(result_dict, list):
                lines = result_dict
                for line in lines:
                    try:
                        if line and isinstance(line, (list, tuple)) and len(line) >= 2:
                            bbox = line[0]
                            text_info = line[1]
                            
                            if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                                text = text_info[0]
                                confidence = text_info[1]
                            else:
                                text = str(text_info)
                                confidence = 0.8
                            
                            if confidence > 0.3:
                                center_x = sum([point[0] for point in bbox]) / len(bbox)
                                center_y = sum([point[1] for point in bbox]) / len(bbox)
                                
                                extracted_data.append({
                                    'text': text.strip(),
                                    'confidence': confidence,
                                    'bbox': bbox,
                                    'center_x': center_x,
                                    'center_y': center_y
                                })
                    except Exception as e:
                        print(f"Error processing line {line}: {e}")
                        continue
        
        return extracted_data

    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove extra spaces and special characters
        cleaned = re.sub(r'\s+', ' ', text)
        cleaned = re.sub(r'[^\w\s/.-]', '', cleaned)
        return cleaned.strip()

    def find_pan_number(self, text_data: List[Dict]) -> Optional[str]:
        """Extract PAN number using pattern matching"""
        for item in text_data:
            text = self.clean_text(item['text'])
            # Look for PAN pattern
            pan_match = re.search(self.pan_pattern, text.upper())
            if pan_match:
                return pan_match.group()
            
            # Sometimes PAN is split across multiple OCR results
            text_upper = text.upper().replace(' ', '')
            if re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$', text_upper):
                return text_upper
        
        return None

    def find_dates(self, text_data: List[Dict]) -> List[str]:
        """Extract all dates from text"""
        dates = []
        for item in text_data:
            text = self.clean_text(item['text'])
            for pattern in self.date_patterns:
                matches = re.findall(pattern, text)
                dates.extend(matches)
        return dates

    def find_names(self, text_data: List[Dict]) -> Dict[str, str]:
        """
        Extract name and father's name using positional analysis and keywords
        """
        # Sort by Y coordinate (top to bottom)
        sorted_data = sorted(text_data, key=lambda x: x['center_y'])
        
        names = {'name': '', 'father_name': ''}
        
        # Look for name patterns
        for i, item in enumerate(sorted_data):
            text = self.clean_text(item['text']).lower()
            
            # Skip government text, pan number, dates
            if any(keyword in text for keyword in ['government', 'income tax', 'permanent', 'account']):
                continue
            if re.search(self.pan_pattern, text.upper()):
                continue
            if any(re.search(pattern, text) for pattern in self.date_patterns):
                continue
            
            # Look for name indicators
            for keyword in self.field_keywords['name']:
                if keyword in text:
                    # Name usually follows the keyword
                    remaining_text = text.split(keyword, 1)[-1].strip()
                    if remaining_text:
                        names['name'] = remaining_text.title()
                    # Also check next line
                    elif i + 1 < len(sorted_data):
                        next_text = self.clean_text(sorted_data[i + 1]['text'])
                        if self.is_likely_name(next_text):
                            names['name'] = next_text.title()
                    break
            
            # Look for father's name indicators
            for keyword in self.field_keywords['father_name']:
                if keyword in text:
                    remaining_text = text.split(keyword, 1)[-1].strip()
                    if remaining_text:
                        names['father_name'] = remaining_text.title()
                    elif i + 1 < len(sorted_data):
                        next_text = self.clean_text(sorted_data[i + 1]['text'])
                        if self.is_likely_name(next_text):
                            names['father_name'] = next_text.title()
                    break
        
        # If direct keyword matching didn't work, use positional analysis
        if not names['name']:
            names = self.extract_names_by_position(sorted_data)
        
        return names

    def is_likely_name(self, text: str) -> bool:
        """Check if text is likely to be a person's name"""
        text = text.strip()
        # Basic checks for name-like text
        if len(text) < 2 or len(text) > 50:
            return False
        if re.search(r'\d', text):  # Contains numbers
            return False
        if len(text.split()) > 5:  # Too many words
            return False
        return True

    def extract_names_by_position(self, sorted_data: List[Dict]) -> Dict[str, str]:
        """Extract names based on typical PAN card layout"""
        names = {'name': '', 'father_name': ''}
        
        # Filter out government headers and PAN number
        content_lines = []
        for item in sorted_data:
            text = self.clean_text(item['text']).lower()
            if not any(skip in text for skip in ['government', 'income tax', 'department', 'permanent account']):
                if not re.search(self.pan_pattern, text.upper()):
                    if self.is_likely_name(item['text']):
                        content_lines.append(item)
        
        # First meaningful line is usually the name
        if len(content_lines) >= 1:
            names['name'] = self.clean_text(content_lines[0]['text']).title()
        
        # Second meaningful line is usually father's name
        if len(content_lines) >= 2:
            names['father_name'] = self.clean_text(content_lines[1]['text']).title()
        
        return names

    def validate_pan(self, pan: str) -> bool:
        """Validate PAN number format"""
        if not pan:
            return False
        return bool(re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$', pan))

    def validate_date(self, date_str: str) -> bool:
        """Validate date format and check if it's reasonable for DOB"""
        try:
            # Try different date formats
            for fmt in ['%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y']:
                try:
                    date_obj = datetime.strptime(date_str, fmt)
                    # Check if date is reasonable (person should be between 1 and 120 years old)
                    current_year = datetime.now().year
                    age = current_year - date_obj.year
                    return 1 <= age <= 120
                except ValueError:
                    continue
            return False
        except:
            return False

    def extract_pan_details(self, image_path: str) -> Dict:
        """
        Main method to extract all PAN card details
        """
        try:
            # Preprocess image
            processed_img = self.preprocess_image(image_path)
            
            # Extract text with coordinates
            text_data = self.extract_text_with_coordinates(processed_img)
            
            if not text_data:
                return {'error': 'No text could be extracted from the image'}
            
            # Debug: Print extracted text
            print("Final extracted text data:")
            for item in text_data:
                print(f"  - {item['text']} (confidence: {item['confidence']:.2f})")
            
            # Extract individual fields
            pan_number = self.find_pan_number(text_data)
            dates = self.find_dates(text_data)
            names = self.find_names(text_data)
            
            # Validate and clean results
            result = {
                'pan_number': pan_number if self.validate_pan(pan_number) else None,
                'name': names.get('name', ''),
                'father_name': names.get('father_name', ''),
                'date_of_birth': None,
                'extraction_confidence': 'high' if pan_number and names.get('name') else 'medium',
                'raw_text': [item['text'] for item in text_data]
            }
            
            # Find most likely date of birth
            valid_dates = [date for date in dates if self.validate_date(date)]
            if valid_dates:
                result['date_of_birth'] = valid_dates[0]  # Take first valid date
            
            # Calculate overall confidence
            confidence_score = 0
            if result['pan_number']: confidence_score += 40
            if result['name']: confidence_score += 30
            if result['father_name']: confidence_score += 20
            if result['date_of_birth']: confidence_score += 10
            
            result['confidence_score'] = confidence_score
            result['extraction_confidence'] = 'high' if confidence_score >= 70 else 'medium' if confidence_score >= 40 else 'low'
            
            return result
            
        except Exception as e:
            import traceback
            print(f"Detailed error: {traceback.format_exc()}")
            return {'error': f'Error processing image: {str(e)}'}

    def save_results(self, results: Dict, output_path: str = 'pan_extraction_results.json'):
        """Save extraction results to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path}")

# Example usage
def main():
    # Initialize extractor
    extractor = PANCardExtractor()
    
    # Extract details from PAN card image
    image_path = "/home/rishabh/openbharatocr/openbharatocr/ocr/PAN2.jpeg"  # Replace with your actual image path
    
    try:
        results = extractor.extract_pan_details(image_path)
        
        print("=== PAN Card Extraction Results ===")
        print(f"PAN Number: {results.get('pan_number', 'Not found')}")
        print(f"Name: {results.get('name', 'Not found')}")
        print(f"Father's Name: {results.get('father_name', 'Not found')}")
        print(f"Date of Birth: {results.get('date_of_birth', 'Not found')}")
        print(f"Confidence: {results.get('extraction_confidence', 'Unknown')}")
        print(f"Confidence Score: {results.get('confidence_score', 0)}/100")
        
        if 'error' in results:
            print(f"Error: {results['error']}")
        
        # Save results
        extractor.save_results(results)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()