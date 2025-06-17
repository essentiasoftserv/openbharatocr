import cv2
import numpy as np
import re
from datetime import datetime
from paddleocr import PaddleOCR
import json
from typing import Dict, List, Tuple, Optional

class PANCardExtractor:
    def __init__(self):
        # Set up PaddleOCR - using English for now but could add more languages later
        self.ocr = PaddleOCR(lang='en')
        
        # This regex should catch all valid PAN numbers (5 letters, 4 digits, 1 letter)
        self.pan_pattern = r'[A-Z]{5}[0-9]{4}[A-Z]{1}'
        
        # Common date formats we see on PAN cards - could expand this if needed
        self.date_patterns = [
            r'\d{2}/\d{2}/\d{4}',
            r'\d{2}-\d{2}-\d{4}',
            r'\d{2}\.\d{2}\.\d{4}'
        ]
        
        # Keywords that typically appear near different fields on PAN cards
        # Adding both English and Hindi terms since PAN cards have both
        self.field_keywords = {
            'name': ['name', 'श्री', 'श्रीमती', 'कुमारी', 'shri', 'smt', 'kumari'],
            'father_name': ['father', 'पिता', "father's name", 'fathers name', 'father name', 'पिता का नाम', 'पिताजी'],
            'dob': ['birth', 'जन्म', 'date of birth', 'dob', 'born', 'जन्म तिथि'],
            'pan': ['permanent account number', 'pan', 'account number']
        }
        
        # Common title prefixes that we want to filter out from names
        self.name_titles = ['श्री', 'श्रीमती', 'कुमारी', 'shri', 'smt', 'kumari', 'mr', 'mrs', 'ms', 'dr', 'sh']
        
        # Words to ignore when trying to extract names - these show up a lot on PAN cards
        # but are definitely not names. Added some common OCR mistakes I've seen too
        self.exclude_words = [
            'government', 'india', 'income', 'tax', 'department', 'permanent', 'account', 
            'number', 'signature', 'photo', 'card', 'pan', 'specimen', 'copy', 'original',
            'भारत', 'सरकार', 'आयकर', 'विभाग', 'स्थायी', 'खाता', 'संख्या', 'हस्ताक्षर',
            'फोटो', 'प्रति', 'मूल', 'govt', 'of', 'deartment', 'govtofindia', 'incometax',
            'pemanentaoun', 'nambercard', 'danotbth', 'hra', 'rr', 'bitor', 'fenhтst',
            'ee', 'enrsh', 'fomse'  # These are OCR errors I've encountered
        ]
        
        # Patterns that match typical Indian names - helps validate if something is actually a name
        self.indian_name_patterns = [
            r'^[A-Z][a-z]+ [A-Z][a-z]+$',  # Standard First Last format
            r'^[A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+$',  # First Middle Last
            r'^[A-Z]+ [A-Z]+ [A-Z]+$',  # Sometimes names are in all caps
            r'^[A-Z][A-Z]+ [A-Z][A-Z]+$',  # Multiple capital letters
            r'^[A-Z][a-z]+[A-Z][a-z]+$',  # Sometimes first and last are combined
        ]
        
        # Regex patterns to catch text we definitely want to ignore
        # This helps filter out boilerplate text from PAN cards
        self.ignore_text_patterns = [
            r'income.*tax.*department',
            r'govt.*of.*india',
            r'permanent.*account.*number',
            r'signature',
            r'date.*of.*birth',
            r'^[A-Z]{5}[0-9]{4}[A-Z]$',  # This would be the PAN number itself
            r'^\d{2}/\d{2}/\d{4}$',  # Date patterns
            r'^[0-9]+$',  # Just numbers
            r'^[A-Z]$',  # Single letters (probably OCR artifacts)
            r'^[A-Z]{1,3}$',  # Short abbreviations
            r'card',
            r'number',
            r'permanent',
            r'account',
            r'specimen',
            r'hra\s+rr',  # Specific OCR error I keep seeing
            r'bitor',
            r'fenhтst',
            r'ee',
            r'enrsh',
            r'fomse'
        ]

    def preprocess_image(self, image_path: str) -> np.ndarray:
        # Load the image and do some basic error checking
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Convert to RGB since that's what most processing expects
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Scale up small images - OCR works better on larger images
        height, width = img_rgb.shape[:2]
        if width < 800:
            scale = 800 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            img_rgb = cv2.resize(img_rgb, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # Enhance contrast using CLAHE - helps with poor quality scans
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Remove noise which can confuse OCR
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        # Sharpen the image to make text clearer
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # Convert back to RGB for final processing
        final_img = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
        
        return final_img

    def extract_text_with_coordinates(self, image: np.ndarray) -> List[Tuple]:
        # Try to run OCR - different versions of PaddleOCR have different interfaces
        try:
            results = self.ocr.predict(image)
        except (AttributeError, TypeError):
            try:
                results = self.ocr.ocr(image)
            except:
                results = self.ocr(image)
        
        extracted_data = []
        
        # Handle the results - format can vary between PaddleOCR versions
        if results and isinstance(results, list) and len(results) > 0:
            result_dict = results[0]
            
            # This handles the newer PaddleX format
            if isinstance(result_dict, dict) and 'rec_texts' in result_dict:
                texts = result_dict.get('rec_texts', [])
                scores = result_dict.get('rec_scores', [])
                polygons = result_dict.get('rec_polys', [])
                
                print("Extracted text from PaddleX format:")
                for i, text in enumerate(texts):
                    confidence = scores[i] if i < len(scores) else 0.8
                    bbox = polygons[i] if i < len(polygons) else []
                    
                    print(f"  - {text} (confidence: {confidence:.2f})")
                    
                    # Only keep text with decent confidence
                    if confidence > 0.3: 
                        if len(bbox) > 0:
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
            
            # This handles the standard PaddleOCR format
            elif isinstance(result_dict, list):
                lines = result_dict
                for line in lines:
                    try:
                        if line and isinstance(line, (list, tuple)) and len(line) >= 2:
                            bbox = line[0]
                            text_info = line[1]
                            
                            # Extract text and confidence
                            if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                                text = text_info[0]
                                confidence = text_info[1]
                            else:
                                text = str(text_info)
                                confidence = 0.8  # Default confidence if not provided
                            
                            # Only keep high enough confidence text
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
        # Basic text cleaning - normalize whitespace and remove weird characters
        cleaned = re.sub(r'\s+', ' ', text)
        cleaned = re.sub(r'[^\w\s/.-]', '', cleaned)
        return cleaned.strip()

    def clean_name(self, name: str) -> str:
        # Clean up extracted names to make them look proper
        if not name:
            return ""
        
        name = name.strip()
        
        # Split into words and filter out titles/prefixes
        words = name.split()
        cleaned_words = []
        
        for word in words:
            # Skip common titles and single letters (usually OCR errors)
            if (word.lower() not in self.name_titles and 
                len(word) > 1 and 
                word.lower() not in ['sh', 'smt', 'shri']):
                cleaned_words.append(word)
        
        if not cleaned_words:
            return ""
            
        cleaned_name = ' '.join(cleaned_words)
        
        # Remove special characters but keep spaces
        cleaned_name = re.sub(r'[^\w\s]', '', cleaned_name)
        
        # Make it look like a proper name (Title Case)
        cleaned_name = cleaned_name.title()
        
        return cleaned_name.strip()

    def is_valid_name(self, text: str, min_confidence: float = 0.6) -> bool:
        # Check if some text actually looks like a real Indian name
        if not text or len(text) < 3:
            return False
            
        # Clean first then check
        cleaned = self.clean_text(text).lower()
        
        # Run through our ignore patterns first
        for pattern in self.ignore_text_patterns:
            if re.search(pattern, cleaned):
                print(f"  -> Rejected '{text}' due to ignore pattern: {pattern}")
                return False
        
        # Check against our exclude word list
        for exclude_word in self.exclude_words:
            if exclude_word in cleaned:
                print(f"  -> Rejected '{text}' due to exclude word: {exclude_word}")
                return False
        
        # Names shouldn't have numbers in them
        if re.search(r'\d', cleaned):
            print(f"  -> Rejected '{text}' due to containing digits")
            return False
        
        # Should be at least 2 words for a full name, or one long word
        words = cleaned.split()
        if len(words) < 2 and len(cleaned) < 6:
            print(f"  -> Rejected '{text}' due to insufficient length/words")
            return False
            
        # Each word should look like part of a name
        for word in words:
            if not word[0].isalpha() or len(word) < 2:
                print(f"  -> Rejected '{text}' due to invalid word: {word}")
                return False
                
        # Check if it has a reasonable mix of vowels and consonants
        # Real names usually have both
        vowels = sum(1 for char in cleaned if char in 'aeiou')
        consonants = sum(1 for char in cleaned if char.isalpha() and char not in 'aeiou')
        
        if vowels == 0 or consonants == 0:
            print(f"  -> Rejected '{text}' due to lack of vowels/consonants")
            return False
            
        # Check vowel ratio - too many or too few vowels is suspicious
        if len(cleaned) > 4:
            vowel_ratio = vowels / len([c for c in cleaned if c.isalpha()])
            if vowel_ratio < 0.1 or vowel_ratio > 0.8:
                print(f"  -> Rejected '{text}' due to unusual vowel ratio: {vowel_ratio}")
                return False
        
        print(f"  -> Accepted '{text}' as valid name")
        return True

    def is_likely_name(self, text: str) -> bool:
        # Keeping this for compatibility with any old code
        return self.is_valid_name(text)

    def find_pan_number(self, text_data: List[Dict]) -> Optional[str]:
        # Look for PAN numbers in the extracted text
        for item in text_data:
            text = self.clean_text(item['text'])
            pan_match = re.search(self.pan_pattern, text.upper())
            if pan_match:
                return pan_match.group()
            
            # Sometimes OCR adds spaces in PAN numbers, so check without spaces too
            text_upper = text.upper().replace(' ', '')
            if re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$', text_upper):
                return text_upper
        
        return None

    def find_dates(self, text_data: List[Dict]) -> List[str]:
        # Extract all date-like strings from the text
        dates = []
        for item in text_data:
            text = self.clean_text(item['text'])
            for pattern in self.date_patterns:
                matches = re.findall(pattern, text)
                dates.extend(matches)
        return dates

    def extract_names_with_keywords(self, text_data: List[Dict]) -> Tuple[Optional[str], Optional[str]]:
        # Try to find names by looking for keywords like "Name:" or "Father's Name:"
        name = None
        father_name = None
        
        for i, item in enumerate(text_data):
            text = item['text'].lower()
            
            # Look for name-related keywords
            if any(keyword in text for keyword in self.field_keywords['name']):
                # Check the next few lines for the actual name
                for j in range(i + 1, min(i + 3, len(text_data))):
                    candidate = text_data[j]['text']
                    if self.is_valid_name(candidate):
                        name = self.clean_name(candidate)
                        print(f"  -> Found name using keyword: {name}")
                        break
                
                # Also check if the name is on the same line after the keyword
                if not name:
                    for keyword in self.field_keywords['name']:
                        if keyword in text:
                            parts = text.split(keyword)
                            if len(parts) > 1 and parts[1].strip():
                                candidate_name = parts[1].strip()
                                candidate_name = re.sub(r'^[/\s]*', '', candidate_name)
                                if len(candidate_name) > 2 and self.is_valid_name(candidate_name):
                                    name = self.clean_name(candidate_name)
                                    print(f"  -> Found name in same line: {name}")
                                    break
            
            # Look for father's name keywords
            elif any(keyword in text for keyword in self.field_keywords['father_name']):
                # Check next few lines for father's name
                for j in range(i + 1, min(i + 3, len(text_data))):
                    candidate = text_data[j]['text']
                    if self.is_valid_name(candidate):
                        father_name = self.clean_name(candidate)
                        print(f"  -> Found father's name using keyword: {father_name}")
                        break
                
                # Check same line after keyword
                if not father_name:
                    for keyword in self.field_keywords['father_name']:
                        if keyword in text:
                            parts = text.split(keyword)
                            if len(parts) > 1 and parts[1].strip():
                                candidate_name = parts[1].strip()
                                candidate_name = re.sub(r'^[/\s]*', '', candidate_name)
                                candidate_name = re.sub(r'^s\s+name', '', candidate_name)  # Fix OCR error "father s name"
                                
                                if len(candidate_name) > 2 and self.is_valid_name(candidate_name):
                                    father_name = self.clean_name(candidate_name)
                                    print(f"  -> Found father's name in same line: {father_name}")
                                    break
        
        return name, father_name

    def extract_names_positional(self, text_data: List[Dict]) -> Tuple[Optional[str], Optional[str]]:
        # If keywords don't work, try to guess names based on position and quality
        valid_candidates = []
        
        print("\n=== Analyzing candidates for positional extraction ===")
        
        for item in text_data:
            print(f"Checking: '{item['text']}' (confidence: {item['confidence']:.2f})")
            
            # Be pickier about what we consider valid names
            if (item['confidence'] >= 0.8 and 
                self.is_valid_name(item['text']) and 
                len(item['text'].split()) >= 2):
                
                cleaned_name = self.clean_name(item['text'])
                
                # Make sure the cleaned name is still reasonable
                if cleaned_name and len(cleaned_name) > 5:
                    valid_candidates.append(cleaned_name)
                    print(f"  -> Added to candidates: {cleaned_name}")
        
        # Remove duplicates but keep order
        seen = set()
        unique_candidates = []
        for candidate in valid_candidates:
            if candidate.lower() not in seen:
                seen.add(candidate.lower())
                unique_candidates.append(candidate)
        
        print(f"\n  -> Final valid name candidates: {unique_candidates}")
        
        # Usually on PAN cards, cardholder name comes first, then father's name
        name = None
        father_name = None
        
        if len(unique_candidates) >= 1:
            # Sort by vertical position to get the right order
            candidate_with_pos = []
            for candidate in unique_candidates:
                # Find where this candidate appears in the original data
                for item in text_data:
                    if self.clean_name(item['text']) == candidate:
                        candidate_with_pos.append((candidate, item['center_y']))
                        break
            
            # Sort from top to bottom
            candidate_with_pos.sort(key=lambda x: x[1])
            sorted_candidates = [candidate for candidate, _ in candidate_with_pos]
            
            # First valid name is usually the cardholder
            name = sorted_candidates[0] if len(sorted_candidates) >= 1 else None
            
            # Second valid name is usually father's name
            if len(sorted_candidates) >= 2:
                father_name = sorted_candidates[1]
        
        return name, father_name

    def find_names_improved(self, text_data: List[Dict]) -> Dict[str, str]:
        # Main name extraction logic - tries multiple approaches
        sorted_data = sorted(text_data, key=lambda x: x['center_y'])
        
        print("\n=== Analyzing text for names ===")
        
        # Try keyword-based approach first (more reliable)
        name, father_name = self.extract_names_with_keywords(sorted_data)
        
        # Fall back to positional approach if we're missing names
        if not name or not father_name:
            print("\n=== Using positional extraction as fallback ===")
            pos_name, pos_father = self.extract_names_positional(sorted_data)
            
            if not name:
                name = pos_name
                if name:
                    print(f"  -> Assigned positional name: {name}")
            
            if not father_name:
                father_name = pos_father
                if father_name:
                    print(f"  -> Assigned positional father's name: {father_name}")
        
        # Sanity check - if both names are identical, something went wrong
        if name and father_name and name.lower() == father_name.lower():
            print("  -> Names are identical, looking for alternative...")
            all_valid_names = []
            for item in sorted_data:
                if self.is_valid_name(item['text']) and item['confidence'] >= 0.8:
                    clean_candidate = self.clean_name(item['text'])
                    if clean_candidate and clean_candidate.lower() not in [name.lower()]:
                        all_valid_names.append(clean_candidate)
            
            if all_valid_names:
                father_name = all_valid_names[0]
                print(f"  -> Updated father's name to: {father_name}")
        
        return {
            'name': name or '',
            'father_name': father_name or ''
        }

    def validate_pan(self, pan: str) -> bool:
        # Check if PAN number follows the correct format
        if not pan:
            return False
        return bool(re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]{1}$', pan))

    def validate_date(self, date_str: str) -> bool:
        # Check if date is valid and reasonable for a birth date
        try:
            for fmt in ['%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y']:
                try:
                    date_obj = datetime.strptime(date_str, fmt)
                    current_year = datetime.now().year
                    age = current_year - date_obj.year
                    # Person should be between 1 and 120 years old
                    return 1 <= age <= 120
                except ValueError:
                    continue
            return False
        except:
            return False

    def extract_pan_details(self, image_path: str) -> Dict:
        # Main function that ties everything together
        try:
            # Process the image to make OCR more accurate
            processed_img = self.preprocess_image(image_path)
            
            # Extract text from the processed image
            text_data = self.extract_text_with_coordinates(processed_img)
            
            if not text_data:
                return {'error': 'No text could be extracted from the image'}
            
            print("Final extracted text data:")
            for item in text_data:
                print(f"  - {item['text']} (confidence: {item['confidence']:.2f})")
            
            # Extract different types of information
            pan_number = self.find_pan_number(text_data)
            dates = self.find_dates(text_data)
            names = self.find_names_improved(text_data)
            
            # Build the final result
            result = {
                'pan_number': pan_number if self.validate_pan(pan_number) else None,
                'name': names.get('name', ''),
                'father_name': names.get('father_name', ''),
                'date_of_birth': None,
                'extraction_confidence': 'high' if pan_number and names.get('name') else 'medium',
                'raw_text': [item['text'] for item in text_data]
            }
            
            # Find valid birth date from extracted dates
            valid_dates = [date for date in dates if self.validate_date(date)]
            if valid_dates:
                result['date_of_birth'] = valid_dates[0]
            
            # Calculate confidence score based on what we found
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
        # Save the extraction results to a JSON file for later use
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path}")

def main():
    # Simple test function to try out the extractor
    extractor = PANCardExtractor()
    
    image_path = "/home/rishabh/openbharatocr/openbharatocr/ocr/pan_sample/37.jpg"
    
    try:
        results = extractor.extract_pan_details(image_path)
        
        print("\n=== PAN Card Extraction Results ===")
        print(f"PAN Number: {results.get('pan_number', 'Not found')}")
        print(f"Name: {results.get('name', 'Not found')}")
        print(f"Father's Name: {results.get('father_name', 'Not found')}")
        print(f"Date of Birth: {results.get('date_of_birth', 'Not found')}")
        print(f"Confidence: {results.get('extraction_confidence', 'Unknown')}")
        print(f"Confidence Score: {results.get('confidence_score', 0)}/100")
        
        if 'error' in results:
            print(f"Error: {results['error']}")
        
        # Save results to file
        extractor.save_results(results)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()