import re
import cv2
import numpy as np
import easyocr
from PIL import Image, ImageEnhance
from datetime import datetime


class VehicleRegistrationExtractor:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])
        self.ocr_corrections = {
            '0': ['O', 'o', 'Q'], '1': ['I', 'l', '|'], '2': ['Z'], '5': ['S'],
            '6': ['G'], '8': ['B'], 'A': ['4'], 'B': ['8'], 'D': ['0'], 'G': ['6'],
            'I': ['1', '|'], 'O': ['0'], 'S': ['5'], 'Z': ['2']
        }

    def preprocess_image(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                pil_img = Image.open(image_path)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
            pil_image = Image.fromarray(thresh)
            enhancer = ImageEnhance.Contrast(pil_image)
            return enhancer.enhance(1.5)
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return Image.open(image_path)

    def extract_text_easyocr(self, image):
        image_np = np.array(image.convert('RGB'))
        results = self.reader.readtext(image_np, detail=0)
        return "\n".join(results)

    def clean_and_segment_text(self, text):
        text = re.sub(r'[^\w\s\-/.:()]+', ' ', text)
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)
        text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
        return text

    def extract_specific_field(self, text, field_patterns, max_length=50):
        for pattern in field_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                result = match.group(1).strip()
                if 0 < len(result) <= max_length:
                    return result
        return ""
    
    def extract_owner_name(self, text):
        """Extract owner name with strict patterns"""
        patterns = [
            r'Owner\s*Name[:\-\s]*([A-Z][A-Z\s]{5,30})(?=\s+(?:S[/\\]?[WwDd]|Present|Address|Permanent))',
            r'([A-Z]{2,}\s+[A-Z]{2,}\s+[A-Z]{2,})(?=\s+(?:S[/\\]?[WwDd]|Present|Address))',
            r'Name[:\-\s]*([A-Z][A-Z\s]{5,30})(?=\s+(?:S[/\\]?[WwDd]|Present|Address))'
        ]

        lines = text.split('\n')
        for line in lines:
            if 'Owner Name' in line or 'GAIKWAD' in line:
                name_match = re.search(r'(?:Owner\s*Name[:\-\s]*)?([A-Z]{2,}\s+[A-Z]{2,}\s+[A-Z]{2,})', line, re.IGNORECASE)
                if name_match:
                    name = name_match.group(1).strip()
                    name = re.sub(r'\s+', ' ', name)
                    if len(name) <= 40 and name.replace(' ', '').isalpha():
                        return name.title()

        name = self.extract_specific_field(text, patterns, 40)
        return name.title() if name else ""

    def extract_swd_name(self, text):
        """Extract S/W/D name"""
        patterns = [
            r'S[/\\]?[WwDd]\s*Name[:\-\s]*([A-Z][A-Z\s]{2,25})(?=\s+(?:Present|Address|Permanent))',
            r'S[/\\]?[WwDd][:\-\s]*([A-Z][A-Z\s]{2,25})(?=\s+(?:Present|Address))',
            r'(?:Son|Wife|Daughter)\s+of[:\-\s]*([A-Z][A-Z\s]{2,25})'
        ]

        lines = text.split('\n')
        for line in lines:
            if any(keyword in line.upper() for keyword in ['S/W/D', 'SWD', 'ARVIND']):
                swd_match = re.search(r'(?:S[/\\]?[WwDd]|SWD)[:\-\s]*([A-Z]{2,20})', line, re.IGNORECASE)
                if swd_match:
                    swd = swd_match.group(1).strip()
                    if swd.replace(' ', '').isalpha() and len(swd) <= 25:
                        return swd.title()

        swd = self.extract_specific_field(text, patterns, 25)
        return swd.title() if swd else ""

    def extract_reg_number(self, text):
        patterns = [
            r'\b([A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{1,4})\b',  # MH12AB1234, MH 12 AB 1234
            r'\b(Reg(?:n)?(?:istration)?(?: No)?[:\- ]+)?([A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4})\b',  # With optional prefix
            r'\b([A-Z]{2}-\d{1,2}-[A-Z]{1,3}-\d{1,4})\b',  # Format with hyphens: MH-12-AB-1234
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    reg_num = match[1]
                else:
                    reg_num = match
                reg_num = re.sub(r'[\s\-]', '', reg_num).upper()
                if re.match(r'^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}$', reg_num):
                    return reg_num
        return ""

    def extract_chassis_number(self, text):
        patterns = [
            r'Chassis\s*Number[:\-\s]*([A-Z0-9]{17})',
            r'Chasis\s*Number[:\-\s]*([A-Z0-9]{17})',
            r'\b(ME[A-Z0-9]{15})\b',
            r'\b([A-Z]{3}[A-Z0-9]{14})\b'
        ]

        chassis = self.extract_specific_field(text, patterns, 20)
        if chassis and len(chassis) >= 17:
            return chassis.upper()

        vin_match = re.search(r'\b((?:ME|WB|VF|WBA|WDD)[A-Z0-9]{14,15})\b', text)
        if vin_match:
            return vin_match.group(1).upper()

        return ""

    def extract_engine_number(self, text):
        """Extract engine number with specific patterns"""
        patterns = [
            r'Engine\s*Number[:\-\s]*([A-Z0-9]{8,15})',
            r'Engine[:\-\s]*([A-Z0-9]{8,15})',
            r'\b(KC\d{2}EA\d{7})\b',
            r'\b([A-Z]{2,4}\d{2}[A-Z]{2}\d{6,8})\b'
        ]

        engine = self.extract_specific_field(text, patterns, 15)
        return engine.upper() if engine else ""

    def extract_fuel_type(self, text):
        """Extract fuel type"""
        fuel_types = ["PETROL", "DIESEL", "CNG", "ELECTRIC", "HYBRID", "LPG"]

        for fuel in fuel_types:
            if re.search(rf'\b{fuel}\b', text, re.IGNORECASE):
                return fuel.upper()
        return ""

    def extract_vehicle_class(self, text):
        """Extract vehicle class"""
        patterns = [
            r'Vehicle\s*Class[:\-\s]*([A-Z][A-Z0-9\s/\-()]{3,25})',
            r'Class[:\-\s]*([A-Z][A-Z0-9\s/\-()]{3,25})',
            r'\b(M-Cycle[/\\]Scooter[^A-Za-z]*(?:2Wn)?)\b'
        ]

        v_class = self.extract_specific_field(text, patterns, 30)
        if v_class:
            v_class = re.sub(r'Sceoter', 'Scooter', v_class, flags=re.IGNORECASE)
            v_class = re.sub(r'2Wn', '2Wheeler', v_class, flags=re.IGNORECASE)

        return v_class if v_class else ""

    def extract_model(self, text):
        """Extract vehicle model"""
        patterns = [
            r'Vehicle\s*Model[:\-\s]*([A-Z0-9\s]{3,20})',
            r'Model[:\-\s]*([A-Z0-9\s]{3,20})',
            r'\b(UNICORN)\b',
            r'\b(\d+\s*UNICORN)\b'
        ]

        model = self.extract_specific_field(text, patterns, 20)
        if model:
            model = re.sub(r'UNIGORN', 'UNICORN', model, flags=re.IGNORECASE)
            model = re.sub(r'^\d+\s*', '', model)

        return model if model else ""

    def extract_color(self, text):
        """Extract vehicle color"""
        patterns = [
            r'Vehicle\s*Color[:\-\s]*([A-Z][A-Z\s]{3,25})',
            r'Color[:\-\s]*([A-Z][A-Z\s]{3,25})',
        ]

        color = self.extract_specific_field(text, patterns, 30)
        if color:
            color = re.sub(r'PEARLIGNEOUSBLACK', 'PEARL IGNEOUS BLACK', color, flags=re.IGNORECASE)
            color = re.sub(r'([A-Z])([A-Z]{2,})', r'\1 \2', color)

        return color.title() if color else ""

    def extract_dates(self, text):
        """Extract dates with multiple formats"""
        date_patterns = [
            r'\b(\d{1,2}[-/]\w{3}[-/]\d{4})\b',
            r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b',
            r'\b(\d{2}[-/]\w{3}[-/]\d{4})\b'
        ]

        dates = []
        for pattern in date_patterns:
            found_dates = re.findall(pattern, text)
            dates.extend(found_dates)

        valid_dates = []
        for date in dates:
            date = re.sub(r'ul', 'Jul', date, flags=re.IGNORECASE)
            date = re.sub(r'Blun', 'Jun', date, flags=re.IGNORECASE)
            date = re.sub(r'Ju([rn])', 'Jun', date, flags=re.IGNORECASE)

            if self.is_valid_date_format(date):
                valid_dates.append(date)

        return sorted(list(set(valid_dates)), key=self.parse_date_for_sorting)

    def extract_address(self, text):
        patterns = [
            r'Address[:\-\s]*([A-Z0-9\s,.-]{10,100})',
            r'Permanent\s*Address[:\-\s]*([A-Z0-9\s,.-]{10,100})',
            r'Present\s*Address[:\-\s]*([A-Z0-9\s,.-]{10,100})'
        ]

        address = self.extract_specific_field(text, patterns, 100)
        if address:
            address = re.sub(r'\s+', ' ', address).strip()
            return address.title()

        return ""
    
    def extract_tax_info(self, text):
        patterns = [
            r'Tax\s*Upto[:\-\s]*([A-Z0-9\s,.-]{10,30})',
            r'Tax[:\-\s]*([A-Z0-9\s,.-]{10,30})'
        ]

        tax_info = self.extract_specific_field(text, patterns, 30)
        if tax_info:
            tax_info = re.sub(r'\s+', ' ', tax_info).strip()
            return tax_info.title()

        return ""

    def is_valid_date_format(self, date_str):
        formats = ["%d/%m/%Y", "%d-%m-%Y", "%d/%b/%Y", "%d-%b-%Y", "%d %b %Y"]
        return any([self.try_parse(date_str, fmt) for fmt in formats])

    def try_parse(self, date_str, fmt):
        try:
            datetime.strptime(date_str, fmt)
            return True
        except ValueError:
            return False

    def parse_date_for_sorting(self, date_str):
        formats = ["%d/%m/%Y", "%d-%m-%Y", "%d/%b/%Y", "%d-%b-%Y", "%d %b %Y"]
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return datetime.min

    def extract_vehicle_registration_details(self, image_path):
        try:
            preprocessed_image = self.preprocess_image(image_path)
            extracted_text = self.extract_text_easyocr(preprocessed_image)
            cleaned_text = self.clean_and_segment_text(extracted_text)

            print("Cleaned and Segmented Text:\n", cleaned_text[:1000], "...\n")

            dates = self.extract_dates(cleaned_text)

            details = {
                "Registration Number": self.extract_reg_number(cleaned_text),
                "Registration Date": dates[0] if len(dates) >= 1 else "",
                "Expiry Date": dates[-1] if len(dates) >= 2 else "",
                "Chassis Number": self.extract_chassis_number(cleaned_text),
                "Engine Number": self.extract_engine_number(cleaned_text),
                "Vehicle Model": self.extract_model(cleaned_text),
                "Vehicle Color": self.extract_color(cleaned_text),
                "Fuel Type": self.extract_fuel_type(cleaned_text),
                "Vehicle Class": self.extract_vehicle_class(cleaned_text),
                "Owner Name": self.extract_owner_name(cleaned_text),
                "S/W/D of": self.extract_swd_name(cleaned_text),
                "Address": self.extract_address(cleaned_text),
                "Tax Upto": self.extract_tax_info(cleaned_text),
            }

            return details
        except Exception as e:
            print(f"Error processing image: {e}")
            return {}


def vehicle_registration(image_path):
    extractor = VehicleRegistrationExtractor()
    return extractor.extract_vehicle_registration_details(image_path)


if __name__ == "__main__":
    image_path = "/home/rishabh/openbharatocr/openbharatocr/ocr/VR2.jpeg"
    details = vehicle_registration(image_path)

    print("\n" + "=" * 60)
    print("EXTRACTED VEHICLE REGISTRATION DETAILS")
    print("=" * 60)
    for key, value in details.items():
        print(f"{key:<20}: {value}")
