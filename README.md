# OpenBharatOCR
[![Build status](https://github.com/essentiasoftserv/openbharatocr/actions/workflows/main.yml/badge.svg)](https://github.com/essentiasoftserv/openbharatocr/actions/workflows/main.yml)

OpenBharatOCR is an open-source Python library specifically designed for optical character recognition (OCR) of Indian government documents.

## Key Features
- **Comprehensive Document Support**: Extract text from major Indian government documents including Aadhaar Card, PAN Card, Driving License, Passport, Voter ID, and more
- **Multi-Language OCR**: Support for English and Hindi text extraction
- **Advanced Image Processing**: Built-in preprocessing techniques for enhanced accuracy
- **Multiple OCR Engines**: Leverages PaddleOCR, EasyOCR, and Tesseract for optimal results
- **Pattern Matching**: Document-specific field extraction with validation 

## Prerequisites

- **Python**: 3.6 or later
- **Operating System**: Linux (Ubuntu/Debian preferred), Windows (via WSL2), or macOS
- **System Dependencies**: Tesseract OCR (for pytesseract functionality)

## Installation

### Install from PyPI

```bash
pip install openbharatocr
```

### Development Setup

1. **Clone the repository:**
```bash
git clone https://github.com/essentiasoftserv/openbharatocr.git
cd openbharatocr
```

2. **Create a virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install in development mode:**
```bash
pip install -e .
```

## Supported Documents

### PAN Card
Extract information from Permanent Account Number cards.

```python
import openbharatocr 

# Extract PAN card details
result = openbharatocr.pan(image_path)
# Returns: {'name': str, 'father_name': str, 'dob': str, 'pan_number': str}
```

### Aadhaar Card
Process both front and back sides of Aadhaar cards.

```python
import openbharatocr 

# Front side
front_result = openbharatocr.front_aadhaar(image_path)
# Returns: {'name': str, 'dob': str, 'gender': str, 'aadhaar_number': str}

# Back side
back_result = openbharatocr.back_aadhaar(image_path)
# Returns: {'address': str, 'aadhaar_number': str, 'pin_code': str}
```

### Driving License
Extract details from Indian driving licenses.

```python
import openbharatocr 

result = openbharatocr.driving_licence(image_path)
# Returns: {'name': str, 'license_number': str, 'dob': str, 'validity': str, 'address': str}
```

### Passport
Process Indian passport information pages.

```python
import openbharatocr 

result = openbharatocr.passport(image_path)
# Returns: {'name': str, 'passport_number': str, 'dob': str, 'doi': str, 'doe': str}
```

### Voter ID
Extract information from both sides of Voter ID cards.

```python
import openbharatocr 
import os

# Note: Requires YOLO model files for enhanced accuracy
# Set environment variables for YOLO model paths:
os.environ['YOLO_CFG'] = 'path/to/yolo.cfg'
os.environ['YOLO_WEIGHT'] = 'path/to/yolo.weights'

# Front side
front_result = openbharatocr.voter_id_front(image_path)
# Returns: {'name': str, 'voter_id': str, 'father_name': str, 'dob': str}

# Back side
back_result = openbharatocr.voter_id_back(image_path)
# Returns: {'address': str, 'voter_id': str}
```

### Vehicle Registration Card/Certificate
Extract vehicle registration details.

```python
import openbharatocr 

result = openbharatocr.vehicle_registration(image_path)
# Returns: {'registration_number': str, 'owner_name': str, 'vehicle_model': str, 
#          'registration_date': str, 'chassis_number': str, 'engine_number': str}
```

### Water Bill
Process water utility bills.

```python
import openbharatocr 

result = openbharatocr.water_bill(image_path)
# Returns: {'consumer_number': str, 'name': str, 'address': str, 
#          'bill_date': str, 'amount': str}
```

### Birth Certificate
Extract information from birth certificates.

```python
import openbharatocr 

result = openbharatocr.birth_certificate(image_path)
# Returns: {'name': str, 'dob': str, 'father_name': str, 'mother_name': str, 
#          'registration_number': str, 'registration_date': str}
```

### Degree Certificate
Process educational degree certificates.

```python
import openbharatocr 

result = openbharatocr.degree(image_path)
# Returns: {'name': str, 'degree': str, 'university': str, 'year': str, 'grade': str}
```

### Bank Passbook
Extract bank passbook details (if available).

```python
import openbharatocr 

# Note: Check if passbook functionality is exposed in the API
result = openbharatocr.passbook(image_path)  # If available
# Returns: {'account_number': str, 'name': str, 'bank_name': str, 'branch': str, 'ifsc': str}
```

## Additional Resources

### YOLO Models for Enhanced Voter ID Processing
For optimal Voter ID extraction, download the following YOLO v3 models:

- **Configuration File**: [Download YOLO Config](https://drive.google.com/file/d/1SEst2lVoFDOgUVLZ5kje9GTb2tHRA8U-/view?usp=sharing)
- **Weights File**: [Download YOLO Weights](https://drive.google.com/file/d/1cGGstycfogmO6O7ToB2DAEXOgTWVgINh/view?usp=drive_link)

After downloading, set the file paths in environment variables:
```python
import os
os.environ['YOLO_CFG'] = '/path/to/yolov3.cfg'
os.environ['YOLO_WEIGHT'] = '/path/to/yolov3.weights'
```

## Contributing

We welcome contributions to OpenBharatOCR! Whether you're fixing bugs, improving documentation, or adding new features, your help is appreciated.

### Development Guidelines

1. **Fork and Clone**: Fork the repository and clone your fork locally
2. **Create a Branch**: Create a feature branch for your changes
3. **Write Tests**: Add tests for any new functionality
4. **Follow Code Style**: Use Black formatter and follow PEP 8 guidelines
5. **Run Pre-commit Hooks**: Before committing, run:
   ```bash
   pre-commit run --all-files
   ```

### Testing

Run the test suite:
```bash
pytest openbharatocr/unit_tests/
```

Run code quality checks:
```bash
# Format code with Black
black openbharatocr/

# Check for spelling errors
codespell

# Run all pre-commit hooks
pre-commit run --all-files
```

### Reporting Issues

Found a bug or have a feature request? Please create an issue:
[https://github.com/essentiasoftserv/openbharatocr/issues](https://github.com/essentiasoftserv/openbharatocr/issues)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Authors

- **Kunal Kumar Kushwaha** - [essentia.dev](http://www.essentia.dev)
- **Contributors** - See [Contributors](https://github.com/essentiasoftserv/openbharatocr/graphs/contributors)

## Acknowledgments

- PaddleOCR team for the excellent OCR engine
- EasyOCR project for multilingual support
- Tesseract OCR community
- All contributors who have helped improve this project

