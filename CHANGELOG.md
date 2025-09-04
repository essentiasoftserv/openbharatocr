# Changelog

All notable changes to the OpenBharatOCR project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.2] - 2024

### Added
- Enhanced PaddleOCR integration with multiple fallback initialization strategies
- Improved image preprocessing with CLAHE, denoising, and sharpening techniques
- Support for region-of-interest cropping for better accuracy
- Comprehensive bank list for passbook processing
- Better error handling and fallback mechanisms
- Extended classifiers in setup.py for better package discovery

### Changed
- Updated documentation with clearer examples and return types
- Improved requirements.txt organization with categorized dependencies
- Enhanced setup.py with proper encoding and metadata
- Refactored __init__.py with proper convenience functions

### Fixed
- Fixed import issues with paddlex OCRResult
- Improved text extraction confidence filtering
- Better handling of multi-language text in documents

## [0.4.1] - Previous Release

### Added
- Water bill document support
- Birth certificate OCR functionality
- Degree certificate processing

### Changed
- Upgraded PaddlePaddle to version 3.0.0
- Updated EasyOCR to version 1.7.1

## [0.4.0] - Previous Release

### Added
- Vehicle registration card/certificate support
- Enhanced Voter ID processing with YOLO models
- Passbook OCR functionality

### Changed
- Improved pattern matching for document-specific fields
- Better handling of Hindi text alongside English

## [0.3.0] - Previous Release

### Added
- Passport OCR support
- Driving license extraction
- Multi-language support (English and Hindi)

### Changed
- Migrated from pure Tesseract to PaddleOCR as primary engine
- Improved accuracy for Aadhaar and PAN cards

## [0.2.0] - Initial Public Release

### Added
- Basic PAN card OCR functionality
- Aadhaar card front and back processing
- Initial test suite
- CI/CD pipeline with AWS CodeBuild

### Changed
- Project structure reorganization
- Added pre-commit hooks

## [0.1.0] - Internal Release

### Added
- Initial project setup
- Basic OCR functionality with Tesseract
- Support for PAN and Aadhaar cards

---

## Upcoming Features (Roadmap)

### Planned for Next Release
- [ ] Ration card support
- [ ] Electricity bill processing
- [ ] Property documents OCR
- [ ] Enhanced multi-page document support
- [ ] REST API interface
- [ ] Docker containerization
- [ ] GPU acceleration support
- [ ] Batch processing capabilities
- [ ] Cloud storage integration (S3, Google Cloud Storage)
- [ ] Web interface for testing

### Long-term Goals
- [ ] Support for all Indian regional languages
- [ ] Machine learning-based field validation
- [ ] Automated document classification
- [ ] Integration with government APIs for verification
- [ ] Mobile SDK (iOS/Android)