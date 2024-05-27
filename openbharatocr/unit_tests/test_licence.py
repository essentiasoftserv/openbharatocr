from openbharatocr.ocr.driving_licence import *
import pytest
from unittest import TestCase
import tempfile
from datetime import datetime, timedelta
import os
from PIL import UnidentifiedImageError


class Test_extract_driving_licence_number(TestCase):
    def test_extract_driving_licence_number_valid_cases(self):
        # Test case 1: Valid driving license number with hyphen
        input_text = (
            "This is a sample text with driving license number DL-1420110012345"
        )
        expected_output1 = "DL-1420110012345"
        assert extract_driving_licence_number(input_text) == expected_output1

        # Test case 2: Valid driving license number with space
        input_text = (
            "This is another sample text with driving license number DL14 20110012345"
        )
        expected_output2 = "DL14 20110012345"
        assert extract_driving_licence_number(input_text) == expected_output2

    def test_extract_driving_licence_number_error_cases(self):
        # Test case 1: Input is None
        input_text = None
        with pytest.raises(TypeError):
            extract_driving_licence_number(input_text)

        # Test case 2: Input is a non-string type
        input_text = 123
        with pytest.raises(TypeError):
            extract_driving_licence_number(input_text)


class Test_extract_all_dates(TestCase):
    def test_extract_all_dates_valid_cases(self):
        # Test case 1: Input with multiple valid dates
        input_text = "This is a sample text with dates 01/01/2000, 15/06/2010, 30/12/2015, and 01/01/2023."
        expected_dob = "01/01/2000"
        expected_doi = ["15/06/2010", "30/12/2015"]
        expected_validity = ["01/01/2023"]
        assert extract_all_dates(input_text) == (
            expected_dob,
            expected_doi,
            expected_validity,
        )

        # Test case 2: Input with only one valid date
        input_text = "This text has a single date: 25/03/2022."
        expected_dob = "25/03/2022"
        expected_doi = []
        expected_validity = []
        assert extract_all_dates(input_text) == (
            expected_dob,
            expected_doi,
            expected_validity,
        )

    def test_extract_all_dates_error_cases(self):
        # Test case 1: Input is None
        input_text = None
        with pytest.raises(TypeError):
            extract_all_dates(input_text)

        # Test case 2: Input is a non-string type
        input_text = 123
        with pytest.raises(TypeError):
            extract_all_dates(input_text)


class Test_clean_input(TestCase):
    def test_clean_input_valid_cases(self):
        # Test case 1: Input with newline characters
        input_match = ["AMIT\nKUMAR", "RITU\nSINGH"]
        expected_output = ["AMIT", "KUMAR", "RITU", "SINGH"]
        assert clean_input(input_match) == expected_output

        # Test case 2: Input without newline characters
        input_match = ["AMIT", "RITU"]
        expected_output = ["AMIT", "RITU"]
        assert clean_input(input_match) == expected_output

    def test_clean_input_error_cases(self):
        # Test case 1: Input is None
        input_match = None
        with pytest.raises(TypeError):
            clean_input(input_match)

        # Test case 2: Input is a non-list type
        input_match = 123
        with pytest.raises(TypeError):
            clean_input(input_match)


class Test_extract_all_names(TestCase):
    def test_extract_all_names_valid_cases(self):
        # Test case 1: Input with valid names and stopwords
        input_text = "AMIT KUMAR INDIA TRANSPORT LICENCE RITU SIGNH"
        expected_output = []
        assert extract_all_names(input_text) == expected_output

        # Test case 2: Input without any stopwords
        input_text = "AMIT KUMAR\nRITU SINGH"
        expected_output = ["AMIT KUMAR", "RITU SINGH"]
        assert extract_all_names(input_text) == expected_output

    def test_extract_all_names_error_cases(self):
        # Test case 1: Input is None
        input_text = None
        with pytest.raises(TypeError):
            extract_all_names(input_text)

        # Test case 2: Input is a non-string type
        input_text = 123
        with pytest.raises(TypeError):
            extract_all_names(input_text)


class Test_extract_address_regex(TestCase):
    def test_extract_address_regex_valid_cases(self):
        # Test case 1: Input with address starting with "Address :"
        input_text = "Address :\nLine 1\nLine 2\n\nOther text"
        expected_output = "Line 1\nLine 2"
        assert extract_address_regex(input_text) == expected_output

        # Test case 2: Input with address starting with "ADDRESS -"
        input_text = "Some text ADDRESS - Line 1, Line 2, 123456 Other text"
        expected_output = "Line 1, Line 2,"
        assert extract_address_regex(input_text) == expected_output

    def test_extract_address_regex_error_cases(self):
        # Test case 1: Input is None
        input_text = None
        with pytest.raises(TypeError):
            extract_address_regex(input_text)

        # Test case 2: Input is a non-string type
        input_text = 123
        with pytest.raises(TypeError):
            extract_address_regex(input_text)


class Test_extract_address(TestCase):
    def test_extract_address_error_cases(self):
        # Test case 1: Invalid image path
        with pytest.raises(FileNotFoundError):
            extract_address("invalid_path.jpg")

        # Test case 2: Invalid image format
        with tempfile.NamedTemporaryFile(suffix=".txt") as temp_file:
            with pytest.raises(Exception):
                extract_address(temp_file.name)


class Test_extract_auth_allowed(TestCase):
    def test_extract_auth_allowed_valid_cases(self):
        # Test case 1: Input with multiple authorized types
        input_text = "This is a sample text with MCWG, LMV, and TRANS authorizations."
        expected_output = ["MCWG", "LMV", "TRANS"]
        assert extract_auth_allowed(input_text) == expected_output

        # Test case 2: Input with a single authorized type
        input_text = "The authorization type is M.CYL."
        expected_output = ["M.CYL."]
        assert extract_auth_allowed(input_text) == expected_output

    def test_extract_auth_allowed_error_cases(self):
        # Test case 1: Input is None
        input_text = None
        with pytest.raises(TypeError):
            extract_auth_allowed(input_text)

        # Test case 2: Input is a non-string type
        input_text = 123
        with pytest.raises(TypeError):
            extract_auth_allowed(input_text)


class Test_expired(TestCase):
    def test_expired_valid_cases(self):
        # Test case 1: Future date (not expired)
        future_date = (datetime.now() + timedelta(days=30)).strftime("%d/%m/%Y")
        assert expired(future_date) is False

        # Test case 2: Past date (expired)
        past_date = (datetime.now() - timedelta(days=30)).strftime("%d/%m/%Y")
        assert expired(past_date) is True

    def test_expired_error_cases(self):
        # Test case 1: Input is None
        input_date = None
        assert expired(input_date) is False

        # Test case 2: Input is an invalid date string
        invalid_date = "Invalid Date"
        assert expired(invalid_date) is False


class Test_extract_extract_driving_license_details(TestCase):
    def test_extract_driving_license_details_invalid_path(self):
        # Test case 1: Invalid image path
        invalid_path = "invalid_path.jpg"
        with pytest.raises(FileNotFoundError):
            extract_driving_license_details(invalid_path)

    def test_extract_driving_license_details_invalid_file(self):
        # Test case 2: Invalid file (non-existent or invalid format)
        invalid_path = "invalid_file.txt"
        invalid_format_path = os.path.join(
            os.path.dirname(__file__), "invalid_format.txt"
        )

        # Create an empty file for testing invalid format
        with open(invalid_format_path, "w"):
            pass

        try:
            with pytest.raises((FileNotFoundError, UnidentifiedImageError)):
                extract_driving_license_details(invalid_path)
            with pytest.raises(UnidentifiedImageError):
                extract_driving_license_details(invalid_format_path)
        finally:
            os.remove(invalid_format_path)
