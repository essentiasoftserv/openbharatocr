# from openbharatocr.ocr.pan import (
#     clean_input,
#     extract_all_names,
#     extract_pan,
#     extract_dob,
#     extract_pan_details,
# )
# import pytest
# from unittest import mock, TestCase


# class Test_clean_input(TestCase):
#     def test_clean_input_with_newlines(self):
#         match = ["Amit\nKumar", "Sunita\nKumar\nAnil"]
#         expected_output = ["Amit", "Kumar", "Sunita", "Kumar", "Anil"]
#         assert clean_input(match) == expected_output

#     def test_clean_input_without_newlines(self):
#         match = ["AMIT KUMAR", "SUNITA KUMAR"]
#         expected_output = ["AMIT KUMAR", "SUNITA KUMAR"]
#         assert clean_input(match) == expected_output

#     def test_clean_input_with_empty_string(self):
#         match = [""]
#         expected_output = []
#         assert clean_input(match) == expected_output

#     def test_clean_input_with_none(self):
#         with pytest.raises(TypeError):
#             clean_input(None)


# class Test_extract_all_names(TestCase):
#     def test_extract_all_names(self):
#         input_string = """
#         This is a sample input string.
#         AMIT KUMAR
#         SUNITA KUMAR
#         INCOME TAX DEPARTMENT INDIA
#         ANIL JOSHI
#         GOVT OF INDIA
#         """
#         expected_output = ["AMIT KUMAR", "SUNITA KUMAR", "ANIL JOSHI"]
#         assert extract_all_names(input_string) == expected_output

#     def test_extract_all_names_individual(slef):
#         assert extract_all_names("") == []

#         input_string = "This is a string without any names."
#         assert extract_all_names(input_string) == []

#         input_string = "INDIA GOVT TAX DEPARTMENT"
#         assert extract_all_names(input_string) == []

#         input_string = "ABC\nXYZ"
#         assert extract_all_names(input_string) == []

#     def test_extract_all_names_with_empty_input(self):
#         input_text = ""
#         expected_output = []
#         assert extract_all_names(input_text) == expected_output

#     def test_extract_all_names_with_invalid_input(self):
#         with pytest.raises(TypeError):
#             extract_all_names(None)


# class Test_extract_pan(TestCase):
#     def test_extract_pan_valid_input(self):
#         input_text = "Amit Kumar's PAN number is ABCDE1234F"
#         expected_output = "ABCDE1234F"
#         assert extract_pan(input_text) == expected_output

#     def test_extract_pan_no_match(self):
#         input_text = "This is a random text without a PAN number"
#         expected_output = ""
#         assert extract_pan(input_text) == expected_output

#     def test_extract_pan_invalid_format(self):
#         input_text = "This is an invalid PAN number: ABCDE12345"
#         assert extract_pan(input_text) == ""

#     def test_extract_pan_invalid_input(self):
#         with pytest.raises(TypeError):
#             extract_pan(None)
#         with pytest.raises(TypeError):
#             extract_pan(123)


# class Test_extract_dob(TestCase):
#     def test_extract_dob_valid_formats(self):
#         inputs = [
#             "Amit Kumar's date of birth is 01/01/1990",
#             "15-12-1985 is sunita kumar's DOB",
#             "Rajiv's birth date is 30.06.78",
#         ]
#         expected_outputs = ["01/01/1990", "15-12-1985", "30.06.78"]
#         for input_str, expected_output in zip(inputs, expected_outputs):
#             assert extract_dob(input_str) == expected_output

#     def test_extract_dob_invalid_formats(self):
#         assert extract_dob("Amit Kumar's DOB is 1990/01/01") == ""
#         assert extract_dob("DOB: 01/Jan/1990") == ""
#         assert extract_dob("My DOB is 01:01:1990") == ""

#         assert extract_dob("DOB: 15 03 1985") == ""

#     def test_extract_dob_invalid_input(self):
#         with pytest.raises(TypeError):
#             extract_dob(None)


# class Test_extract_pan_details(TestCase):
#     def test_image_without_pan_details(self):
#         # Mock the image reading and text extraction
#         with mock.patch("PIL.Image.open") as mock_open, mock.patch(
#             "pytesseract.image_to_string"
#         ) as mock_image_to_string, mock.patch("imghdr.what") as mock_what:
#             mock_image = mock.MagicMock()
#             mock_open.return_value = mock_image
#             mock_image_to_string.return_value = "This is a test without PAN details."
#             mock_what.return_value = "jpeg"

#             result = extract_pan_details("dummy_img.jpg")

#             assert result["Full Name"] == ""
#             assert result["Parent's Name"] == ""
#             assert result["Date of Birth"] == ""
#             assert result["PAN Number"] == ""

#             mock_open.assert_called_once_with("dummy_img.jpg")
#             mock_image_to_string.assert_called_once_with(mock_image)
#             mock_what.assert_called_once_with("dummy_img.jpg")

#             with mock.patch(
#                 "openbharatocr.ocr.pan.extract_all_names"
#             ) as mock_extract_all_names, mock.patch(
#                 "openbharatocr.ocr.pan.extract_dob"
#             ) as mock_extract_dob, mock.patch(
#                 "openbharatocr.ocr.pan.extract_pan"
#             ) as mock_extract_pan:
#                 mock_extract_all_names.return_value = []
#                 mock_extract_dob.return_value = ""
#                 mock_extract_pan.return_value = ""

#                 result = extract_pan_details("dummy_img.jpg")

#                 assert result["Full Name"] == ""
#                 assert result["Parent's Name"] == ""
#                 assert result["Date of Birth"] == ""
#                 assert result["PAN Number"] == ""

#                 mock_extract_all_names.assert_called_once_with(
#                     "This is a test without PAN details."
#                 )
#                 mock_extract_dob.assert_called_once_with(
#                     "This is a test without PAN details."
#                 )
#                 mock_extract_pan.assert_called_once_with(
#                     "This is a test without PAN details."
#                 )


########## NEW TEST CODE ###########
from openbharatocr.ocr.pan import PANCardExtractor
import pytest
from unittest import mock, TestCase
import numpy as np
import json


class Test_clean_text(TestCase):
    def setUp(self):
        self.extractor = PANCardExtractor()

    def test_clean_text_with_spaces(self):
        text = "  AMIT   KUMAR  "
        expected_output = "AMIT KUMAR"
        assert self.extractor.clean_text(text) == expected_output

    def test_clean_text_with_special_chars(self):
        text = "AMIT@#$%KUMAR"
        expected_output = "AMITKUMAR"
        assert self.extractor.clean_text(text) == expected_output

    def test_clean_text_with_empty_string(self):
        text = ""
        expected_output = ""
        assert self.extractor.clean_text(text) == expected_output

    def test_clean_text_with_numbers_and_letters(self):
        text = "ABCDE1234F"
        expected_output = "ABCDE1234F"
        assert self.extractor.clean_text(text) == expected_output


class Test_clean_name(TestCase):
    def setUp(self):
        self.extractor = PANCardExtractor()

    def test_clean_name_normal(self):
        name = "AMIT KUMAR SHARMA"
        expected_output = "Amit Kumar Sharma"
        assert self.extractor.clean_name(name) == expected_output

    def test_clean_name_with_titles(self):
        name = "SHRI AMIT KUMAR"
        expected_output = "Amit Kumar"
        assert self.extractor.clean_name(name) == expected_output

    def test_clean_name_with_multiple_spaces(self):
        name = "  AMIT    KUMAR  "
        expected_output = "Amit Kumar"
        assert self.extractor.clean_name(name) == expected_output

    def test_clean_name_with_empty_string(self):
        name = ""
        expected_output = ""
        assert self.extractor.clean_name(name) == expected_output


class Test_is_valid_name(TestCase):
    def setUp(self):
        self.extractor = PANCardExtractor()

    def test_is_valid_name_valid_names(self):
        assert self.extractor.is_valid_name("AMIT KUMAR") == True
        assert self.extractor.is_valid_name("Rajesh Kumar Sharma") == True
        assert self.extractor.is_valid_name("Sunita Devi") == True

    def test_is_valid_name_invalid_names(self):
        assert self.extractor.is_valid_name("GOVT OF INDIA") == False
        assert self.extractor.is_valid_name("INCOME TAX") == False
        assert self.extractor.is_valid_name("123456") == False
        assert self.extractor.is_valid_name("A") == False
        assert self.extractor.is_valid_name("") == False
        assert self.extractor.is_valid_name("PERMANENT ACCOUNT") == False

    def test_is_valid_name_with_numbers(self):
        assert self.extractor.is_valid_name("AMIT123") == False


class Test_find_pan_number(TestCase):
    def setUp(self):
        self.extractor = PANCardExtractor()

    def test_find_pan_number_valid_pan(self):
        text_data = [{"text": "ABCDE1234F", "confidence": 0.9}]
        expected_output = "ABCDE1234F"
        assert self.extractor.find_pan_number(text_data) == expected_output

    def test_find_pan_number_no_match(self):
        text_data = [{"text": "AMIT KUMAR", "confidence": 0.9}]
        expected_output = None
        assert self.extractor.find_pan_number(text_data) == expected_output

    def test_find_pan_number_invalid_format(self):
        text_data = [{"text": "ABCD1234F", "confidence": 0.9}]
        assert self.extractor.find_pan_number(text_data) is None

    def test_find_pan_number_with_spaces(self):
        text_data = [{"text": "ABCD E123 4F", "confidence": 0.9}]
        expected_output = "ABCDE1234F"
        assert self.extractor.find_pan_number(text_data) == expected_output


class Test_find_dates(TestCase):
    def setUp(self):
        self.extractor = PANCardExtractor()

    def test_find_dates_valid_formats(self):
        text_data = [
            {"text": "Date of Birth: 15/08/1990", "confidence": 0.9},
            {"text": "DOB: 20-12-1985", "confidence": 0.8},
            {"text": "Born: 05.03.1995", "confidence": 0.85},
        ]
        result = self.extractor.find_dates(text_data)
        assert "15/08/1990" in result
        assert "20-12-1985" in result
        assert "05.03.1995" in result

    def test_find_dates_no_dates(self):
        text_data = [
            {"text": "AMIT KUMAR", "confidence": 0.9},
            {"text": "GOVT OF INDIA", "confidence": 0.8},
        ]
        result = self.extractor.find_dates(text_data)
        assert result == []


class Test_validate_pan(TestCase):
    def setUp(self):
        self.extractor = PANCardExtractor()

    def test_validate_pan_valid_pans(self):
        assert self.extractor.validate_pan("ABCDE1234F") == True
        assert self.extractor.validate_pan("ZYXWV9876A") == True

    def test_validate_pan_invalid_pans(self):
        assert self.extractor.validate_pan("ABCD1234F") == False
        assert self.extractor.validate_pan("ABCDE12345") == False
        assert self.extractor.validate_pan("abcde1234f") == False
        assert self.extractor.validate_pan("") == False
        assert self.extractor.validate_pan(None) == False


class Test_validate_date(TestCase):
    def setUp(self):
        self.extractor = PANCardExtractor()

    def test_validate_date_valid_dates(self):
        assert self.extractor.validate_date("15/08/1990") == True
        assert self.extractor.validate_date("01-01-1980") == True
        assert self.extractor.validate_date("31.12.1995") == True

    def test_validate_date_invalid_dates(self):
        assert self.extractor.validate_date("32/01/1990") == False
        assert self.extractor.validate_date("15/13/1990") == False
        assert self.extractor.validate_date("15/08/1800") == False
        assert self.extractor.validate_date("15/08/2030") == False
        assert self.extractor.validate_date("") == False
        assert self.extractor.validate_date("invalid") == False


class Test_extract_pan_details(TestCase):
    def setUp(self):
        self.extractor = PANCardExtractor()

    def test_extract_pan_details_success(self):
        # Mock the preprocessing and text extraction
        with mock.patch.object(
            self.extractor, "preprocess_image"
        ) as mock_preprocess, mock.patch.object(
            self.extractor, "extract_text_with_coordinates"
        ) as mock_extract_text:

            mock_preprocess.return_value = np.ones((100, 100, 3), dtype=np.uint8)

            mock_text_data = [
                {"text": "GOVT OF INDIA", "confidence": 0.95, "center_y": 20.0},
                {"text": "AMIT KUMAR SHARMA", "confidence": 0.89, "center_y": 110.0},
                {"text": "RAJESH KUMAR SHARMA", "confidence": 0.87, "center_y": 140.0},
                {"text": "ABCDE1234F", "confidence": 0.92, "center_y": 180.0},
                {"text": "15/08/1990", "confidence": 0.85, "center_y": 210.0},
            ]
            mock_extract_text.return_value = mock_text_data

            result = self.extractor.extract_pan_details("test_image.jpg")

            assert "pan_number" in result
            assert "name" in result
            assert "father_name" in result
            assert "date_of_birth" in result
            assert result["pan_number"] == "ABCDE1234F"
            assert result["date_of_birth"] == "15/08/1990"

    def test_extract_pan_details_preprocessing_error(self):
        with mock.patch.object(self.extractor, "preprocess_image") as mock_preprocess:
            mock_preprocess.side_effect = ValueError("Could not read image")

            result = self.extractor.extract_pan_details("invalid_image.jpg")

            assert "error" in result
            assert "Could not read image" in result["error"]

    def test_extract_pan_details_no_text(self):
        with mock.patch.object(
            self.extractor, "preprocess_image"
        ) as mock_preprocess, mock.patch.object(
            self.extractor, "extract_text_with_coordinates"
        ) as mock_extract_text:

            mock_preprocess.return_value = np.ones((100, 100, 3), dtype=np.uint8)
            mock_extract_text.return_value = []

            result = self.extractor.extract_pan_details("test_image.jpg")

            assert result["error"] == "No text could be extracted from the image"

    def test_extract_pan_details_without_pan_details(self):
        with mock.patch.object(
            self.extractor, "preprocess_image"
        ) as mock_preprocess, mock.patch.object(
            self.extractor, "extract_text_with_coordinates"
        ) as mock_extract_text:

            mock_preprocess.return_value = np.ones((100, 100, 3), dtype=np.uint8)

            mock_text_data = [
                {"text": "GOVT OF INDIA", "confidence": 0.95, "center_y": 20.0},
                {"text": "INCOME TAX DEPARTMENT", "confidence": 0.93, "center_y": 50.0},
            ]
            mock_extract_text.return_value = mock_text_data

            result = self.extractor.extract_pan_details("test_image.jpg")

            assert result["pan_number"] is None
            assert result["name"] == ""
            assert result["father_name"] == ""
            assert result["date_of_birth"] is None


class Test_preprocess_image(TestCase):
    def setUp(self):
        self.extractor = PANCardExtractor()

    @mock.patch("cv2.imread")
    def test_preprocess_image_success(self, mock_imread):
        mock_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        mock_imread.return_value = mock_image

        result = self.extractor.preprocess_image("test_image.jpg")

        assert result is not None
        assert isinstance(result, np.ndarray)
        mock_imread.assert_called_once_with("test_image.jpg")

    @mock.patch("cv2.imread")
    def test_preprocess_image_failure(self, mock_imread):
        mock_imread.return_value = None

        with pytest.raises(ValueError, match="Could not read image from"):
            self.extractor.preprocess_image("invalid_image.jpg")


class Test_save_results(TestCase):
    def setUp(self):
        self.extractor = PANCardExtractor()

    def test_save_results_success(self, tmp_path=None):
        # Create a simple test since we can't use pytest fixtures in unittest
        test_results = {
            "pan_number": "ABCDE1234F",
            "name": "Amit Kumar",
            "father_name": "Rajesh Kumar",
            "confidence_score": 90,
        }

        output_file = "test_results.json"

        # Test that save_results doesn't raise an exception
        try:
            self.extractor.save_results(test_results, output_file)
            # Clean up
            import os

            if os.path.exists(output_file):
                os.remove(output_file)
        except Exception as e:
            pytest.fail(f"save_results raised {e} unexpectedly!")


class Test_extract_names_with_keywords(TestCase):
    def setUp(self):
        self.extractor = PANCardExtractor()

    def test_extract_names_with_keywords_success(self):
        text_data = [
            {"text": "Name:", "confidence": 0.9, "center_y": 100},
            {"text": "AMIT KUMAR SHARMA", "confidence": 0.9, "center_y": 110},
            {"text": "Father's Name:", "confidence": 0.9, "center_y": 130},
            {"text": "RAJESH KUMAR", "confidence": 0.9, "center_y": 140},
        ]

        name, father_name = self.extractor.extract_names_with_keywords(text_data)

        assert name is not None
        assert father_name is not None

    def test_extract_names_with_keywords_no_keywords(self):
        text_data = [
            {"text": "AMIT KUMAR SHARMA", "confidence": 0.9, "center_y": 110},
            {"text": "RAJESH KUMAR", "confidence": 0.9, "center_y": 140},
        ]

        name, father_name = self.extractor.extract_names_with_keywords(text_data)

        assert name is None
        assert father_name is None


class Test_extract_names_positional(TestCase):
    def setUp(self):
        self.extractor = PANCardExtractor()

    def test_extract_names_positional_success(self):
        text_data = [
            {"text": "GOVT OF INDIA", "confidence": 0.9, "center_y": 50},
            {"text": "AMIT KUMAR SHARMA", "confidence": 0.9, "center_y": 100},
            {"text": "RAJESH KUMAR SHARMA", "confidence": 0.9, "center_y": 130},
            {"text": "ABCDE1234F", "confidence": 0.9, "center_y": 160},
        ]

        name, father_name = self.extractor.extract_names_positional(text_data)

        # Should extract some names based on position
        assert name is not None or father_name is not None

    def test_extract_names_positional_no_valid_names(self):
        text_data = [
            {"text": "GOVT OF INDIA", "confidence": 0.9, "center_y": 50},
            {"text": "INCOME TAX", "confidence": 0.9, "center_y": 80},
            {"text": "DEPARTMENT", "confidence": 0.9, "center_y": 110},
        ]

        name, father_name = self.extractor.extract_names_positional(text_data)

        assert name is None
        assert father_name is None


class Test_find_names_improved(TestCase):
    def setUp(self):
        self.extractor = PANCardExtractor()

    def test_find_names_improved_success(self):
        text_data = [
            {"text": "AMIT KUMAR", "confidence": 0.9, "center_y": 100},
            {"text": "RAJESH SHARMA", "confidence": 0.9, "center_y": 130},
        ]

        result = self.extractor.find_names_improved(text_data)

        assert "name" in result
        assert "father_name" in result

    def test_find_names_improved_identical_names(self):
        text_data = [
            {"text": "AMIT KUMAR", "confidence": 0.9, "center_y": 100},
            {"text": "AMIT KUMAR", "confidence": 0.9, "center_y": 130},
            {"text": "RAJESH SHARMA", "confidence": 0.85, "center_y": 160},
        ]

        result = self.extractor.find_names_improved(text_data)

        # Should handle identical names and find alternatives
        assert result["name"] != result["father_name"] or not result["father_name"]
