from openbharatocr.ocr.pan import (
    clean_input,
    extract_all_names,
    extract_pan,
    extract_dob,
    extract_pan_details,
)
import pytest
from unittest import mock, TestCase


class Test_clean_input(TestCase):
    def test_clean_input_with_newlines(self):
        match = ["Amit\nKumar", "Sunita\nKumar\nAnil"]
        expected_output = ["Amit", "Kumar", "Sunita", "Kumar", "Anil"]
        assert clean_input(match) == expected_output

    def test_clean_input_without_newlines(self):
        match = ["AMIT KUMAR", "SUNITA KUMAR"]
        expected_output = ["AMIT KUMAR", "SUNITA KUMAR"]
        assert clean_input(match) == expected_output

    def test_clean_input_with_empty_string(self):
        match = [""]
        expected_output = [""]
        assert clean_input(match) == expected_output

    def test_clean_input_with_none(self):
        with pytest.raises(TypeError):
            clean_input(None)


class Test_extract_all_names(TestCase):
    def test_extract_all_names(self):
        input_string = """
        This is a sample input string.
        AMIT KUMAR
        SUNITA KUMAR
        INCOME TAX DEPARTMENT INDIA
        ANIL JOSHI
        GOVT OF INDIA
        """
        expected_output = ["AMIT KUMAR", "SUNITA KUMAR", "ANIL JOSHI"]
        assert extract_all_names(input_string) == expected_output

    def test_extract_all_names_individual(slef):
        assert extract_all_names("") == []

        input_string = "This is a string without any names."
        assert extract_all_names(input_string) == []

        input_string = "INDIA GOVT TAX DEPARTMENT"
        assert extract_all_names(input_string) == []

        input_string = "ABC\nXYZ"
        assert extract_all_names(input_string) == []

    def test_extract_all_names_with_empty_input(self):
        input_text = ""
        expected_output = []
        assert extract_all_names(input_text) == expected_output

    def test_extract_all_names_with_invalid_input(self):
        with pytest.raises(TypeError):
            extract_all_names(None)


class Test_extract_pan(TestCase):
    def test_extract_pan_valid_input(self):
        input_text = "Amit Kumar's PAN number is ABCDE1234F"
        expected_output = "ABCDE1234F"
        assert extract_pan(input_text) == expected_output

    def test_extract_pan_no_match(self):
        input_text = "This is a random text without a PAN number"
        expected_output = ""
        assert extract_pan(input_text) == expected_output

    def test_extract_pan_invalid_format(self):
        input_text = "This is an invalid PAN number: ABCDE12345"
        assert extract_pan(input_text) == ""

    def test_extract_pan_invalid_input(self):
        with pytest.raises(TypeError):
            extract_pan(None)
        with pytest.raises(TypeError):
            extract_pan(123)


class Test_extract_dob(TestCase):
    def test_extract_dob_valid_formats(self):
        inputs = [
            "Amit Kumar's date of birth is 01/01/1990",
            "15-12-1985 is sunita kumar's DOB",
            "Rajiv's birth date is 30.06.78",
        ]
        expected_outputs = ["01/01/1990", "15-12-1985", "30.06.78"]
        for input_str, expected_output in zip(inputs, expected_outputs):
            assert extract_dob(input_str) == expected_output

    def test_extract_dob_invalid_formats(self):
        assert extract_dob("Amit Kumar's DOB is 1990/01/01") == ""
        assert extract_dob("DOB: 01/Jan/1990") == ""
        assert extract_dob("My DOB is 01:01:1990") == ""

        assert extract_dob("DOB: 15 03 1985") == ""

    def test_extract_dob_invalid_input(self):
        with pytest.raises(TypeError):
            extract_dob(None)


class Test_extract_pan_details(TestCase):
    def test_image_without_pan_details(self):
        # Mock the image reading and text extraction
        with mock.patch("PIL.Image.open") as mock_open, mock.patch(
            "pytesseract.image_to_string"
        ) as mock_image_to_string, mock.patch("imghdr.what") as mock_what:
            mock_image = mock.MagicMock()
            mock_open.return_value = mock_image
            mock_image_to_string.return_value = "This is a test without PAN details."
            mock_what.return_value = "jpeg"

            result = extract_pan_details("dummy_img.jpg")

            assert result["Full Name"] == ""
            assert result["Parent's Name"] == ""
            assert result["Date of Birth"] == ""
            assert result["PAN Number"] == ""

            mock_open.assert_called_once_with("dummy_img.jpg")
            mock_image_to_string.assert_called_once_with(mock_image)
            mock_what.assert_called_once_with("dummy_img.jpg")

            with mock.patch(
                "openbharatocr.ocr.pan.extract_all_names"
            ) as mock_extract_all_names, mock.patch(
                "openbharatocr.ocr.pan.extract_dob"
            ) as mock_extract_dob, mock.patch(
                "openbharatocr.ocr.pan.extract_pan"
            ) as mock_extract_pan:
                mock_extract_all_names.return_value = []
                mock_extract_dob.return_value = ""
                mock_extract_pan.return_value = ""

                result = extract_pan_details("dummy_img.jpg")

                assert result["Full Name"] == ""
                assert result["Parent's Name"] == ""
                assert result["Date of Birth"] == ""
                assert result["PAN Number"] == ""

                mock_extract_all_names.assert_called_once_with(
                    "This is a test without PAN details."
                )
                mock_extract_dob.assert_called_once_with(
                    "This is a test without PAN details."
                )
                mock_extract_pan.assert_called_once_with(
                    "This is a test without PAN details."
                )
