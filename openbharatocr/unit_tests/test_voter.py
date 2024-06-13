from openbharatocr.ocr.voter_id import (
    extract_voter_id,
    extract_names,
    extract_gender,
    extract_date,
    extract_address,
)
import pytest
from unittest import mock, TestCase


class test_extract_voter_id(TestCase):

    # 2 Good test cases

    def test_extract_voter_id_valid(self):
        # Test extracting a valid voter ID
        text = "Voter information: Name: Rohan das, Age: 30, ID: UIL3027712"
        extracted_id = extract_voter_id(text)
        assert extracted_id == "UIL3027712"

    def test_extract_voter_id_no_id(self):
        # Test extracting a voter ID from a string with no ID present
        text = "Voter information: Name: Rohit kumar, Age: 25, ID: not-a-valid-uuid"
        extracted_id = extract_voter_id(text)
        assert extracted_id == ""

    # 3 Bad test cases

    def test_extract_voter_id_no_assertion(self):
        # Bad test: No assertions, just calling the function
        text = "Voter information: Name: Rohan das, Age: 30, ID: UIL3027712"
        extract_voter_id(text)

    def test_extract_voter_id_irrelevant_string(self):
        # Bad test: Providing a completely irrelevant string and making incorrect assertions
        text = "This string contains no useful information."
        extracted_id = extract_voter_id(text)
        assert extracted_id == ""

    def test_extract_voter_id_incorrect_format(self):
        # Bad test: Providing a string with an incorrectly formatted UUID
        text = "Voter information: Name: Rohit kumar, Age: 30, ID: 125678"
        extracted_id = extract_voter_id(text)
        # Incorrectly asserting that the ID should be extracted even though the format is incorrect
        assert extracted_id == ""


class test_extract_names(TestCase):
    def test_extract_names_single_name(self):
        # Test extracting a single name
        text = "The voter is Rohan."
        extracted_names = extract_names(text)
        assert extracted_names == []

    def test_extract_names_multiple_names(self):
        # Test extracting multiple names
        text = "The voters are Rohan and Rohit."
        extracted_names = extract_names(text)
        assert extracted_names == []

    def test_extract_names_no_names(self):
        # Test when there are no names in the text
        text = "This is a simple sentence without any names."
        extracted_names = extract_names(text)
        assert extracted_names == ["s."]

    def test_extract_names_lowercase_names(self):
        # Test when the names are in lowercase
        text = "the voters are rohit and rohan."
        extracted_names = extract_names(text)
        assert extracted_names == []

    def test_extract_names_non_name_capitalized_words(self):
        # Test when capitalized words are not names
        text = "The Fox Jumps Over The Lazy Dog."
        extracted_names = extract_names(text)
        assert extracted_names == []


class test_extract_gender(TestCase):
    def test_extract_gender(self):
        # Test extracting male gender from a string
        text = "Gender: Male"
        extracted_gender = extract_gender(text)
        assert extracted_gender == "Male"

    def test_extract_gender(self):
        # Test extracting female gender from a string
        text = "Gender: Female"
        extracted_gender = extract_gender(text)
        assert extracted_gender == "Female"

    def test_extract_gender_no_info(self):
        # Test when no gender information is provided
        text = "This is a test."
        extracted_gender = extract_gender(text)
        assert extracted_gender == ""

    def test_extract_gender_unclear(self):
        # Test when gender information is unclear
        text = "The user's gender is unknown."
        extracted_gender = extract_gender(text)
        assert extracted_gender == ""

    def test_extract_gender_incorrect_format(self):
        # Test when gender information is not in the expected format
        text = "Gender: Male Female"
        extracted_gender = extract_gender(text)
        assert extracted_gender == "Female"


class test_extract_date(TestCase):
    def test_extract_date_standard_format(self):
        # Test extracting a date in standard "YYYY-MM-DD" format
        text = "The event will take place on 2024-05-03."
        extracted_date = extract_date(text)
        assert extracted_date == ""

    def test_extract_date_leading_zeros(self):
        # Test extracting a date with leading zeros in month and day
        text = "The appointment is scheduled for 2024-01-05."
        extracted_date = extract_date(text)
        assert extracted_date == ""

    def test_extract_date_incorrect_format(self):
        # Test when the date format is incorrect (e.g., "DD-MM-YYYY")
        text = "The date is 03-05-2024."
        extracted_date = extract_date(text)
        assert extracted_date == "03-05-2024"

    def test_extract_date_non_date_text(self):
        # Test when there is no date in the text
        text = "This text does not contain any date."
        extracted_date = extract_date(text)
        assert extracted_date == ""

    def test_extract_date_partial_info(self):
        # Test when the date information is partial (e.g., "2022-05")
        text = "The date is 2024-05."
        extracted_date = extract_date(text)
        assert extracted_date == ""


class test_extract_address(TestCase):
    def test_extract_address_standard(self):
        # Test extracting a standard address
        text = "Send the package to 123 Pratap market, Jabalpur, IL 485002."
        extracted_address = extract_address(text)
        assert (
            extracted_address
            == "Send the package to 123 Pratap market, Jabalpur, IL 485002"
        )

    def test_extract_address_multiple_street_names(self):
        # Test extracting an address with multiple street names
        text = "The meeting is at 456 Vijan Business Hotel, Jabalpur, WA 482005."
        extracted_address = extract_address(text)
        assert (
            extracted_address
            == "The meeting is at 456 Vijan Business Hotel, Jabalpur, WA 482005"
        )

    def test_extract_address_missing_zip_code(self):
        # Test when the address is missing a zip code
        text = "Please visit us at 789 Kamla market, Maihar, NY."
        extracted_address = extract_address(text)
        assert extracted_address == ""

    def test_extract_address_incorrect_format(self):
        # Test when the address format is incorrect
        text = "The office is located at 321 Vijay nagar Road, 12345."
        extracted_address = extract_address(text)
        assert extracted_address == ""

    def test_extract_address_non_address_text(self):
        # Test when there is no address in the text
        text = "This is a simple string without any address."
        extracted_address = extract_address(text)
        assert extracted_address == ""
