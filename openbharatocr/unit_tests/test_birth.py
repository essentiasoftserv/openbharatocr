import pytest
from openbharatocr.ocr.birth_certificate import (
    extract_address_of_birth_place,
    extract_dob,
    extract_father_name,
    extract_mother_name,
    extract_name,
    extract_registration_date,
    extract_registration_number,
    parse_birth_certificate,
)


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("Name Rishabh Sharma Sex Male ", "Rishabh Sharma"),
        ("Name Pushkara Sharma Sex Male Date ", "Pushkara Sharma"),
        ("Name Aditya Sharma Sex Male 03/03/1985", "Aditya Sharma"),
        ("Sex Male Date Birth 04/04/1975", None),
        ("Name ", None),
        ("", None),
    ],
)
def test_extract_name(input_text, expected_output):
    assert extract_name(input_text) == expected_output


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("Place of Birth: DELHI CITY HOSPITAL Name Mcther Arti", "DELHI CITY HOSPITAL"),
        ("Place of Birth: QRT HOSPITAL Name Mcther Pooja", "QRT HOSPITAL"),
        ("Place of Birth: LMN GOVT. HOSPITAL Name Mcther Bharti", "LMN GOVT. HOSPITAL"),
        ("Name of Mother", None),
        ("Place of Birth: ", None),
        ("", None),
    ],
)
def test_extract_address_of_birth_place(input_text, expected_output):
    assert extract_address_of_birth_place(input_text) == expected_output


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("Date Birth 01/01/2000", "01/01/2000"),
        ("Date Birth 02/02/1995", "02/02/1995"),
        ("Date Birth 03/03/1985", "03/03/1985"),
        ("Birth Date 04/04/1975", None),
        ("Date Birth ", None),
        ("", None),
    ],
)
def test_extract_dob(input_text, expected_output):
    assert extract_dob(input_text) == expected_output


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("Name Mother Arti", "Arti"),
        ("Name Mother Pooja", "Pooja"),
        ("Name Mother Bharti", "Bharti"),
        ("Mother's Name Anjli", None),
        ("Name Mother ", None),
        ("", None),
    ],
)
def test_extract_mother_name(input_text, expected_output):
    assert extract_mother_name(input_text) == expected_output


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("Name the Father Mukesh ", "Mukesh"),
        ("Name the Father Sudheer", "Sudheer"),
        ("Name the Father Naresh", "Naresh"),
        ("Father's Name Ajeet", None),
        ("Name the Father ", None),
        ("", None),
    ],
)
def test_extract_father_name(input_text, expected_output):
    assert extract_father_name(input_text) == expected_output


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("Registration No 123456", "123456"),
        ("Registration No 789012", "789012"),
        ("Registration No 345678", "345678"),
        ("No Registration No", None),
        ("Registration No ", None),
        ("", None),
    ],
)
def test_extract_registration_number(input_text, expected_output):
    assert extract_registration_number(input_text) == expected_output


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("Date of Registration 08/11/2004", "08/11/2004"),
        ("Date of Registration 11/03/1998", "11/03/1998"),
        ("Date of Registration 12/06/1991", "12/06/1991"),
        ("Registration Date 04/04/1985", None),
        ("Date of Registration ", None),
        ("", None),
    ],
)
def test_extract_registration_date(input_text, expected_output):
    assert extract_registration_date(input_text) == expected_output


@pytest.mark.parametrize(
    "image_path, expected_output",
    [
        (
            "dummy_img1.jpeg",  # Change the expected output as per your dummy image(Provide a real image path otherwise test case will fail)
            {
                "Name": "Rishabh Sharma",
                "Address of Birth Place": "DELHI CITY HOSPITAL",
                "Date of Birth": "08/11/2004",
                "Mother's Name": "Arti",
                "Father's Name": "Mukesh",
                "Registration Number": "5933",
                "Registration Date": "08/11/2004",
            },
        ),
        (
            "dummy_img2.jpeg",  # Change the expected output as per your dummy image(Provide a real image path otherwise test case will fail)
            {
                "Name": " Pushkara Sharma",
                "Address of Birth Place": "QRT HOSPITAL",
                "Date of Birth": "11/03/1998",
                "Mother's Name": "Pooja",
                "Father's Name": "Sudheer",
                "Registration Number": "789012",
                "Registration Date": "11/03/1998",
            },
        ),
        (
            "dummy_img3.jpeg",  # Change the expected output as per your dummy image(Provide a real image path otherwise test case will fail)
            {
                "Name": "Aditya ",
                "Address of Birth Place": "LMN GOVT. HOSPITAL",
                "Date of Birth": "03/03/1985",
                "Mother's Name": "Bharti",
                "Father's Name": "Naresh",
                "Registration Number": "345678",
                "Registration Date": "12/06/1991",
            },
        ),
        (
            "dummy_img4.jpeg",  # Change the expected output as per your dummy image(Provide a real image path otherwise test case will fail)
            {
                "Name": None,
                "Address of Birth Place": "ST. EFG Hospital",
                "Date of Birth": None,
                "Mother's Name": None,
                "Father's Name": "def",
                "Registration Number": None,
                "Registration Date": "12/09/2008",
            },
        ),
        (
            "dummy_img5.jpeg",  # Change the expected output as per your dummy image(Provide a real image path otherwise test case will fail)
            {
                "Name": None,
                "Address of Birth Place": None,
                "Date of Birth": None,
                "Mother's Name": None,
                "Father's Name": None,
                "Registration Number": None,
                "Registration Date": None,
            },
        ),
        (
            "dummy_img6.jpeg",  # Change the expected output as per your dummy image(Provide a real image path otherwise test case will fail)
            {
                "Name": "xyz",
                "Address of Birth Place": None,
                "Date of Birth": None,
                "Mother's Name": "pqr",
                "Father's Name": None,
                "Registration Number": "85648",
                "Registration Date": None,
            },
        ),
    ],
)
def test_parse_birth_certificate(image_path, expected_output):
    assert parse_birth_certificate(image_path) == expected_output
