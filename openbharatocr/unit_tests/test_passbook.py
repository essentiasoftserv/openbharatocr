import pytest
from unittest import mock, TestCase
from openbharatocr.ocr.passbook import (
    extract_name,
    extract_open_date,
    extract_phone,
    extract_branch_name,
    extract_account_no,
    extract_cif_no,
    extract_address,
)


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("Customer Name PUSHKARA SHARMA", "PUSHKARA SHARMA"),
        ("Customer Name RISHABH SHARMA", "RISHABH SHARMA"),
        ("Customer Name AADITYA SHARMA", "AADITYA SHARMA"),
        ("No Customer Name", None),
        ("Customer Name ", None),
    ],
)
def test_extract_name(input_text, expected_output):
    assert extract_name(input_text) == expected_output


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("Open Date 27 Oct 2020", "27 Oct 2020"),
        ("Open Date 1 Feb 2021", "1 Feb 2021"),
        ("Open Date 2 Dec 2019", "2 Dec 2019"),
        ("No Open Date", None),
        ("Open Date", None),
    ],
)
def test_extract_open_date(input_text, expected_output):
    assert extract_open_date(input_text) == expected_output


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("HDFC BANK", "HDFC BANK"),
        ("DEF UNION BANK", "DEF UNION BANK"),
        ("GHI BANK", "GHI BANK"),
        ("No bank info", None),
        ("BANK", None),
    ],
)
def test_extract_bank_name(input_text, expected_output):
    assert extract_bank_name(input_text) == expected_output


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("Mobile No 1234567890", "1234567890"),
        ("Mobile No 1987654321", "1987654321"),
        ("Mobile No 1122334455", "1122334455"),
        ("No mobile number", None),
        ("Mobile No ", None),
    ],
)
def test_extract_phone(input_text, expected_output):
    assert extract_phone(input_text) == expected_output


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("Branch Name Delhi", "Delhi"),
        ("Branch Name Uttar Pradesh", "Uttar Pradesh"),
        ("Branch Name Haryana", "Haryana"),
        ("No branch info", None),
        ("Banh me", None),
    ],
)
def test_extract_branch_name(input_text, expected_output):
    assert extract_branch_name(input_text) == expected_output


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("Nomination Rishabh Sharma", "Rishabh Sharma"),
        ("Nomination Aaditya Sharma", "Aaditya Sharma"),
        ("Nomination Pushkara Sharma", "Pushkara Sharma"),
        ("No nomination info", None),
        ("Nomination ", None),
    ],
)
def test_extract_nomination_name(input_text, expected_output):
    assert extract_nomination_name(input_text) == expected_output


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("Email pushkara12@example.com", "pushkara12@example.com"),
        ("Email: rishabh.sharma@example.org", "rishabh.sharma@example.org"),
        ("aaditya@example.net", "aaditya@example.net"),
        ("No email", None),
        ("Email ", None),
    ],
)
def test_extract_email(input_text, expected_output):
    assert extract_email(input_text) == expected_output


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("Account Number: 123456789", "123456789"),
        ("Account Number: 987654321012", "987654321012"),
        ("Account Number: 112233445566", "112233445566"),
        ("No account number", None),
        ("Account Number: ", None),
    ],
)
def test_extract_account_no(input_text, expected_output):
    assert extract_account_no(input_text) == expected_output


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("CIF No 123456", "123456"),
        ("CIF 987654", "987654"),
        ("CIF No. 112233", "112233"),
        ("No CIF info", None),
        ("CIF No ", None),
    ],
)
def test_extract_cif_no(input_text, expected_output):
    assert extract_cif_no(input_text) == expected_output


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("Address: 123 Main Street", "123 Main Street"),
        ("456 xyz St, Delhi", "456 xyz St, Delhi"),
        ("789 Main Market, Haryana", "789 Main Market, Haryana"),
        ("No address info", None),
        ("123", None),
    ],
)
def test_extract_address(input_text, expected_output):
    assert extract_address(input_text) == expected_output


# @pytest.mark.parametrize(
#     "image_path, expected_output",
#     [
#         (
#             "dummy_img1.jpeg",
#             {
#                 "cif_no": None,
#                 "name": "PUSHKARA SHARMA",
#                 "account_no": "501003871659",
#                 "address": "418 A Milak DLF Pusta Road",
#                 "phone": "919354454113",
#                 "email": None,
#                 "nomination_name": "RISHABH SHARMA",
#                 "branch_name": "DELHI",
#                 "bank_name": "HDFC BANK",
#                 "date_of_issue": "27 Oct 2020",
#             },
#         ),
#         (
#             "dummy_img2.jpeg",
#             {
#                 "cif_no": "654321",
#                 "name": "RISHABH SHARMA",
#                 "account_no": "987654321012",
#                 "address": "456 xyz St, Delhi",
#                 "phone": "0987654321",
#                 "email": "rishabh.sharma@example.org",
#                 "nomination_name": "Aaditya Sharma",
#                 "branch_name": "Uttar Pradesh",
#                 "bank_name": "DEF UNION Bank",
#                 "date_of_issue": "1 Feb 2021",
#             },
#         ),
#         (
#             "dummy_img3.jpeg",
#             {
#                 "cif_no": "112233",
#                 "name": "AADITYA SHARMA",
#                 "account_no": "112233445566",
#                 "address": "789 Main MArket 4, Uttar Pradesh",
#                 "phone": "1122334455",
#                 "email": "aaditya@example.net",
#                 "nomination_name": "Pushkara Sharma",
#                 "branch_name": "Haryana",
#                 "bank_name": "GHI BANK",
#                 "date_of_issue": "2 Dec 2019",
#             },
#         ),
#         (
#             "dummy_img4.jpeg",
#             {
#                 "cif_no": None,
#                 "name": None,
#                 "account_no": None,
#                 "address": None,
#                 "phone": None,
#                 "email": None,
#                 "nomination_name": None,
#                 "branch_name": None,
#                 "bank_name": None,
#                 "date_of_issue": None,
#             },
#         ),
#         (
#             "dummy_img5.jpeg",
#             {
#                 "cif_no": None,
#                 "name": None,
#                 "account_no": None,
#                 "address": None,
#                 "phone": None,
#                 "email": None,
#                 "nomination_name": None,
#                 "branch_name": None,
#                 "bank_name": None,
#                 "date_of_issue": None,
#             },
#         ),
#     ],
# )
# def test_parse_passbook_frontpage(image_path, expected_output):
#     assert parse_passbook_frontpage(image_path) == expected_output
