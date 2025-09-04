import pytest
from openbharatocr.ocr.passbook import (
    extract_name,
    extract_open_date,
    extract_phone,
    extract_branch_name,
    extract_account_no,
    extract_cif_no,
    extract_address,
    extract_generic_bank_name,
    parse_passbook,
)


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("Customer Name PUSHKARA SHARMA", "PUSHKARA SHARMA"),
        ("Customer Name RISHABH SHARMA", "RISHABH SHARMA"),
        ("Customer Name AADITYA SHARMA", "AADITYA SHARMA"),
        ("No Customer Name", None),
        ("Customer Name ", None),
        ("", None),
        (None, None),
        ("Customer Name1234", None),
        ("customer name ANKIT KUMAR", "ANKIT KUMAR"),
        ("CUSTOMER NAME NEHA", "NEHA"),
    ],
)
def test_extract_name(input_text, expected_output):
    assert extract_name(input_text) == expected_output


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("Open Date 27/10/2020", "27/10/2020"),
        ("Open Date 01/02/2021", "01/02/2021"),
        ("Open Date 02/12/2019", "02/12/2019"),
        ("No Open Date", None),
        ("Open Date", None),
    ],
)
def test_extract_open_date(input_text, expected_output):
    assert extract_open_date(input_text) == expected_output


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("HDFC Bank", "HDFC Bank"),
        ("Union Bank of India", "Union Bank of India"),
        ("State Bank of India", "State Bank of India"),
        ("No bank info", None),
    ],
)
def test_extract_generic_bank_name(input_text, expected_output):
    assert extract_generic_bank_name(input_text) == expected_output


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("Phone No 1234567890", "1234567890"),
        ("Phone 1987654321", "1987654321"),
        ("Phone 1122334455", "1122334455"),
        ("No mobile number", None),
        ("Phone No ", None),
    ],
)
def test_extract_phone(input_text, expected_output):
    assert extract_phone(input_text) == expected_output


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("Branch Delhi", "Delhi"),
        ("Branch Uttar Pradesh", "Uttar Pradesh"),
        ("Branch Haryana", "Haryana"),
        ("No branch info", None),
    ],
)
def test_extract_branch_name(input_text, expected_output):
    assert extract_branch_name(input_text) == expected_output


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("Account Number: 123456789", "123456789"),
        ("Account No. 987654321012", "987654321012"),
        ("Account Number 112233445566", "112233445566"),
        ("No account number", None),
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
    ],
)
def test_extract_address(input_text, expected_output):
    assert extract_address(input_text) == expected_output


# Example for integration test (works if dummy images exist)
# def test_parse_passbook(monkeypatch):
#     def fake_image_to_string(*args, **kwargs):
#         return "Name: PUSHKARA SHARMA\nAccount Number: 123456789\nPhone 1234567890\nCIF No 123456\nBranch Delhi\nHDFC Bank\nOpen Date 27/10/2020\nAddress: 123 Main Street"
#
#     monkeypatch.setattr("pytesseract.image_to_string", fake_image_to_string)
#     result = parse_passbook("dummy_img.jpeg")
#     assert result == {
#         "cif_no": "123456",
#         "name": "PUSHKARA SHARMA",
#         "account_no": "123456789",
#         "address": "123 Main Street",
#         "phone": "1234567890",
#         "branch_name": "Delhi",
#         "bank_name": "HDFC Bank",
#         "date_of_issue": "27/10/2020",
#     }
