from ..pan import (
    extract_pan,
    extract_dob,
    extract_all_names,
    clean_input,
    extract_pan_details,
)
from PIL import Image
from io import BytesIO
import pytest


# Good Test 1: Test extract_pan function with valid input
def test_extract_pan_valid():
    input_text = "ABCDE1234F"
    assert extract_pan(input_text) == "ABCDE1234F"


# Good Test 2: Test extract_dob function with valid input
def test_extract_dob_valid():
    input_text = "Date of Birth: 01/01/1990"
    assert extract_dob(input_text) == "01/01/1990"


# Bad Test 1: Test extract_all_names with invalid input type
def test_extract_all_names_invalid_input():
    input_text = 123  # Invalid input type
    with pytest.raises(TypeError):
        extract_all_names(input_text)


# Bad Test 2: Test clean_input with None input
def test_clean_input_none():
    with pytest.raises(TypeError):
        clean_input(None)


# Bad Test 3: Test extract_pan_details with invalid image path
def test_extract_pan_details_invalid_path(tmpdir):
    invalid_path = str(tmpdir.join("invalid.jpg"))
    with pytest.raises(FileNotFoundError):
        extract_pan_details(invalid_path)
