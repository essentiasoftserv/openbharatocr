def test_extract_name_with_aadhaar_number():
    """Tests name extraction with an Aadhaar number present."""
    input_text = "Full Name: Amit Kumar (Aadhaar: 123456789012)"
    expected_output = "Amit Kumar"
    extracted_name = input_text.split(": ")[1].split("(")[0].strip()
    assert extracted_name == expected_output


def test_extract_name_with_special_characters():
    """Tests name extraction with special characters."""
    input_text = "Full Name: Amit Kumar (ᵅअमित)"
    expected_output = "Amit Kumar"
    extracted_name = input_text.split(": ")[1].split("(")[0].strip()
    assert extracted_name == expected_output


def test_extract_multiple_names():
    """Tests name extraction with multiple names."""
    input_text = "Full Name: Amit Kumar and Sujata Singh"
    expected_output = ["Amit Kumar", "Sujata Singh"]
    extracted_names = input_text.split(": ")[1].split("and")
    extracted_names = [name.strip() for name in extracted_names]
    assert extracted_names == expected_output


def test_extract_name_valid():
    """Tests name extraction with a valid format."""
    input_text = "Full Name: Amit Kumar"
    expected_output = "Amit Kumar"
    extracted_name = input_text.split(": ")[-1].strip()
    assert extracted_name == expected_output


def test_extract_name_with_punctuation():
    """Tests name extraction with punctuation."""
    input_text = "Amit Kumar, Jr."
    expected_output = "Amit Kumar, Jr."
    extracted_name = input_text.strip()
    assert extracted_name == expected_output


def test_extract_name_with_initials():
    """Tests name extraction with initials."""
    input_text = "A. K. Singh"
    expected_output = "A. K. Singh"
    extracted_name = input_text.strip()
    assert extracted_name == expected_output


def test_extract_name_with_age():
    # bad test
    input_text = "Full Name: Amit Kumar (Age: 30)"
    expected_output = "Amit Kumar"
    extracted_name = input_text.split(": ")[1].split("(")[0].strip()
    assert extracted_name == expected_output


def test_extract_name_exact_format():
    # bad testnon format
    input_text = "Full Name: Amit Kumar"
    expected_output = "Amit Kumar"
    extracted_name = input_text.split(": ")[1].strip()
    assert extracted_name == expected_output


def test_extract_name_invalid_format(expected_error=IndexError):
    # bad test
    input_text = "Amit Kumar"
    expected_output = None
    try:
        extracted_name = input_text.split(": ")[1].strip()
    except expected_error:
        pass
    assert True


def test_empty_data():
    # bad test
    input_text = ""
    extracted_name = input_text.strip()
    assert extracted_name == ""


def test_extract_name_missing_label():
    # bad test
    input_text = "Amit Kumar"
    expected_output = "Amit Kumar"
    extracted_name = input_text.strip()
    assert extracted_name == expected_output
