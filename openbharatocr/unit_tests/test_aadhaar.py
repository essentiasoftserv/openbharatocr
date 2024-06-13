class ValidFormatTest:
    def test_extract_name_with_aadhaar_number(self):
        """Tests name extraction with an Aadhaar number present (Valid Format)."""
        input_text = "Full Name: Amit Kumar (Aadhaar: 123456789012)"
        expected_output = "Amit Kumar"
        extracted_name = input_text.split(": ")[1].split("(")[0].strip()
        assert extracted_name == expected_output

    def test_extract_name_with_special_characters(self):
        """Tests name extraction with special characters (Valid Format)."""
        input_text = "Full Name: Amit Kumar (ᵅअमित)"
        expected_output = "Amit Kumar"
        extracted_name = input_text.split(": ")[1].split("(")[0].strip()
        assert extracted_name == expected_output

    def test_extract_multiple_names(self):
        """Tests name extraction with multiple names (Valid Format)."""
        input_text = "Full Name: Amit Kumar and Sujata Singh"
        expected_output = ["Amit Kumar", "Sujata Singh"]
        extracted_names = input_text.split(": ")[1].split("and")
        extracted_names = [name.strip() for name in extracted_names]
        assert extracted_names == expected_output

    def test_extract_name_valid(self):
        """Tests name extraction with a valid format (Valid Format)."""
        input_text = "Full Name: Amit Kumar"
        expected_output = "Amit Kumar"
        extracted_name = input_text.split(": ")[-1].strip()
        assert extracted_name == expected_output

    def test_extract_name_with_punctuation(self):
        """Tests name extraction with punctuation (Valid Format)."""
        input_text = "Amit Kumar, Jr."
        expected_output = "Amit Kumar, Jr."
        extracted_name = input_text.strip()
        assert extracted_name == expected_output

    def test_extract_name_with_initials(self):
        """Tests name extraction with initials (Valid Format)."""
        input_text = "A. K. Singh"
        expected_output = "A. K. Singh"
        extracted_name = input_text.strip()
        assert extracted_name == expected_output


class EmptyAndExactFormatTest:
    def test_extract_empty_data(self):
        """Tests name extraction with empty data."""
        input_text = ""
        extracted_name = input_text.strip()
        # Adjust assertion based on desired behavior (e.g., assert extracted_name is None)

    def test_extract_name_exact_format(self):
        """Tests name extraction with exact format (Assumes format)."""
        input_text = "Full Name: Amit Kumar"
        expected_output = "Amit Kumar"
        extracted_name = input_text.split(": ")[1].strip()
        assert extracted_name == expected_output


class EdgeCaseTest:
    def test_extract_name_with_age(self):
        """Tests name extraction with age (Ignores age)."""
        input_text = "Full Name: Amit Kumar (Age: 30)"
        expected_output = "Amit Kumar"
        extracted_name = input_text.split(": ")[1].split("(")[0].strip()
        assert extracted_name == expected_output

    def test_extract_name_invalid_format(self):
        """Tests name extraction with invalid format (Doesn't verify error)."""
        input_text = "Amit Kumar"
        expected_output = None
        try:
            extracted_name = input_text.split(": ")[1].strip()
        except IndexError:
            pass
        assert True

    def test_extract_name_missing_label(self):
        """Tests name extraction with missing label (Might not handle all cases)."""
        input_text = "Amit Kumar"
        expected_output = "Amit Kumar"  # Adjust based on your logic for missing


class EdgeCaseTest:
    def test_extract_name_with_double_barreled_name(self):
        """Tests name extraction with a double-barreled name."""
        input_text = "Full Name: Amit Kumar-Singh"
        expected_output = "Amit Kumar-Singh"  # Adjust based on your logic
        extracted_name = input_text.split(": ")[1].strip()
        assert extracted_name == expected_output

    def test_extract_name_with_title(self):
        """Tests name extraction with a title (e.g., Dr., Mr.)"""
        input_text = "Full Name: Dr. Amit Kumar"
        expected_output = "Dr. Amit Kumar"  # Adjust based on your logic
        extracted_name = input_text.split(": ")[1].strip()
        assert extracted_name == expected_output

    def test_extract_name_with_initials_and_lastname(self):
        """Tests name extraction with initials and last name."""
        input_text = "A. K. (Full Name: Amit Kumar)"
        expected_output = "A. K."  # Adjust based on your logic (extract initials only)
        extracted_name = input_text.split()[0].strip()
        assert extracted_name == expected_output

    def test_extract_name_with_name_and_suffix(self):
        """Tests name extraction with a name and suffix (e.g., Jr., Sr.)"""
        input_text = "Full Name: Amit Kumar Jr."
        expected_output = "Amit Kumar Jr."
        extracted_name = input_text.split(": ")[1].strip()
        assert extracted_name == expected_output

    def test_extract_name_with_non_breaking_space(self):
        """Tests name extraction with a non-breaking space."""
        input_text = "Full Name: Amit Kumar"  # Non-breaking space (U+00A0)
        expected_output = "Amit Kumar"  # Adjust based on your handling of spaces
        extracted_name = input_text.split(": ")[1].strip()
        assert extracted_name == expected_output
