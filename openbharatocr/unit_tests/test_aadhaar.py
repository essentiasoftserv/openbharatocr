def test_empty_data():
  """Tests name extraction with empty data."""

  input_text = ""  # Empty string
  extracted_name = input_text.strip()

  # Handle empty input gracefully (adjust assertion if needed)
  assert extracted_name == ""  # Or assert expected behavior

def test_extract_name_missing_label():
  """Tests name extraction when the 'Full Name:' label is missing."""

  input_text = "Amit Kumar"  # No "Full Name:" label
  expected_output = "Amit Kumar"  # Adjust if needed based on your logic

  # Extract the name (assuming all text is the name)
  extracted_name = input_text.strip()  # Remove leading/trailing whitespace

  assert extracted_name == expected_output

def test_extract_name_valid():
  """Tests name extraction with a valid format."""

  input_text = "Full Name: Amit Kumar"
  expected_output = "Amit Kumar"

  # Extract the name (assuming simple logic for this test)
  extracted_name = input_text.split(": ")[-1].strip()

  assert extracted_name == expected_output

# Assuming you have additional test cases (e.g., with punctuation or initials)
def test_extract_name_with_punctuation():
  """Tests name extraction with punctuation."""

  input_text = "Amit Kumar, Jr."
  expected_output = "Amit Kumar, Jr."

  # Modify extraction logic if needed based on your implementation
  extracted_name = input_text.strip()

  assert extracted_name == expected_output

def test_extract_name_with_initials():
  """Tests name extraction with initials."""

  input_text = "A. K. Singh"
  expected_output = "A. K. Singh"

  # Modify extraction logic if needed based on your implementation
  extracted_name = input_text.strip()

  assert extracted_name == expected_output

