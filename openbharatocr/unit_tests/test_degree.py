import pytest
from unittest import mock, TestCase
from openbharatocr.ocr.degree import (
    extract_name,
    extract_institution_name,
    extract_degree_name,
    extract_year_of_passing,
    parse_degree_certificate,
)


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("has conferred upon Pushkara Sharma", "Pushkara Sharma"),
        (
            "the Academic Council has conferred upon Shivalingayya R Mathad",
            "Shivalingayya R Mathad",
        ),
        ("is conferred upon Usman Yusuf Bello", "Usman Yusuf Bello"),
        ("This is a certificate without a name.", None),
        ("Congratulations to someone special.", None),
        ("Name: [Placeholder]", None),
    ],
)
def test_extract_name(input_text, expected_output):
    assert extract_name(input_text) == expected_output


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("He has earned the Bachelor of Science degree.", "Bachelor of Science"),
        ("She was awarded a Master of Arts in History.", "Master of Arts"),
        ("Doctor of Philosophy (Ph.D.) in Mathematics.", "Doctor of Philosophy"),
        ("This person has a degree.", None),
        ("Studied a lot.", None),
        ("Completed studies in 2020.", None),
    ],
)
def test_extract_degree_name(input_text, expected_output):
    assert extract_degree_name(input_text) == expected_output


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("Amity University", "Amity University"),
        ("Amity University", "Amity University"),
        ("Sharda University.", "Sharda University"),
        ("Studied at a renowned institution.", None),
        ("Completed online courses.", None),
        ("School of hard knocks.", None),
    ],
)
def test_extract_institution_name(input_text, expected_output):
    assert extract_institution_name(input_text) == expected_output


@pytest.mark.parametrize(
    "input_text, expected_output",
    [
        ("Passed in 2020 with flying colors.", "2020"),
        ("Graduated in the year 2023.", "2023"),
        ("Year of passing: 2020.", "2020"),
        ("Completed studies recently.", None),
        ("Passed the exam.", None),
        ("Graduated a few years ago.", None),
    ],
)
def test_extract_year_of_passing(input_text, expected_output):
    assert extract_year_of_passing(input_text) == expected_output


# @pytest.mark.parametrize(
#     "image_path, expected_output",
#     [
#         (
#             "dummy_img1.jpeg",
#             {
#                 "Name": "Pushkara Sharma",
#                 "Degree Name": "Bachelor of Technology",
#                 "University Name": "AMITY UNIVERSITY",
#                 "Year of Passing": "2020",
#             },
#         ),
#         (
#             "dummy_img2.jpeg",
#             {
#                 "Name": "Shivalingayya R Mathad",
#                 "Degree Name": "Master of Technology",
#                 "University Name": "AMITY UNIVERSITY",
#                 "Year of Passing": "2022",
#             },
#         ),
#         (
#             "dummy_img3.jpeg",
#             {
#                 "Name": "Usman Yusuf Bello",
#                 "Degree Name": "Master of Science",
#                 "University Name": "SHARDA\nUNIVERSITY",
#                 "Year of Passing": "2020",
#             },
#         ),
#         (
#             "dummy_img4.jpeg",
#             {
#                 "Name": None,
#                 "Degree Name": None,
#                 "University Name": None,
#                 "Year of Passing": None,
#             },
#         ),
#         (
#             "dummy_img5.jpeg",
#             {
#                 "Name": None,
#                 "Degree Name": None,
#                 "University Name": None,
#                 "Year of Passing": None,
#             },
#         ),
#     ],
# )
# def test_parse_degree_certificate(image_path, expected_output):
#     assert parse_degree_certificate(image_path) == expected_output
