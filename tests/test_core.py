import pandas as pd
from transformers import parse_number, clean_account_key


def test_parse_number_thousands_decimal_comma():
    assert parse_number("73.274.131,02") == 73274131.02


def test_parse_number_decimal_comma():
    assert parse_number("123,45") == 123.45


def test_parse_number_thousands_comma_decimal_dot():
    assert parse_number("1,234.56") == 1234.56


def test_clean_account_key_excel_dot_zero():
    assert clean_account_key("110505.0") == "110505"


def test_clean_account_key_keep_leading_zeros():
    assert clean_account_key("001234") == "001234"