from decimal import ROUND_HALF_UP, Decimal

import pytest

from awbot.data_utils import convert_to_decimal, quantize_number


def test_convert_to_decimal():
    # Test valid numbers
    assert convert_to_decimal(3.14159) == Decimal("3.14")
    assert convert_to_decimal(2.71828) == Decimal("2.72")
    assert convert_to_decimal(1.41421) == Decimal("1.41")
    assert convert_to_decimal(123.4567, rounding=ROUND_HALF_UP) == Decimal("123.46")

    # Test invalid numbers
    with pytest.raises(Exception):
        convert_to_decimal("Invalid Number")
    with pytest.raises(Exception):
        convert_to_decimal(123.4567, rounding="Invalid Rounding Mode")
    with pytest.raises(Exception):
        convert_to_decimal(123.4567, rounding=10)


def test_quantize_number():
    # Test valid numbers
    assert quantize_number(3.14159) == 3.14
    assert quantize_number(2.71828) == 2.72
    assert quantize_number(1.41421) == 1.41
    assert quantize_number(123.4567, rounding=ROUND_HALF_UP) == 123.46

    # Test invalid numbers
    with pytest.raises(Exception):
        quantize_number("Invalid Number")
    with pytest.raises(Exception):
        quantize_number(123.4567, rounding="Invalid Rounding Mode")
    with pytest.raises(Exception):
        quantize_number(123.4567, rounding=10)
