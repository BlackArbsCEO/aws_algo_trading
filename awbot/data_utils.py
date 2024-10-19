from decimal import ROUND_HALF_DOWN, Decimal


def convert_to_decimal(
    number: float, precision: int = 2, rounding: str = ROUND_HALF_DOWN
):
    """
    Quantizes a number to a given precision and returns a float.

    Args:
        number (float): The number to quantize.
        precision (float, optional): The precision to quantize to. Defaults to 2.
        rounding (str, optional): The rounding mode to use. Defaults to ROUND_HALF_DOWN.

    Returns:
        Decimal: The quantized number.
    """
    try:
        number_decimal = Decimal(str(number))
    except Exception as e:
        raise Exception(f"unable to convert {number} to decimal: {e}")

    # Set the precision to the desired number of decimal places
    precision_decimal = Decimal("0." + "0" * precision)

    # Quantize the number using ROUND_HALF_DOWN rounding mode
    quantized_decimal = number_decimal.quantize(precision_decimal, rounding=rounding)
    return quantized_decimal


def quantize_number(number, precision=2, rounding=ROUND_HALF_DOWN):
    """Quantizes a number to a given precision and returns a float.

    Args:
      number: The number to quantize.
      precision: The precision to quantize to.

    Returns:
      A float.
    """
    # Convert the input number to a Decimal object
    try:
        number_decimal = Decimal(str(number))
    except Exception as e:
        raise Exception(f"unable to convert {number} to decimal: {e}")

    # Set the precision to the desired number of decimal places
    precision_decimal = Decimal("0." + "0" * precision)

    # Quantize the number using ROUND_HALF_DOWN rounding mode
    quantized_decimal = number_decimal.quantize(precision_decimal, rounding=rounding)

    # Convert the quantized Decimal back to a float
    quantized_float = float(quantized_decimal)

    return quantized_float
