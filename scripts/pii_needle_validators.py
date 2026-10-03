"""Checksum and shape validators for identifier needles.

Each validator takes the matched surface and returns True only when its
structure is valid; separators (spaces, hyphens, dots, slashes) are ignored
where the format allows them. Validators raise precision; they never assert
that a value is real or personal.
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Callable

_SEPARATORS = re.compile(r"[\s\-./]")


def _digits(value: str) -> str:
    return _SEPARATORS.sub("", value)


def luhn(value: str) -> bool:
    digits = _digits(value)
    if not digits.isdigit() or len(digits) < 12:
        return False
    total = 0
    for index, char in enumerate(reversed(digits)):
        digit = int(char)
        if index % 2:
            digit *= 2
            digit -= 9 if digit > 9 else 0
        total += digit
    return total % 10 == 0


def payment_card(value: str) -> bool:
    digits = _digits(value)
    return 13 <= len(digits) <= 19 and digits[0] in "23456" and luhn(digits)


def imei(value: str) -> bool:
    digits = _digits(value)
    return len(digits) == 15 and luhn(digits)


def iban(value: str) -> bool:
    compact = re.sub(r"\s", "", value).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", compact):
        return False
    rearranged = compact[4:] + compact[:4]
    number = "".join(str(int(char, 36)) for char in rearranged)
    return int(number) % 97 == 1


def pesel(value: str) -> bool:
    digits = _digits(value)
    if not re.fullmatch(r"\d{11}", digits):
        return False
    weights = (1, 3, 7, 9, 1, 3, 7, 9, 1, 3)
    check = (10 - sum(int(d) * w for d, w in zip(digits, weights)) % 10) % 10
    month = int(digits[2:4]) % 20
    return check == int(digits[10]) and 1 <= month <= 12 and 1 <= int(digits[4:6]) <= 31


def rodne_cislo(value: str) -> bool:
    digits = _digits(value)
    if not re.fullmatch(r"\d{10}", digits):
        return False
    month = int(digits[2:4])
    if month > 70:
        month -= 70
    elif month > 50:
        month -= 50
    elif month > 20:
        month -= 20
    if not (1 <= month <= 12 and 1 <= int(digits[4:6]) <= 31):
        return False
    remainder = int(digits[:9]) % 11
    return (0 if remainder == 10 else remainder) == int(digits[9])


def tckn(value: str) -> bool:
    digits = _digits(value)
    if not re.fullmatch(r"[1-9]\d{10}", digits):
        return False
    d = [int(c) for c in digits]
    tenth = ((d[0] + d[2] + d[4] + d[6] + d[8]) * 7 - (d[1] + d[3] + d[5] + d[7])) % 10
    return tenth == d[9] and sum(d[:10]) % 10 == d[10]


_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def aadhaar(value: str) -> bool:
    digits = _digits(value)
    if not re.fullmatch(r"[2-9]\d{11}", digits):
        return False
    check = 0
    for index, char in enumerate(reversed(digits)):
        check = _VERHOEFF_D[check][_VERHOEFF_P[index % 8][int(char)]]
    return check == 0


def korean_rrn(value: str) -> bool:
    digits = _digits(value)
    if not re.fullmatch(r"\d{6}[1-4]\d{6}", digits):
        return False
    weights = (2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5)
    check = (11 - sum(int(d) * w for d, w in zip(digits, weights)) % 11) % 10
    return check == int(digits[12]) and 1 <= int(digits[2:4]) <= 12


def chinese_resident_id(value: str) -> bool:
    compact = _digits(value).upper()
    if not re.fullmatch(r"\d{17}[\dX]", compact):
        return False
    weights = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
    check = "10X98765432"[sum(int(d) * w for d, w in zip(compact, weights)) % 11]
    return check == compact[17] and 1 <= int(compact[10:12]) <= 12


def israeli_id(value: str) -> bool:
    digits = _digits(value)
    if not re.fullmatch(r"\d{5,9}", digits):
        return False
    digits = digits.zfill(9)
    total = 0
    for index, char in enumerate(digits):
        product = int(char) * (1 if index % 2 == 0 else 2)
        total += product - 9 if product > 9 else product
    return total % 10 == 0 and int(digits) > 0


def snils(value: str) -> bool:
    digits = _digits(value)
    if not re.fullmatch(r"\d{11}", digits):
        return False
    total = sum(int(d) * (9 - i) for i, d in enumerate(digits[:9]))
    check = total % 101
    return (0 if check in (100, 101) else check) == int(digits[9:])


def ipv4(value: str) -> bool:
    try:
        address = ipaddress.IPv4Address(value)
    except ValueError:
        return False
    return not address.is_unspecified and value.count(".") == 3


def ipv6(value: str) -> bool:
    try:
        ipaddress.IPv6Address(value)
    except ValueError:
        return False
    return value.count(":") >= 3


def latitude_longitude(value: str) -> bool:
    numbers = re.findall(r"[-+]?\d{1,3}\.\d+", value)
    if len(numbers) != 2:
        return False
    latitude, longitude = (float(number) for number in numbers)
    return -90 <= latitude <= 90 and -180 <= longitude <= 180 and (latitude, longitude) != (0.0, 0.0)


VALIDATORS: dict[str, Callable[[str], bool]] = {
    "luhn": luhn,
    "payment_card": payment_card,
    "imei": imei,
    "iban": iban,
    "pesel": pesel,
    "rodne_cislo": rodne_cislo,
    "tckn": tckn,
    "aadhaar": aadhaar,
    "korean_rrn": korean_rrn,
    "chinese_resident_id": chinese_resident_id,
    "israeli_id": israeli_id,
    "snils": snils,
    "ipv4": ipv4,
    "ipv6": ipv6,
    "latitude_longitude": latitude_longitude,
}
