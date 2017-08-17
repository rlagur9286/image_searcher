import re
from django.forms import ValidationError


def phone_number_validator(value):
    if not re.match(r"^010[1-9]\d{7}$", value):
        raise ValidationError('{} is not an phone number'.format(value))
