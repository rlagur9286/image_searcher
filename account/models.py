from django.db import models
from django.contrib.auth.models import User
from django.conf import settings
from .validators import phone_number_validator


class Profile(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL)
    phone_number = models.CharField(blank=True, max_length=20, validators=[phone_number_validator])
    address = models.EmailField(blank=True)
    photo = models.ImageField(blank=True, upload_to='account/profile/%Y')
