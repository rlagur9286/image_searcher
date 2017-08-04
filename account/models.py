from django.db import models
from django.contrib.auth.models import User
from django.conf import settings


class Profile(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL)   # Bad case
    phone = models.CharField(max_length=20)
    address = models.CharField(max_length=100)
