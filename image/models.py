from django.db import models


class Image(models.Model):
    image_path = models.CharField(max_length=255, unique=True)
    upload_time = models.DateTimeField(auto_now_add=True)
    label_name = models.CharField(max_length=255)
