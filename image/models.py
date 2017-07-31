from django.db import models


class Image(models.Model):
    STATUS_CHOICES = (
        ('d', 'Draft'),
        ('p', 'Published'),
        ('w', 'Withdrawn'),
    )

    image_path = models.CharField(max_length=255, unique=True)
    upload_time = models.DateTimeField(auto_now_add=True)
    label_name = models.CharField(max_length=255)
    tags = models.CharField(max_length=100, blank=True)
    abs = models.CharField(max_length=10)
    status = models.CharField(max_length=1, choices=STATUS_CHOICES)
