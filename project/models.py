from django.db import models
from django.conf import settings
from django.shortcuts import reverse
from django import forms


def min_length_2_validator(value):
    if len(value) < 2:
        raise forms.ValidationError('2글자 이상 입력해주세요')


class Project(models.Model):
    project_name = models.CharField(max_length=30, unique=True, validators=[min_length_2_validator])
    description = models.CharField(max_length=255, blank=True)
    model = models.CharField(max_length=255, blank=True)
    is_changed = models.BooleanField(default=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.project_name

    def get_absolute_url(self):
        return reverse('project:list_label', args=[self.id])


class Label(models.Model):
    label_name = models.CharField(max_length=30, validators=[min_length_2_validator], unique=True)
    upload_time = models.DateTimeField(auto_now_add=True)
    description = models.CharField(max_length=100, blank=True)
    project = models.ForeignKey(Project, on_delete=models.CASCADE)

    def __str__(self):
        return self.label_name

    class Meta:
        ordering = ['id']   # 오름차순
        # ordering = ['-id']   # 내림차순

    def get_absolute_url(self):
        return reverse('project:detail_label', args=[self.project_id, self.id])


class Product(models.Model):
    PRODUCT_CD = models.CharField(max_length=255, unique=True)
    GOODS_NAME = models.CharField(max_length=255, blank=True)
    GOODS_IMAGE_URL = models.CharField(max_length=255, blank=True)
    BRAND = models.CharField(max_length=255, blank=True)
    MODEL = models.CharField(max_length=255, blank=True)
    CATEGORY = models.CharField(max_length=255, blank=True)
    PRICE = models.FloatField(blank=False)

    def __str__(self):
        return self.PRODUCT_CD

    class Meta:
        ordering = ['id']   # 오름차순
