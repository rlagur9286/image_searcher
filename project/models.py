from django.db import models
from django.conf import settings
from django.shortcuts import reverse
from django.core.validators import MinLengthValidator
from django import forms


def min_length_3_validator(value):
    if len(value) < 3:
        raise forms.ValidationError('3글자 이상 입력해주세요')


class Project(models.Model):
    project_name = models.CharField(max_length=30, unique=True, validators=[min_length_3_validator])
    description = models.CharField(max_length=255, blank=True)
    model = models.CharField(max_length=255, blank=True)
    is_changed = models.BooleanField(default=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.project_name


class Label(models.Model):
    label_name = models.CharField(max_length=30, validators=[min_length_3_validator])
    upload_time = models.DateTimeField(auto_now_add=True)
    description = models.CharField(max_length=100, blank=True)
    project = models.ForeignKey(Project)

    def __str__(self):
        return self.label_name

    class Meta:
        unique_together = ['project', 'label_name']
        ordering = ['id']   # 오름차순
        # ordering = ['-id']   # 내림차순

    def get_absolute_url(self):
        return reverse('project:list_image', args=[self.id])
