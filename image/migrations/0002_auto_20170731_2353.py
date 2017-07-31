# -*- coding: utf-8 -*-
# Generated by Django 1.11.3 on 2017-07-31 14:53
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('image', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='image',
            name='tags',
            field=models.CharField(blank=True, max_length=100),
        ),
        migrations.AlterField(
            model_name='image',
            name='image_path',
            field=models.CharField(max_length=255, unique=True),
        ),
    ]