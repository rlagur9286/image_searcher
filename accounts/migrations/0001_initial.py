# -*- coding: utf-8 -*-
# Generated by Django 1.11.3 on 2017-08-17 00:25
from __future__ import unicode_literals

import accounts.validators
from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Profile',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('phone_number', models.CharField(blank=True, max_length=20, validators=[accounts.validators.phone_number_validator])),
                ('address', models.EmailField(blank=True, max_length=254)),
                ('photo', models.ImageField(blank=True, upload_to='account/profile/%Y')),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]