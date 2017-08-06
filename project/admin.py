from django.contrib import admin
from .models import Project
from .models import Label


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ['id', 'project_name', 'description', 'user', 'model', 'is_changed', 'created_at']  # display 리스트 지정


@admin.register(Label)
class LabelAdmin(admin.ModelAdmin):
    list_display = ['id', 'label_name', 'description', 'upload_time']  # display 리스트 지정