from django import forms
from .models import Project
from .models import Label


class ProjectModelForm(forms.ModelForm):
    class Meta:
        model = Project
        fields = ('project_name', 'description', )


class LabelModelForm(forms.ModelForm):
    class Meta:
        model = Label
        fields = ('label_name', 'description', )

