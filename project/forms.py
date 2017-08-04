from django import forms
from .models import Project
from .models import Label


class ProjectForm(forms.ModelForm):
    class Meta:
        model = Project
        fields = ('project_name', 'description', )


class LabelForm(forms.ModelForm):
    class Meta:
        model = Label
        fields = ('label_name', 'description', )
