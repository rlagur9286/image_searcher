from django import forms
from .models import Label


class PostForm(forms.ModelForm):
    class Meta:
        model = Label
        fields = ('label_name', 'description', )
