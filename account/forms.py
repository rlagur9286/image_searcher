from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.forms import AuthenticationForm
from django import forms
from .models import Profile


class SignupForm(UserCreationForm):
    phone_number = forms.CharField(max_length=20)
    address = forms.CharField(max_length=100)

    class Meta(UserCreationForm.Meta):
        # fileds = ('username', 'email')
        fields = UserCreationForm.Meta.fields + ('email', )

    def save(self):
        user = super().save()
        profile = Profile.objects.create(user=user, phone_number=self.cleaned_data.get('phone_number'),
                                         address=self.cleaned_data.get('address'))
        return user


class LoginForm(AuthenticationForm):
    answer = forms.IntegerField(label='3+3 = ?')

    def clean_answer(self):
        answer = self.cleaned_data.get('answer', None)
        if answer != 6:
            raise forms.ValidationError('mismatched')
        return answer
