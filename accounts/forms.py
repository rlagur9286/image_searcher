from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.forms import AuthenticationForm
from django import forms
from .models import Profile
from .validators import phone_number_validator


class SignupForm(UserCreationForm):
    phone_number = forms.CharField(validators=[phone_number_validator], required=True,
                                   help_text='Enter the phone number without \'-\'.')
    address = forms.CharField(max_length=30, required=False)
    photo = forms.ImageField(required=False)

    class Meta(UserCreationForm.Meta):
        # fileds = ('username', 'email')
        fields = UserCreationForm.Meta.fields + ('email', )

    def save(self):
        user = super().save()
        profile = Profile.objects.create(user=user, phone_number=self.cleaned_data.get('phone_number'),
                                         address=self.cleaned_data.get('address'),
                                         photo=self.cleaned_data.get('photo'))
        return user


class LoginForm(AuthenticationForm):
    answer = forms.IntegerField(label='3+3 = ?(당신은 로봇입니까)')

    def clean_answer(self):
        answer = self.cleaned_data.get('answer', None)
        if answer != 6:
            raise forms.ValidationError('mismatched')
        return answer
