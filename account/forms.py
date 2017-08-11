from django.contrib.auth.forms import UserCreationForm


class SignupForm(UserCreationForm):
    class Meta(UserCreationForm.Meta):
        # fileds = ('username', 'email')
        fields = UserCreationForm.Meta.fields + ('email', )
