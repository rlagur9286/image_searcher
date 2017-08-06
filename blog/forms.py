from django import forms


def min_length_3_validator(value):
    if len(value) < 3:
        raise forms.ValidationError('3글자 이상 입력해주세요')


class PostForm(forms.Form):
    title = forms.CharField(validators=[min_length_3_validator])
    content = forms.CharField(widget=forms.Textarea)    # 둘다 문자열이지만 입력 받는 문자열이 여러줄 입력받게 보임
    # 여기 까지 하면 단순히 해당 입력이 있는지만 검사