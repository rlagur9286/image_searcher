from django import forms
from .models import Post
from django.views.generic import CreateView


class PostModelForm(forms.ModelForm):
    class Meta:  # 아래 코드와 같은 코드임 form을 사용하느냐 ModelForm을 사용하느냐 차이임
        model = Post
        # fields = '__all__'
        fields = '__all__'   # form에서는 유저에게 직접 입력 받을 필드만 써주면 됨. ip같은거는 입력 받을게 아니라
                                        # 자동으로 채워질 필드이므로 view에서 처리해줌


"""
title = forms.CharField(validators=[min_length_3_validator])    # 배열 형태로 validator를 넘겨줌
content = forms.CharField(widget=forms.Textarea)    # 둘다 문자열이지만 입력 받는 문자열이 여러줄 입력받게 보임
# 여기 까지 하면 단순히 해당 입력이 있는지만 검사
def save(self, commit=True):
    post = Post(**self.cleaned_data)
    if commit:
        post.save()
    return post
"""