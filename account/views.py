from django.shortcuts import render
from django.contrib.auth.forms import UserCreationForm
from django.shortcuts import redirect
from django.conf import settings


def user_profile(request):
    return render(request, 'account/profile.html')


def signup(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        print('haha')
        if form.is_valid():
            print("haha")
            user = form.save()
            return redirect(settings.LOGIN_URL)  # default : accouonts/login/
    else:
        form = UserCreationForm()
    return render(request, 'account/signup_form.html', {'form': form, })
