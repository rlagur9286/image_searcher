from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect
from django.conf import settings
from .forms import SignupForm
from django.views.decorators.csrf import csrf_exempt


@login_required
def user_profile(request):
    return render(request, 'account/profile.html')


@csrf_exempt
def signup(request):
    if request.method == 'POST':
        form = SignupForm(request.POST)
        if form.is_valid():
            user = form.save()
            return redirect(settings.LOGIN_URL)  # default : accouonts/login/
    else:
        form = SignupForm()
    return render(request, 'account/signup_form.html', {'form': form, })
