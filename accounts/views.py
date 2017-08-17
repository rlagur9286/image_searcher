from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.views import login as auth_login

from allauth.socialaccount.models import SocialApp
from allauth.socialaccount.templatetags.socialaccount import get_providers
from .forms import SignupForm, LoginForm


@login_required
def user_profile(request):
    if request.user.profile:
        profile = request.user.profile
    return render(request, 'accounts/profile.html', {'profile': profile})


@csrf_exempt
def signup(request):
    if request.method == 'POST':
        form = SignupForm(request.POST)
        if form.is_valid():
            user = form.save()
            return redirect(settings.LOGIN_URL)  # default : accouonts/login/
    else:
        form = SignupForm()
    return render(request, 'accounts/signup_form.html', {'form': form, })


def login(request):
    providers = []
    for provider in get_providers():
        # SocialApp 속성은 provider에는 없는 속성입니다.
        try:
            provider.social_app = SocialApp.objects.get(provider=provider.id, sites=settings.SITE_ID)
        except SocialApp.DoesNotExist:
            provider.social_app = None
        providers.append(provider)

    return auth_login(request,
                      authentication_form=LoginForm,
                      template_name='accounts/login_form.html',
                      extra_context={'providers': providers})
