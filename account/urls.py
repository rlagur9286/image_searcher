from django.conf.urls import url
from django.contrib.auth import views as auth_views
from django.conf import settings
from . import views

urlpatterns = [     # 순차적으로 검사됨
    url(r'^profile/$', views.user_profile, name='profile'),
    url(r'^signup/$', views.signup, name='signup'),
    url(r'^login/$', auth_views.login, name='login', kwargs={'template_name': 'account/login_form.html'}),
    url(r'^logout/$', auth_views.logout, name='logout', kwargs={'next_page': settings.LOGIN_URL}),
]
