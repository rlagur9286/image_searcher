from django.conf.urls import url
from . import views

urlpatterns = [     # 순차적으로 검사됨
    url(r'^profile/$', views.user_profile, name='profile'),
    url(r'^signup/$', views.signup, name='signup'),
]
