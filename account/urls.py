from django.conf.urls import url
from . import views

urlpatterns = [     # 순차적으로 검사됨
    url(r'^/$', views.user_profile),
]
