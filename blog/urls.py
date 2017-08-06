from django.conf.urls import url, include
from . import views


urlpatterns = [     # 순차적으로 검사됨
    url(r'^$', views.post_list, name='post_list'),
    url(r'^(?P<id>\d+)/edit/$', views.post_edit, name='post_edit'),
    url(r'^(?P<id>\d+)/$', views.post_detail, name='post_detail'),
    url(r'^new/$', views.post_new, name='post_new'),
]
