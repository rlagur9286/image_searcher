from django.conf.urls import url
from . import views
from . import cbv_views

urlpatterns = [     # 순차적으로 검사됨
    # url(r'^$', views.post_list, name='post_list'),
    # url(r'^(?P<pk>\d+)/$', views.post_detail, name='post_detail'),
    # url(r'^(?P<id>\d+)/edit/$', views.post_edit, name='post_edit'),
    # url(r'^new/$', views.post_new, name='post_new'),

    url(r'^cbv/(?P<id>\d+)/$', cbv_views.post_detail, name='post_detail'),
    url(r'^cbv/$', cbv_views.post_list, name='post_list'),
    url(r'^cbv/new/$', cbv_views.post_new, name='post_new'),
    url(r'^cbv/(?P<id>\d+)/edit/$', cbv_views.post_edit, name='post_edit'),
]
