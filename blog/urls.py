from django.conf.urls import url, include
from . import views
from . import cbv_views
from rest_framework import routers


urlpatterns = [     # 순차적으로 검사됨
    # url(r'^post/$', views.post_list, name='post_list'),
    # url(r'^post/(?P<id>\d+)/$', views.post_detail, name='post_detail'),

    url(r'^post/$', cbv_views.PostList.as_view(), name='cbv_post_list'),
    url(r'^post/(?P<id>\d+)/$', cbv_views.PostDetail.as_view(), name='cbv_post_detail'),
    # url(r'^$', views.post_list, name='post_list'),
    # url(r'^(?P<pk>\d+)/$', views.post_detail, name='post_detail'),
    # url(r'^(?P<id>\d+)/edit/$', views.post_edit, name='post_edit'),
    # url(r'^new/$', views.post_new, name='post_new'),
    # url(r'^cbv/(?P<id>\d+)/$', cbv_views.post_detail, name='post_detail'),
    # url(r'^cbv/$', cbv_views.post_list, name='post_list'),
    # url(r'^cbv/new/$', cbv_views.post_new, name='post_new'),
    # url(r'^cbv/(?P<id>\d+)/edit/$', cbv_views.post_edit, name='post_edit'),
    # url(r'^cbv/(?P<id>\d+)/delete/$', cbv_views.post_delete, name='post_delete'),
]
