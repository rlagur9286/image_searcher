from django.conf.urls import url
from . import views

urlpatterns = [     # 순차적으로 검사됨
    url(r'^$', views.list_project, name='list_project'),
    url(r'^(?P<id>\d+)/$', views.list_label, name='list_label'),
    url(r'^create/$', views.create_project, name='create_project'),
    url(r'^train/(?P<p_id>\d+)$', views.train, name='train'),
    url(r'^(?P<p_id>\d+)/predict/$', views.predict, name='predict'),
    url(r'^create/(?P<p_id>\d+)$', views.create_label, name='create_label'),
    url(r'^(?P<p_id>\d+)/(?P<l_id>\d+)/$', views.detail_label, name='detail_label'),
    url(r'^(?P<p_id>\d+)/upload/(?P<l_id>\d+)/$', views.upload_image, name='upload_image'),
    url(r'^(?P<p_id>\d+)/prediction/$', views.display_prediction, name='display_prediction'),
]
