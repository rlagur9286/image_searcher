from django.conf.urls import url
from . import views
from . import apis

urlpatterns = [     # 순차적으로 검사됨
    url(r'^$', views.list_project, name='list_project'),
    url(r'^(?P<id>\d+)/$', views.list_label, name='list_label'),
    url(r'^create/$', views.create_project, name='create_project'),
    url(r'^api/train/$', apis.train, name='train'),
    url(r'^api/predict/$', apis.predict, name='predict'),
    url(r'^create/(?P<id>\d+)$', views.create_label, name='create_label'),
    url(r'^detail/(?P<id>\d+)/$', views.detail_label, name='detail_label'),
    url(r'^upload/(?P<label>\d+)/$', apis.upload_image, name='upload_image'),
    url(r'^predict/display/$', views.display_prediction, name='display_prediction'),
]
