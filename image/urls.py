from django.conf.urls import url, include
from django.contrib import admin
from . import apis
from . import views


urlpatterns = [     # 순차적으로 검사됨
    url(r'^$', views.list_label, name='list_label'),
    url(r'^sum/(?P<x>\d+)/(?P<y>\d+)/$', views.my_sum, name='sum'),    # 시작 ^ 끝 $ 사이에 아무것도 없으므로 아무것도 없으면 이란 뜻
    url(r'^api/train/$', apis.train, name='train'),
    url(r'^api/predict/$', apis.predict, name='predict'),
    url(r'^label/create/$', views.create_label, name='create_label'),
    url(r'^label/detail/(?P<id>\d+)/$', views.detail_label, name='detail_label'),
    url(r'^image/upload/(?P<label>\d+)/$', apis.upload_image, name='upload_image'),
    url(r'^predict/display/$', views.display_prediction, name='display_prediction'),
]
