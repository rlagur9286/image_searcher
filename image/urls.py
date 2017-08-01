from django.conf.urls import url, include
from django.contrib import admin


from . import views
urlpatterns = [     # 순차적으로 검사됨
    url(r'^sum/(?P<x>\d+)/(?P<y>\d+)/$', views.my_sum),    # 시작 ^ 끝 $ 사이에 아무것도 없으므로 아무것도 없으면 이란 뜻
    url(r'^down/$', views.img_download),
    url(r'^$', views.post_image)
]
