from django.conf.urls import url
from . import apis

urlpatterns = [     # 순차적으로 검사됨
    # For App api
    url(r'^(?P<p_id>\d+)/search/$', apis.predict, name='api_search'),
    url(r'^(?P<p_id>\d+)/train/$', apis.train, name='train'),
    url(r'^search/$', apis.search_image, name='api_search_image'),
    url(r'^upload/$', apis.upload_image, name='api_upload_image'),
    url(r'^recommend/$', apis.recommend_product, name='api_recommend_product'),
    url(r'^create/$', apis.create_project, name='create_project'),
]
