from django.conf.urls import url
from . import apis

urlpatterns = [     # 순차적으로 검사됨
    # For App api
    url(r'^(?P<p_id>\d+)/search/$', apis.search, name='api_search'),
    url(r'^(?P<p_id>\d+)/train/$', apis.train, name='train'),
    url(r'^search/$', apis.search_image, name='api_search_image'),
    url(r'^recommend/$', apis.recommend_product, name='api_recommend_product'),

    url(r'^create/$', apis.create_project, name='create_project'),
    url(r'^retrieve/$', apis.retrieve_project, name='retrieve_project'),
    url(r'^delete/$', apis.delete_project, name='delete_project'),
    url(r'^create/(?P<p_id>\d+)/$', apis.create_label, name='create_label'),
    url(r'^retrieve/(?P<p_id>\d+)/$', apis.retrieve_label, name='retrieve_label'),
    url(r'^delete/(?P<p_id>\d+)/$', apis.delete_label, name='delete_label'),
    url(r'^upload/(?P<p_id>\d+)/$', apis.upload_image, name='upload_image'),
]
