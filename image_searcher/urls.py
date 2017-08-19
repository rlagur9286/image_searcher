"""image_searcher URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url, include
from django.contrib import admin
from django.shortcuts import render
from django.conf.urls.static import static
from django.conf import settings
from rest_framework import routers, serializers, viewsets
from django.contrib.auth.models import User


class UserSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = User
        fields = ('url', 'username', 'email', 'is_staff')


class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer

router = routers.DefaultRouter()
router.register(r'users', UserViewSet)


def root(request):
    return render(request, 'index.html')


def main(request):
    return render(request, 'main.html')


urlpatterns = [
    url(r'^$', root, name='root'),
    url(r'^main$', main, name='main'),
    url(r'^admin/', admin.site.urls),
    url(r'^accounts/', include('accounts.urls')),
    url(r'^accounts/', include('allauth.urls')),  # include 시에는 $ 표시 금지
    url(r'^blog/', include('blog.urls', namespace='blog')),
    url(r'^project/', include('project.urls', namespace='project')),
    url(r'^api/project/', include('project.api_urls', namespace='api_project')),
    url(r'^api-auth/', include('rest_framework.urls', namespace='rest_framework'))
]

# settings.DEBUG 가 False 면 작동 안함
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
