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
from django.shortcuts import redirect
from django.conf.urls.static import static
from django.conf import settings


def root(request):
    return redirect('project:list_project')

urlpatterns = [
    url(r'^$', root, name='root'),
    url(r'^admin/', admin.site.urls),
    url(r'^account/', include('account.urls')),
    url(r'^blog/', include('blog.urls', namespace='blog')),
    url(r'^project/', include('project.urls', namespace='project')),
]

# settings.DEBUG 가 False 면 작동 안함
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
