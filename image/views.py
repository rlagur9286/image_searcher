from django.shortcuts import render
from django.http import HttpResponse
import os
from image_searcher import settings
from django.views.generic import View
# Create your views here.

def my_sum(request, x, y):
    return HttpResponse(int(x) + int(y))


def img_download(request):  # 특정 파일 response
    filepath = os.path.join(settings.BASE_DIR, '19.jpg')
    filename = os.path.basename(filepath)
    with open(filepath, 'rb') as f:
        response = HttpResponse(f, content_type='application/jpeg')
        response['Content-Disposition'] = 'attachment; filename="{}"'.format(filename)
        return response
