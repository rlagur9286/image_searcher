from django.shortcuts import render
from django.http import HttpResponse
import os
from image_searcher import settings
from .models import Image
from django.views.generic import View

def my_sum(request, x, y):
    return HttpResponse(int(x) + int(y))


def img_download(request):  # 특정 파일 response
    filepath = os.path.join(settings.BASE_DIR, '19.jpg')
    filename = os.path.basename(filepath)
    with open(filepath, 'rb') as f:
        response = HttpResponse(f, content_type='application/jpeg')
        response['Content-Disposition'] = 'attachment; filename="{}"'.format(filename)
        return response


def post_image(request):
    queryset = Image.objects.all()
    return render(request, 'image/post_list.html', {'post_list': queryset, })
