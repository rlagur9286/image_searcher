import os
import datetime as dt
import random
from django.shortcuts import render
from django.http import HttpResponse
from pytz import timezone
from django.shortcuts import redirect

from .forms import PostForm
from image.models import Label

EXTENSIONS = ['.jpg', '.jpeg', '.JPG', '.JPEG', 'jpg']


def my_sum(request, x, y):
    return HttpResponse(int(x) + int(y))


def create_label(request):
    form = PostForm(request.POST)
    if request.method == "POST":
        if form.is_valid():
            post = form.save(commit=False)
            post.author = request.user
            post.published_date = dt.datetime.now(timezone('Asia/Seoul'))
            post.save()
            return redirect('root')
    return render(request, 'image/list_label.html', {'form': form})


def list_label(request):
    result_set = []
    queryset = Label.objects.all()
    for qs in queryset:
        dir_path = 'image/static/images/%s' % qs.id
        image_list = []
        for (path, dir, files) in os.walk(dir_path):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext in EXTENSIONS:
                    image_list.append('/' + '/'.join((path + '/' + filename).split('/')[1:]))
        if len(image_list) == 0:
            tmp = dict()
            tmp['img'] = '/static/empty.JPG'
            tmp['label'] = qs.label_name
            tmp['des'] = qs.description
            tmp['id'] = qs.id
            result_set.append(tmp)
        else:
            random_int = random.randrange(0, len(image_list))
            tmp = dict()
            tmp['img'] = image_list[random_int]
            tmp['label'] = qs.label_name
            tmp['des'] = qs.description
            tmp['id'] = qs.id
            result_set.append(tmp)
    return render(request, 'image/list_label.html', {'label_list': result_set})


def detail_label(request, id):
    label = Label.objects.all().get(id=id)
    dir_path = 'image/static/images/%s' % label.id
    image_list = []
    for (path, dir, files) in os.walk(dir_path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext in EXTENSIONS:
                image_list.append('/' + '/'.join((path + '/' + filename).split('/')[1:]))
    return render(request, 'image/detail_label.html', {'images': image_list, 'label': label})


def display_prediction(request):
    return render(request, 'image/display_prediction.html')
# def list_label(request):
#     result_set = []
#     queryset = Label.objects.all()
#     for qs in queryset:
#         tmp = dict()
#         tmp['img'] = Image.objects.all().filter(label_name=qs.label_name)[0]
#         tmp['label'] = qs.label_name
#         tmp['des'] = qs.description
#         result_set.append(tmp)
    # q = request.GET.get('q', '')
    # if q:
    #     queryset = queryset.filter(image_path__icontains=q)
    # return render(request, 'image/list_image.html', {'label_list': result_set, 'q': q})