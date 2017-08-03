import os
import datetime as dt
import random
from django.shortcuts import render
from django.shortcuts import get_object_or_404
from django.http import HttpResponse
from pytz import timezone
from django.shortcuts import redirect
from .apis import allowed_file
from .apis import save_file
from .forms import PostForm
from image.models import Label

EXTENSIONS = ['.jpg', '.jpeg', '.JPG', '.JPEG', 'jpg']


def my_sum(request, x, y):
    return HttpResponse(int(x) + int(y))


def create_label(request):
    form = PostForm(request.POST)
    if request.method == "POST":
        if form.is_valid():
            label = form.save(commit=False)
            label.description = request.POST.get('description')
            file = request.FILES.get('image')
            label.upload_time = dt.datetime.now(timezone('Asia/Seoul'))
            label.save()
            if file is None:
                return redirect('image:list_label')
            if allowed_file(str(file)):
                save_file(file=file, label=label.id)
            return redirect('root')
    return redirect('root')


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
    label = get_object_or_404(Label, id=id)
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
