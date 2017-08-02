from django.shortcuts import render
from django.http import HttpResponse
import os
from pytz import timezone
import datetime as dt
from image_searcher import settings
from image.models import Image
from .forms import PostForm
from .models import Label
from django.shortcuts import redirect
from django.views.generic import View

def my_sum(request, x, y):
    return HttpResponse(int(x) + int(y))


def create_label(request):
    form = PostForm(request.POST)
    if request.method == "POST":
        print('form : ', form)
        print("here?1")
        if form.is_valid():
            print("here?")
            post = form.save(commit=False)
            post.author = request.user
            post.published_date = dt.datetime.now(timezone('Asia/Seoul'))
            post.save()
            return redirect('root')

    print("Rmx")
    return render(request, 'image/list_label.html', {'form': form})


def list_label(request):
    result_set = []
    queryset = Label.objects.all()
    for qs in queryset:
        try:
            image_queryset = Image.objects.filter(label=qs)[0]
        except:
            tmp = dict()
            tmp['img'] = '/static/empty.JPG'
            tmp['label'] = qs.label_name
            tmp['des'] = qs.description
            tmp['id'] = qs.id
            result_set.append(tmp)
            continue
        tmp = dict()
        tmp['img'] = image_queryset.image_name
        tmp['label'] = qs.label_name
        tmp['des'] = qs.description
        tmp['id'] = qs.id
        result_set.append(tmp)
    return render(request, 'image/list_label.html', {'label_list': result_set})


def detail_label(request, id):
    label = Label.objects.all().get(id=id)
    queryset = label.image_set.all()
    return render(request, 'image/detail_label.html', {'images': queryset, 'label': label.label_name})


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