import os
import random

from django.shortcuts import redirect
from django.shortcuts import render
from django.shortcuts import get_object_or_404

from .models import Project
from .forms import ProjectForm
from .forms import LabelForm
from .models import Label
from .apis import allowed_file
from .apis import save_file

EXTENSIONS = ['.jpg', '.jpeg', '.JPG', '.JPEG', 'jpg']


def list_project(request):
    queryset = Project.objects.all()
    return render(request, 'project/list_project.html', {'project_list': queryset})


def create_project(request):
    form = ProjectForm(request.POST)
    if request.method == "POST":
        if form.is_valid():
            project = form.save(commit=False)
            project.user = request.user
            project.save()
    return redirect('root')


def list_label(request, id):
    result_set = []
    project = get_object_or_404(Project, id=id)
    queryset = project.label_set.all()
    for qs in queryset:
        dir_path = 'project/static/images/%s' % qs.id
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
    return render(request, 'project/list_label.html', {'label_list': result_set, 'project': project})


def detail_label(request, id):
    label = get_object_or_404(Label, id=id)
    dir_path = 'project/static/images/%s' % label.id
    image_list = []
    for (path, dir, files) in os.walk(dir_path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext in EXTENSIONS:
                image_list.append('/' + '/'.join((path + '/' + filename).split('/')[1:]))
    return render(request, 'project/detail_label.html', {'images': image_list, 'label': label, 'project_id': label.project_id})


def display_prediction(request):
    return render(request, 'project/display_prediction.html')


def create_label(request, id):
    project = get_object_or_404(Project, id=id)
    form = LabelForm(request.POST)
    if request.method == "POST":
        print('ha213ha')
        if form.is_valid():
            print('haha')
            label = form.save(commit=False)
            label.description = request.POST.get('description')
            label.project = project
            file = request.FILES.get('image')
            label.save()
            if file is None:
                return redirect('project:list_label')
            if allowed_file(str(file)):
                save_file(file=file, label=label.id)
            return redirect('root')
    return redirect('root')