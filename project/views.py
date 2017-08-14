import os
import random
import logging
import zipfile
import tarfile
import stat
import shutil

from django.shortcuts import redirect
from django.shortcuts import render
from django.shortcuts import get_object_or_404

from .models import Project
from .forms import ProjectModelForm
from .forms import LabelModelForm
from .models import Label
from image_searcher.settings import BASE_DIR

from django.contrib import messages
from project.engine.utils.configs import ARGS
from project.engine.utils.ops import get_similarity_func

EXTENSIONS = ['.jpg', '.jpeg', '.JPG', '.JPEG', 'jpg']
logging.basicConfig(
    format="[%(name)s][%(asctime)s] %(message)s",
    handlers=[logging.StreamHandler()],
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

similarity_func = get_similarity_func()
ALLOWED_FORMAT = ['zip', 'ZIP', 'tar', 'TAR', 'jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'gif', 'GIF']
IMG_ALLOWED_FORMAT = ['jpg', 'JPG', 'jpeg', 'JPEG']
args = ARGS()
IV4_vec2list_path = os.path.join(BASE_DIR, 'project/engine/vectors/')


def list_project(request):
    result_set = []
    if not request.user.is_authenticated():
        return render(request, 'project/list_project.html', {'project_list': result_set})
    project_qs = Project.objects.filter(user=request.user)
    for project in project_qs:
        label = project.label_set.first()
        if label is None:
            tmp = dict()
            tmp['img'] = None
            tmp['project'] = project
            result_set.append(tmp)
            continue
        dir_path = 'media/images/%s/%s' % (project.id, label.id)
        image_list = []
        for (path, dir, files) in os.walk(dir_path):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext in EXTENSIONS:
                    image_list.append('/' + os.path.join(path, filename))
        if len(image_list) == 0:
            tmp = dict()
            tmp['img'] = None
            tmp['project'] = project
            result_set.append(tmp)
        else:
            random_int = random.randrange(0, len(image_list))
            tmp = dict()
            tmp['img'] = image_list[random_int]
            tmp['project'] = project
            result_set.append(tmp)
    return render(request, 'project/list_project.html', {'project_list': result_set})


def create_project(request):
    form = ProjectModelForm(request.POST, request.FILES)
    if request.method == "POST":
        if form.is_valid():
            project = form.save(commit=False)
            project.user = request.user
            project.save()
            messages.success(request, "새 Project 가 등록되었습니다.")
            dir_path = 'media/images/%s' % project.id
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
    return redirect('root')


def list_label(request, id):
    result_set = []
    project = get_object_or_404(Project, id=id)
    queryset = project.label_set.all()
    for qs in queryset:
        dir_path = 'media/images/%s/%s' % (id, qs.id)
        image_list = []
        for (path, dir, files) in os.walk(dir_path):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext in EXTENSIONS:
                    image_list.append('/' + os.path.join(path, filename))
        if len(image_list) == 0:
            tmp = dict()
            tmp['img'] = None
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


def detail_label(request, p_id, l_id):
    label = get_object_or_404(Label, id=l_id)
    dir_path = 'media/images/%s/%s' % (p_id, label.id)
    image_list = []
    for (path, dir, files) in os.walk(dir_path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext in EXTENSIONS:
                image_list.append('/' + os.path.join(path, filename))
    return render(request, 'project/detail_label.html', {'images': image_list, 'label': label, 'project_id': label.project_id})


def display_prediction(request, p_id):
    project = Project.objects.get(id=p_id)
    return render(request, 'project/display_prediction.html', {'project': project})


def display_pretrained_model(request):
    return render(request, 'project/display_pretrained_model.html')


def create_label(request, p_id):
    project = get_object_or_404(Project, id=p_id)
    form = LabelModelForm(request.POST)
    if request.method == "POST":
        if form.is_valid():
            if project.label_set.filter(label_name=request.POST.get('label_name')).exists():
                return redirect('project:list_label', id=project.id)
            label = form.save(commit=False)
            label.description = request.POST.get('description')
            label.project = project
            file = request.FILES.get('image')
            label.save()
            messages.success(request, "새 Label 가 등록되었습니다.")
            if file is None:
                return redirect('project:list_label', id=project.id)
            if allowed_file(str(file)):
                save_file(file=file, label=label.id, project=project.id)
            return redirect('project:list_label', id=project.id)
    return redirect('project:list_label', id=project.id)


def upload_image(request, p_id, l_id):
    logger.debug(request)
    try:
        if request.method == 'POST':
            file = request.FILES.get('image')
            if file is None:
                return redirect('project:list_label')
            if allowed_file(str(file)):
                save_file(file=file, label=l_id, project=p_id)
                project = Project.objects.get(id=p_id)
                project.is_changed = True
                project.save()
                return redirect('project:detail_label', p_id=p_id, l_id=l_id)
            else:
                return redirect('project:list_label')

    except Exception as exp:
        logger.exception(exp)
        return redirect('root')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_FORMAT


def img_allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in IMG_ALLOWED_FORMAT


def save_file(file, label, project=None):
    if label is None:
        filename = file._get_name()
        dir_path = 'media/upload'
    else:
        filename = file._get_name()
        dir_path = 'media/images/%s/%s' % (project, label)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    fd = open(os.path.join(dir_path, filename), 'wb')
    for chunk in file.chunks():
        fd.write(chunk)
    fd.close()

    if 'zip' in filename or 'ZIP' in filename:
        try:
            zip = zipfile.ZipFile(os.path.join(dir_path, filename))
            zip.extractall(dir_path)
            zip.close()
            os.remove(os.path.join(dir_path, filename))
            return True
        except Exception as e:
            print('ZIP error : ', e)
            return False

    elif 'tar' in filename or 'TAR' in filename:
        try:
            tar = tarfile.open(os.path.join(dir_path, filename))
            tar.extractall(dir_path)
            tar.close()
            os.remove(os.path.join(dir_path, filename))
            return True
        except Exception as e:
            print('ZIP error : ', e)
            return False
    else:
        return os.path.join(dir_path, filename)


def remove_dir_tree(remove_dir):
    try:
        shutil.rmtree(remove_dir, ignore_errors=False, onerror=remove_readonly)
    except PermissionError as e:  ## if failed, report it back to the user ##
        print("[Delete Error] %s - %s." % (e.filename,e.strerror))


def remove_readonly(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)
