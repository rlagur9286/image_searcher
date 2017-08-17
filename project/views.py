import logging
import os
import heapq
import random
import shutil
import stat
import tarfile
import pickle
import zipfile
import tensorflow as tf

from django.contrib import messages
from django.shortcuts import get_object_or_404
from django.shortcuts import redirect
from django.shortcuts import render
from django.http import JsonResponse

from .engine.utils import configs
from .engine.utils.database import ImageManager
from image_searcher.settings.common import BASE_DIR
from project.engine.utils.configs import ARGS
from project.engine.utils.ops import get_similarity_func
from .forms import LabelModelForm
from .forms import ProjectModelForm
from .models import Label
from .models import Project

EXTENSIONS = ['.jpg', '.jpeg', '.JPG', '.JPEG', 'jpg']
logging.basicConfig(
    format="[%(name)s][%(asctime)s] %(message)s",
    handlers=[logging.StreamHandler()],
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

ALLOWED_FORMAT = ['zip', 'ZIP', 'tar', 'TAR', 'jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'gif', 'GIF']
IMG_ALLOWED_FORMAT = ['jpg', 'JPG', 'jpeg', 'JPEG']
UPLOAD_FOLDER = 'media/upload/'
args = ARGS()
IV4_vec2list_path = 'project/engine/vectors/vectors_i4_app/vec2list.pickle'
vec2list_path = os.path.join(BASE_DIR, 'project/engine/vectors/')
similarity_func = get_similarity_func()

image_db = ImageManager()
with tf.gfile.FastGFile(configs.output_graph + 'output_graph.pb', 'rb') as fp:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fp.read())
    tf.import_graph_def(graph_def, name='')
config = tf.ConfigProto(allow_soft_placement=True)
iv4_sess = tf.Session(config=config)
iv4_bottleneck = iv4_sess.graph.get_tensor_by_name('input/BottleneckInputPlaceholder:0')
logits = iv4_sess.graph.get_tensor_by_name('final_result:0')

with open(IV4_vec2list_path, 'rb') as handle:
    iv4_vector_list = pickle.load(handle)


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
    return redirect('project:list_project')


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
                project.is_changed = True
                project.save()
            return redirect('project:list_label', id=project.id)
    return redirect('project:list_label', id=project.id)


def search(request, p_id):
    logger.debug(request)
    try:
        if request.method == 'POST':
            file = request.FILES.get('image')
            if file is None:
                return render(request, 'project/display_prediction.html', {'project': Project.objects.get(id=p_id)})

            if img_allowed_file(str(file)):
                img_path = 'media/upload/' + file.name
                save_file(file=file)
            else:
                return render(request, 'project/display_prediction.html', {'project': Project.objects.get(id=p_id)})

            project = get_object_or_404(Project, id=p_id)
            if not project.model:
                return render(request, 'project/display_prediction.html', {'project': Project.objects.get(id=p_id)})

            output_graph = configs.output_graph + str(p_id) + '/output_graph.pb'
            with tf.gfile.FastGFile(os.path.join(output_graph), 'rb') as f:
                graph = tf.GraphDef()
                graph.ParseFromString(f.read())
                tf.import_graph_def(graph, name='')
            conf = tf.ConfigProto(allow_soft_placement=True)
            sess = tf.Session(config=conf)
            bottleneck = sess.graph.get_tensor_by_name('input/BottleneckInputPlaceholder:0')

            with open(vec2list_path + str(p_id) + '/vectors_i4_app/vec2list.pickle', 'rb') as f:
                vector_list = pickle.load(f)

            # For inception v4
            iv4_img_list = {}
            iv4_image = tf.gfile.FastGFile(img_path, 'rb').read()
            iv4_image_vector = sess.run(bottleneck, {'DecodeJpeg/contents:0': iv4_image})
            for vec in vector_list:
                dist = similarity_func(iv4_image_vector, vec[1])
                iv4_img_list[vec[0]] = dist
            iv4_keys_sorted = heapq.nsmallest(5, iv4_img_list, key=iv4_img_list.get)
            iv4_images = []
            for result in iv4_keys_sorted:
                tmp = dict()
                tmp['distance'] = iv4_img_list.get(result)
                tmp['img'] = '/' + result.replace('\\', '/')
                tmp['label'] = Label.objects.all().get(id=int(result.replace('\\', '/').split('/')[-2])).label_name
                iv4_images.append(tmp)

            print('ICEPTION : ', iv4_images)
            return render(request, 'project/display_prediction.html', {'images': iv4_images, 'project': Project.objects.get(id=p_id)})
    except Exception as exp:
        logger.exception(exp)
        return redirect('root')


def pretrained_predict(request):
    if request.method == 'POST':
        try:
            result_set = dict()
            file = request.FILES.get('image')
            if not file:
                return JsonResponse({'success': False, 'reason': '파일은 필수 입니다.'})
            img_path = UPLOAD_FOLDER + file.name
            if not allowed_file(file.name):
                return JsonResponse({'success': False, 'reason': '파일은 형식을 확인해주세요'})
            save_file(file=file)

            # For inception v4 Model
            img_list = {}
            image = tf.gfile.FastGFile(img_path, 'rb').read()
            image_vector = iv4_sess.run(iv4_bottleneck, {'DecodeJpeg/contents:0': image})
            labels = [line.rstrip() for line in tf.gfile.GFile(configs.output_graph + 'output_labels.txt')]
            prediction = iv4_sess.run(logits, {'DecodeJpeg/contents:0': image})
            s_label = heapq.nlargest(3, range(len(prediction[0])), prediction[0].__getitem__)
            s_label = [labels[idx] for idx in s_label]
            selected_list = [v for v in iv4_vector_list if v[0].split('/')[0].split('\\')[1] in s_label]

            for vec in selected_list:
                dist = similarity_func(image_vector, vec[1])
                img_list[vec[0]] = dist

            keys_sorted = heapq.nsmallest(5, img_list, key=img_list.get)

            products = []
            for result in keys_sorted:
                product = image_db.retrieve_info_by_PRODUCT_CD(PRODUCT_CD=str('/' + result.replace('\\', '/')).split('/')[-1].split('.jpg')[0])
                if product is None:
                    continue
                products.append(product)

            result_set['products'] = products
            return render(request, 'project/display_pretrained_model.html', {'result': result_set})

        except Exception as exp:
            print(exp)
            return redirect('root')

    else:
        return JsonResponse({'success': False, 'message': '이 method는 POST 만 지원합니다.'})


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


def save_file(file, label=None, project=None):
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
