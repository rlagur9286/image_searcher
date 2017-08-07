import os
import random
import tensorflow as tf
import pickle
import timeit
import heapq
import logging
import zipfile
import tarfile
import stat
import shutil
import datetime
import time

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
from project.engine.utils.vector_file_handler import save_vec2list
from project.engine.Incept_v4_Trainer import Incept_v4_Trainer
from project.engine.utils import configs
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
    project_qs = Project.objects.all()
    for project in project_qs:
        label = project.label_set.first()
        if label is None:
            tmp = dict()
            tmp['img'] = '/static/empty.JPG'
            tmp['project'] = project
            result_set.append(tmp)
            continue
        dir_path = 'project/static/images/%s/%s' % (project.id, label.id)
        image_list = []
        for (path, dir, files) in os.walk(dir_path):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext in EXTENSIONS:
                    image_list.append('/' + '/'.join((path + '/' + filename).split('/')[1:]))
        if len(image_list) == 0:
            tmp = dict()
            tmp['img'] = '/static/empty.JPG'
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
            dir_path = 'project/static/images/%s' % project.id
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
    return redirect('root')


def list_label(request, id):
    result_set = []
    project = get_object_or_404(Project, id=id)
    queryset = project.label_set.all()
    for qs in queryset:
        dir_path = 'project/static/images/%s/%s' % (id, qs.id)
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


def detail_label(request, p_id, l_id):
    label = get_object_or_404(Label, id=l_id)
    dir_path = 'project/static/images/%s/%s' % (p_id, label.id)
    image_list = []
    for (path, dir, files) in os.walk(dir_path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext in EXTENSIONS:
                image_list.append('/' + '/'.join((path + '/' + filename).split('/')[1:]))
    return render(request, 'project/detail_label.html', {'images': image_list, 'label': label, 'project_id': label.project_id})


def display_prediction(request, p_id):
    project = Project.objects.get(id=p_id)
    return render(request, 'project/display_prediction.html', {'project': project})


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


def predict(request, p_id):
    logger.debug(request)
    try:
        if request.method == 'POST':
            start = timeit.default_timer()
            output_graph = configs.output_graph + str(p_id) + '/output_graph.pb'
            with tf.gfile.FastGFile(os.path.join(output_graph), 'rb') as fp:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(fp.read())
                tf.import_graph_def(graph_def, name='')
            config = tf.ConfigProto(allow_soft_placement=True)
            iv4_sess = tf.Session(config=config)
            iv4_bottleneck = iv4_sess.graph.get_tensor_by_name('input/BottleneckInputPlaceholder:0')

            with open(IV4_vec2list_path + str(p_id) + '/vectors_i4_app/vec2list.pickle', 'rb') as handle:
                iv4_vector_list = pickle.load(handle)

            file = request.FILES.get('image')
            if file is None:
                return render(request, 'project/display_prediction.html')
            if img_allowed_file(str(file)):
                img_path = save_file(file=file, label=None)
            else:
                return render(request, 'project/display_prediction.html')

            # For inception v4
            iv4_img_list = {}
            iv4_image = tf.gfile.FastGFile(img_path, 'rb').read()
            iv4_image_vector = iv4_sess.run(iv4_bottleneck, {'DecodeJpeg/contents:0': iv4_image})
            for vec in iv4_vector_list:
                dist = similarity_func(iv4_image_vector, vec[1])
                iv4_img_list[vec[0]] = dist
            iv4_keys_sorted = heapq.nsmallest(5, iv4_img_list, key=iv4_img_list.get)
            iv4_images = []
            for result in iv4_keys_sorted:
                tmp = dict()
                tmp['distance'] = iv4_img_list.get(result)
                tmp['img'] = '/' + '/'.join(result.replace('\\', '/').split('/')[1:])
                tmp['label'] = Label.objects.all().get(id=int(result.replace('\\', '/').split('/')[-2])).label_name
                iv4_images.append(tmp)

            end = timeit.default_timer()
            print('Time to load : ', end - start)
            print('ICEPTION : ', iv4_images)
            return render(request, 'project/display_prediction.html', {'images': iv4_images, 'project': Project.objects.get(id=p_id)})
    except Exception as exp:
        logger.exception(exp)
        return render({'result': False, 'reason': 'INTERNAL SERVER ERROR'})


def train(request, p_id):
    logger.debug(request)
    try:
        project = Project.objects.get(id=p_id)
        if not project.is_changed:
            return redirect('project:list_label', id=p_id)
        imgage_dir = args.image_dir + '/' + str(p_id)
        vector_path = args.vector_path + '/' + str(p_id)
        if not os.path.exists(vector_path):
            os.mkdir(vector_path)
        output_graph = args.output_graph + '/' + str(p_id) + '/output_graph.pb'
        output_labels = args.output_labels + '/' + str(p_id) + '/output_labels.txt'
        if not os.path.exists(args.output_graph + '/' + str(p_id)):
            os.mkdir(args.output_graph + '/' + str(p_id))
        bottleneck_dir = args.bottleneck_dir + '/' + str(p_id)
        if not os.path.exists(bottleneck_dir):
            os.mkdir(bottleneck_dir)
        check_point_path = args.check_point_path + '/' + str(p_id)
        if not os.path.exists(check_point_path):
            os.mkdir(check_point_path)
        summaries_dir = args.summaries_dir + '/' + str(p_id)
        if not os.path.exists(summaries_dir):
            os.mkdir(summaries_dir)

        trainer = Incept_v4_Trainer(image_dir=imgage_dir, output_graph=output_graph,
                                    output_labels=output_labels, vector_path=vector_path,
                                    summaries_dir=summaries_dir,
                                    how_many_training_steps=args.how_many_training_steps,
                                    learning_rate=args.learning_rate, testing_percentage=args.testing_percentage,
                                    eval_step_interval=args.eval_step_interval,
                                    train_batch_size=args.train_batch_size, test_batch_size=args.test_batch_size,
                                    validation_batch_size=args.validation_batch_size,
                                    print_misclassified_test_images=args.print_misclassified_test_images,
                                    model_dir=args.model_dir,
                                    bottleneck_dir=bottleneck_dir, final_tensor_name=args.final_tensor_name,
                                    flip_left_right=args.flip_left_right, random_crop=args.random_crop,
                                    random_scale=args.random_scale, random_brightness=args.random_brightness,
                                    check_point_path=check_point_path, max_ckpts_to_keep=args.max_ckpts_to_keep,
                                    gpu_list=args.gpu_list, validation_percentage=args.validation_percentage)
        trainer.do_train_with_GPU(gpu_list=['/gpu:0'])
        vector_actual_path = trainer.vectorize_with_GPU(gpu_list=['/gpu:0'])
        save_vec2list(vector_actual_path=vector_actual_path)
        remove_dir_tree(check_point_path)
        remove_dir_tree(summaries_dir)
        project.model = str(project.id) + '_' + str(int(time.mktime(datetime.datetime.now().timetuple())))
        project.is_changed = True
        project.save()
        return redirect('project:list_label', id=p_id)

    except Exception as exp:
        logger.exception(exp)
        return redirect('root')


def upload_image(request, p_id, l_id):
    logger.debug(request)
    try:
        if request.method == 'POST':
            file = request.FILES.get('image')
            if file is None:
                return redirect('project:list_label')
            if allowed_file(str(file)):
                save_file(file=file, label=l_id, project=p_id)
                return redirect('project:list_label')
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
        dir_path = 'project/static/upload'
    else:
        filename = file._get_name()
        dir_path = 'project/static/images/%s/%s' % (project, label)
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
