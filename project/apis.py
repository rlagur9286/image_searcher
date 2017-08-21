import datetime
import heapq
import logging
import os
import pickle
import shutil
import stat
import time
import json
import random
import tensorflow as tf
import zipfile
import tarfile
import re

from django.http import JsonResponse
from django.shortcuts import redirect
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.models import User
from django.http import HttpResponse
from rest_framework.renderers import JSONRenderer

from image_searcher.settings.common import BASE_DIR
from project.engine.Incept_v4_Trainer import Incept_v4_Trainer
from project.engine.utils.configs import ARGS
from project.engine.utils.vector_file_handler import save_vec2list
from .engine.utils import configs
from .engine.utils.database import ImageManager
from .engine.utils.ops import get_similarity_func
from .models import Label
from .models import Project
from .serializers import ProjectSerializer
from .serializers import LabelSerializer
from .forms import LabelModelForm

logging.basicConfig(
    format="[%(name)s][%(asctime)s] %(message)s",
    handlers=[logging.StreamHandler()],
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)
args = ARGS()

hangul = re.compile('[^ ㄱ-ㅣ가-힣0-9]+')
ALLOWED_FORMAT = ['zip', 'ZIP', 'tar', 'TAR', 'jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'gif', 'GIF']
IMG_ALLOWED_FORMAT = ['jpg', 'JPG', 'jpeg', 'JPEG']
IV4_vec2list_path = 'project/engine/vectors/vectors_i4_app/vec2list.pickle'
vec2list_path = os.path.join(BASE_DIR, 'project/engine/vectors/')
UPLOAD_FOLDER = 'media/api_upload/'
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

similarity_func = get_similarity_func()

image_db = ImageManager()
info_all = image_db.retrieve_info_all()
with tf.gfile.FastGFile(configs.output_graph + 'output_graph.pb', 'rb') as fp:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fp.read())
    tf.import_graph_def(graph_def, name='')
config = tf.ConfigProto(allow_soft_placement=True)
iv4_sess = tf.Session(config=config)
iv4_bottleneck = iv4_sess.graph.get_tensor_by_name('input/BottleneckInputPlaceholder:0')
logits = iv4_sess.graph.get_tensor_by_name('final_result:0')

# with open(IV4_vec2list_path, 'rb') as handle:
#     iv4_vector_list = pickle.load(handle)


@csrf_exempt
def create_project(request):
    if request.method == 'POST':
        result_set = dict()
        data = json.loads(request.body.decode('utf-8'))
        logger.debug("INPUT %s", data)

        project_name = data.get('project_name')
        description = data.get('description')
        user_id = data.get('user_id')

        if not user_id:
            return JsonResponse({'success': False, 'result': None, 'message': 'user_id는 필수 입니다.'})

        try:
            user = User.objects.get(username=user_id)
        except Exception as e:
            return JSONResponse({'success': False, 'result': 'user_id를 확인 해주세요', 'message': str(e)})

        if not project_name:
            return JsonResponse({'success': False, 'result': None, 'message': 'project_name는 필수 입니다'})
        try:
            project = Project.objects.create(project_name=project_name, description=description, user=user)
        except Exception as e:
            return JsonResponse({'success': False, 'result': None, 'message': str(e)})
        dir_path = 'media/images/%s' % project.id
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        result_set['id'] = project.id
        result_set['project_name'] = project.project_name
        result_set['description'] = project.description

        return JsonResponse({'success': True, 'result': result_set,
                             'message': '새 Project[{}] 가 등록되었습니다.'.format(project.project_name)})


@csrf_exempt
def retrieve_project(request):
    if request.method == 'GET':
        data = json.loads(json.dumps(request.GET))
        logger.debug("[GET]: %s", data)
        user_id = data.get('user_id')
        p_id = data.get('p_id')

        if not user_id:
            return JsonResponse({'success': False, 'result': None, 'message': 'user_id는 필수 입니다.'})

        try:
            user = User.objects.get(username=user_id)
        except Exception as e:
            return JSONResponse({'success': False, 'result': 'user_id를 확인 해주세요', 'message': str(e)})

        if not p_id:
            projects = Project.objects.filter(user=user)
            serialized_project = ProjectSerializer(projects, many=True)
        else:
            try:
                project = Project.objects.get(id=p_id, user=user)
            except Exception as e:
                return JSONResponse({'success': False, 'result': None, 'message': str(e)})
            serialized_project = ProjectSerializer(project)
        return JSONResponse({'success': True, 'result': serialized_project.data, 'message': '성공!'})


@csrf_exempt
def delete_project(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        logger.debug("INPUT %s", data)

        user_id = data.get('user_id')
        p_id = data.get('p_id')

        if not user_id:
            return JsonResponse({'success': False, 'result': None, 'message': 'user_id는 필수 입니다.'})

        try:
            user = User.objects.get(username=user_id)
        except Exception as e:
            return JSONResponse({'success': False, 'result': 'user_id를 확인 해주세요', 'message': str(e)})

        if not p_id:
            return JSONResponse({'success': False, 'result': 'p_id는 필수 입니다.', 'message': None})

        try:
            project = Project.objects.get(id=p_id, user=user)
            project.delete()
            project_path = 'media/images/%s' % p_id
            if os.path.exists(project_path):
                remove_dir_tree(project_path)
        except Exception as e:
            return JSONResponse({'success': False, 'result': 'p_id를 확인 해주세요', 'message': str(e)})
        return JsonResponse({'success': True, 'result': str(p_id) + ' 가 성공적으로 지워졌습니다.', 'message': '성공!'})


@csrf_exempt
def create_label(request, p_id):
    if request.method == 'POST':
        result_set = dict()
        label_name = request.POST.get('label_name')
        description = request.POST.get('description')
        if not description:
            description = ''
        user_id = request.POST.get('user_id')

        if not label_name or not user_id:
            return JsonResponse({'success': False, 'result': None, 'message': 'label_name, user_id는 필수입니다.'})

        try:
            user = User.objects.get(username=user_id)
        except Exception as e:
            return JSONResponse({'success': False, 'result': 'user_id를 확인 해주세요', 'message': str(e)})

        try:
            project = Project.objects.get(id=p_id, user=user)
        except Exception as e:
            return JsonResponse({'success': False, 'result': None, 'message': str(e)})
        try:
            file = request.FILES.get('image')
            label = Label.objects.create(label_name=label_name, project=project, description=description)
            if file:
                if allowed_file(str(file)):
                    save_file(file=file, label=label.id, project=project.id)
                    project.is_changed = True
                    project.save()
        except Exception as e:
            return JsonResponse({'success': False, 'result': None, 'message': str(e)})

        result_set['id'] = label.id
        result_set['label_name'] = label.label_name
        result_set['description'] = label.description
        return JsonResponse({'success': True, 'result': result_set,
                             'message': '새 Label[{}] 가 등록되었습니다.'.format(label.label_name)})


@csrf_exempt
def retrieve_label(request, p_id):
    if request.method == 'GET':
        data = json.loads(json.dumps(request.GET))
        logger.debug("[GET]: %s", data)
        user_id = data.get('user_id')
        l_id = data.get('l_id')

        if not user_id:
            return JsonResponse({'success': False, 'result': None, 'message': 'user_id, p_id는 필수 입니다.'})

        try:
            user = User.objects.get(username=user_id)
        except Exception as e:
            return JSONResponse({'success': False, 'result': 'user_id를 확인 해주세요', 'message': str(e)})

        try:
            project = Project.objects.get(id=p_id, user=user)
        except Exception as e:
            return JSONResponse({'success': False, 'result': None, 'message': str(e)})

        if not l_id:
            labels = project.label_set
            serialized_label = LabelSerializer(labels, many=True)
        else:
            try:
                label = Label.objects.get(project=project, id=l_id)
            except Exception as e:
                return JSONResponse({'success': False, 'result': None, 'message': str(e)})
            serialized_label = LabelSerializer(label)
        return JSONResponse({'success': True, 'result': serialized_label.data, 'message': '성공!'})


@csrf_exempt
def delete_label(request, p_id):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        logger.debug("INPUT %s", data)

        user_id = data.get('user_id')
        l_id = data.get('l_id')

        if not user_id or not l_id:
            return JsonResponse({'success': False, 'result': None, 'message': 'user_id, l_id는 필수 입니다.'})

        try:
            user = User.objects.get(username=user_id)
        except Exception as e:
            return JSONResponse({'success': False, 'result': 'user_id를 확인 해주세요', 'message': str(e)})

        try:
            project = Project.objects.get(id=p_id, user=user)
            label = Label.objects.get(project=project, id=l_id)
            label.delete()
            label_path = 'media/images/%s/%s' % (p_id, l_id)
            if os.path.exists(label_path):
                remove_dir_tree(label_path)
        except Exception as e:
            return JSONResponse({'success': False, 'result': 'l_id 또는 p_id를 확인 해주세요', 'message': str(e)})
        return JsonResponse({'success': True, 'result': str(l_id) + ' 가 성공적으로 지워졌습니다.', 'message': '성공!'})


@csrf_exempt
def search_image(request):
    if request.method == 'POST':
        try:
            result_set = dict()

            # Get Variable for Meta data Info
            brand = request.POST.get('brand')
            class_type = request.POST.get('class_type')

            if class_type is not None:
                if not os.path.isdir(UPLOAD_FOLDER + class_type):
                    os.mkdir(UPLOAD_FOLDER + class_type)
                image_dir = UPLOAD_FOLDER + class_type + '/'
            else:
                image_dir = UPLOAD_FOLDER
            file = request.FILES.get('image')

            if not file:
                return JsonResponse({'success': False, 'message': '파일은 필수 입니다.'})
            img_path = image_dir + file.name
            if not allowed_file(file.name):
                return JsonResponse({'success': False, 'message': '파일은 형식을 확인해주세요'})
            save_file(file=file, img_path=img_path)

            # For inception v4 Model
            img_list = {}
            image = tf.gfile.FastGFile(img_path, 'rb').read()
            image_vector = iv4_sess.run(iv4_bottleneck, {'DecodeJpeg/contents:0': image})
            labels = [line.rstrip() for line in tf.gfile.GFile(configs.output_graph + 'output_labels.txt')]
            prediction = iv4_sess.run(logits, {'DecodeJpeg/contents:0': image})
            s_label = heapq.nlargest(3, range(len(prediction[0])), prediction[0].__getitem__)
            s_label = [labels[idx] for idx in s_label]
            # selected_list = [v for v in iv4_vector_list if v[0].split('/')[0].split('\\')[1] in s_label]

            # for vec in selected_list:
            #     dist = similarity_func(image_vector, vec[1])
            #     img_list[vec[0]] = dist

            keys_sorted = heapq.nsmallest(5, img_list, key=img_list.get)

            products = []
            for result in keys_sorted:
                product = image_db.retrieve_info_by_PRODUCT_CD(PRODUCT_CD=str('/' + result.replace('\\', '/')).split('/')[-1].split('.jpg')[0])
                if product is None:
                    continue
                products.append(product)

            result_set['products'] = products
            print(result_set)
            return JsonResponse({'success': True, 'result': result_set, 'message': '성공!'})

        except Exception as exp:
            return JsonResponse({'success': False, 'message': exp})

    else:
        return JsonResponse({'success': False, 'message': '이 method는 POST 만 지원합니다.'})


@csrf_exempt
def upload_image(request, p_id):
    if request.method == 'POST':
        l_id = request.POST.get('l_id')
        user_id = request.POST.get('user_id')
        file = request.FILES.get('image')

        if not l_id or not user_id:
            return JsonResponse({'success': False, 'result': None, 'message': 'l_id, user_id는 필수입니다.'})

        try:
            user = User.objects.get(username=user_id)
        except Exception as e:
            return JSONResponse({'success': False, 'result': 'user_id를 확인 해주세요', 'message': str(e)})

        try:
            project = Project.objects.get(id=p_id, user=user)
        except Exception as e:
            return JsonResponse({'success': False, 'result': 'p_id를 확인해주세요.', 'message': str(e)})

        try:
            label = Label.objects.get(project=project, id=l_id)
        except Exception as e:
            return JsonResponse({'success': False, 'result': 'l_id를 확인해주세요.', 'message': str(e)})

        if not file:
            return JsonResponse({'success': False, 'message': '파일은 필수 입니다.'})

        if not allowed_file(file.name):
            return JsonResponse({'success': False, 'message': '파일은 형식을 확인해주세요'})
        save_file(file=file, label=l_id, project=p_id)

        return JsonResponse({'success': True, 'message': '성공!'})

    else:
        return JsonResponse({'success': False, 'message': '이 method는 POST 만 지원합니다.'})


@csrf_exempt
def recommend_product(request):
    if request.method == 'GET':
        try:
            page = request.GET.get('page')
            result_set = dict()
            if page is None:
                page = 0
            try:
                page = int(page)
                result_set['products'] = info_all[page:page+20]
                if page > len(info_all):
                    return JsonResponse({'success': False, 'message': 'database 초과'})
            except Exception as exp:
                return JsonResponse({'success': False, 'message': exp})
            return JsonResponse({'success': True, 'result': result_set, 'message': '성공!'})

        except Exception as exp:
            return JsonResponse({'success': False, 'message': exp})

    else:
        return JsonResponse({'success': False, 'message': '이 method는 POST 만 지원합니다.'})


@csrf_exempt
def train(request, p_id):
    logger.debug(request)
    try:
        project = Project.objects.get(id=p_id)
        if not project.is_changed:
            return JsonResponse({'success': True, 'result': 2, 'message': '이미 학습된 모델입니다.'})
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
        res = trainer.do_train_with_GPU(gpu_list=['/cpu:0'])
        if res is False:
            return JsonResponse({'success': True, 'result': -1, 'message': 'data가 부족합니다.'})
        vector_actual_path = trainer.vectorize_with_GPU(gpu_list=['/cpu:0'])
        save_vec2list(vector_actual_path=vector_actual_path)
        project.model = str(project.id) + '_' + str(int(time.mktime(datetime.datetime.now().timetuple())))
        project.is_changed = False
        project.save()
        remove_dir_tree(check_point_path)
        remove_dir_tree(summaries_dir)
        return JsonResponse({'success': True, 'result': 0, 'message': '성공!'})

    except Exception as exp:
        logger.exception(exp)
        return redirect('root')


@csrf_exempt
def search(request, p_id):
    logger.debug(request)
    try:
        if request.method == 'POST':
            file = request.FILES.get('image')
            if file is None:
                return JsonResponse({'success': False, 'result': None, 'message': '파일은 필수 입니다.'})

            if img_allowed_file(str(file)):
                img_path = 'media/api_upload/' + file.name
                save_file(file=file, img_path=img_path)
            else:
                return JsonResponse({'success': False, 'result': None, 'message': '파일은 형식을 확인해주세요'})

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
            return JsonResponse({'success': True, 'result': iv4_images, 'message': '성공'})

    except Exception as exp:
        logger.exception(exp)
        return redirect('root')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_FORMAT


def save_file(file, label=None, project=None, img_path=None):
    if img_path:
        fd = open(os.path.join(BASE_DIR, img_path), 'wb')
        for chunk in file.chunks():
            fd.write(chunk)
        fd.close()
        return
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

            for filename in os.listdir(dir_path):
                idx = 0
                ext = os.path.splitext(filename)[-1]
                if ext not in ['.jpg', '.jpeg', '.JPEG', '.JPG']:
                    os.remove(os.path.join(dir_path, filename))
                if hangul.findall(filename) is not []:
                    rename = re.sub('[^0-9a-zA-Z]', '', os.path.splitext(filename)[0]) + str(random.randint(0, 10000000))
                    os.rename(os.path.join(BASE_DIR, dir_path + '/' + filename), os.path.join(BASE_DIR, dir_path + '/' + rename + ext))
                    idx += 1

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

            for filename in os.listdir(dir_path):
                ext = os.path.splitext(filename)[-1]
                if ext not in ['.jpg', '.jpeg', '.JPEG', '.JPG']:
                    os.remove(os.path.join(dir_path, filename))
                if hangul.findall(filename) is not []:
                    rename = re.sub('[^0-9a-zA-Z]', '', os.path.splitext(filename)[0]) + str(random.randint(0, 10000000))
                    os.rename(os.path.join(BASE_DIR, dir_path + '/' + filename), os.path.join(BASE_DIR, dir_path + '/' + rename + ext))

            return True
        except Exception as e:
            print('ZIP error : ', e)
            return False
    else:
        ext = os.path.splitext(filename)[-1]
        if ext not in ['.jpg', '.jpeg', '.JPEG', '.JPG']:
            os.remove(os.path.join(dir_path, filename))
        if hangul.findall(filename) is not []:
            rename = re.sub('[^0-9a-zA-Z]', '', os.path.splitext(filename)[0]) + str(random.randint(0, 10000000))
            os.rename(os.path.join(BASE_DIR, dir_path + '/' + filename),
                      os.path.join(BASE_DIR, dir_path + '/' + rename + ext))
            return os.path.join(BASE_DIR, dir_path + '/' + rename + ext)
        return os.path.join(dir_path, filename)


def remove_dir_tree(remove_dir):
    try:
        shutil.rmtree(remove_dir, ignore_errors=False, onerror=remove_readonly)
    except PermissionError as e:
        print("[Delete Error] %s - %s." % (e.filename, e.strerror))


def remove_readonly(func, path):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def img_allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in IMG_ALLOWED_FORMAT


class JSONResponse(HttpResponse):
    def __init__(self, data, **kwargs):
        content = JSONRenderer().render(data)
        kwargs['content_type'] = 'application/json'
        super(JSONResponse, self).__init__(content, **kwargs)
