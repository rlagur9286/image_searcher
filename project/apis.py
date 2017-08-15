import os
import tensorflow as tf
import pickle
import logging
import heapq
import datetime
import time
import stat
import shutil
import timeit

from .models import Project
from .models import Label
from .engine.utils.database import ImageManager
from .engine.utils import configs
from .engine.utils.ops import get_similarity_func

from image_searcher.settings import BASE_DIR

from project.engine.utils.vector_file_handler import save_vec2list
from project.engine.Incept_v4_Trainer import Incept_v4_Trainer
from project.engine.utils.configs import ARGS

from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.shortcuts import redirect
from django.shortcuts import render
from django.contrib import messages

logging.basicConfig(
    format="[%(name)s][%(asctime)s] %(message)s",
    handlers=[logging.StreamHandler()],
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)
args = ARGS()

ALLOWED_FORMAT = ['png', 'jpg', 'jpeg', 'gif', 'JPG', 'PNG', 'JPEG']
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

with open(IV4_vec2list_path, 'rb') as handle:
    iv4_vector_list = pickle.load(handle)


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
                return JsonResponse({'success': False, 'reason': '파일은 필수 입니다.'})
            img_path = image_dir + file.name
            if not allowed_file(file.name):
                return JsonResponse({'success': False, 'reason': '파일은 형식을 확인해주세요'})
            save_file(file=file, img_path=img_path)

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
            print(result_set)
            return JsonResponse({'success': True, 'result': result_set, 'message': ''})

        except Exception as exp:
            return JsonResponse({'success': False, 'message': exp})

    else:
        return JsonResponse({'success': False, 'message': '이 method는 POST 만 지원합니다.'})


@csrf_exempt
def upload_image(request):
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
                return JsonResponse({'success': False, 'reason': '파일은 필수 입니다.'})
            img_path = image_dir + file.name
            if not allowed_file(file.name):
                return JsonResponse({'success': False, 'reason': '파일은 형식을 확인해주세요'})
            save_file(file=file, img_path=img_path)

            return JsonResponse({'success': True, 'message': '성공!'})

        except Exception as exp:
            return JsonResponse({'success': False, 'message': exp})

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
            return JsonResponse({'success': True, 'result': 2})
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
            return JsonResponse({'success': True, 'result': -1})
        vector_actual_path = trainer.vectorize_with_GPU(gpu_list=['/cpu:0'])
        save_vec2list(vector_actual_path=vector_actual_path)
        project.model = str(project.id) + '_' + str(int(time.mktime(datetime.datetime.now().timetuple())))
        project.is_changed = False
        project.save()
        remove_dir_tree(check_point_path)
        remove_dir_tree(summaries_dir)
        return JsonResponse({'success': True, 'result': 0})

    except Exception as exp:
        logger.exception(exp)
        return redirect('root')


@csrf_exempt
def predict(request, p_id):
    logger.debug(request)
    try:
        if request.method == 'POST':
            start = timeit.default_timer()
            file = request.FILES.get('image')
            if file is None:
                return render(request, 'project/display_prediction.html', {'project': Project.objects.get(id=p_id)})

            if img_allowed_file(str(file)):
                img_path = 'media/upload/' + file.name
                save_file(file=file, img_path=img_path)
            else:
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

            end = timeit.default_timer()
            print('Time to load : ', end - start)
            print('ICEPTION : ', iv4_images)
            return render(request, 'project/display_prediction.html', {'images': iv4_images, 'project': Project.objects.get(id=p_id)})
    except Exception as exp:
        logger.exception(exp)
        return redirect('root')


@csrf_exempt
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
            save_file(file=file, img_path=img_path)

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
            return redirect('root')

    else:
        return JsonResponse({'success': False, 'message': '이 method는 POST 만 지원합니다.'})


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_FORMAT


def save_file(file=None, img_path=None):
    fd = open(os.path.join(BASE_DIR, img_path), 'wb')
    for chunk in file.chunks():
        fd.write(chunk)
    fd.close()


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
