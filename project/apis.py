import os
import tensorflow as tf
import pickle
import timeit
import logging
import heapq
import json

from .engine.utils.database import ImageManager
from .engine.utils import configs
from .engine.utils.ops import get_similarity_func

from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse

logging.basicConfig(
    format="[%(name)s][%(asctime)s] %(message)s",
    handlers=[logging.StreamHandler()],
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

ALLOWED_FORMAT = ['png', 'jpg', 'jpeg', 'gif', 'JPG', 'PNG', 'JPEG']
IV4_vec2list_path = 'engine/vectors/vectors_i4_app/vec2list.pickle'
UPLOAD_FOLDER = 'static/upload/'

similarity_func = get_similarity_func()

image_db = ImageManager()
info_all = image_db.retrieve_info_all()
with tf.gfile.FastGFile(configs.output_graph, 'rb') as fp:
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
            data = json.loads(request.body.decode('utf-8'))
            logger.debug("INPUT %s", data)
            result_set = dict()

            # Get Variable for Meta data Info
            brand = data.get('brand')
            class_type = data.get('class_type')

            if class_type is not None:
                if not os.path.isdir(UPLOAD_FOLDER + class_type):
                    os.mkdir(UPLOAD_FOLDER + class_type)
            image_dir = UPLOAD_FOLDER + class_type
            file = request.FILES.get('image')

            if not file:
                return JsonResponse({'success': False, 'reason': '파일은 필수 입니다.'})
            img_path = image_dir + '/' + file.filename
            if not allowed_file(file.filename):
                return JsonResponse({'success': False, 'reason': '파일은 형식을 확인해주세요'})
            file.save(img_path)

            # For inception v4 Model
            iv4_img_list = {}
            iv4_image = tf.gfile.FastGFile(img_path, 'rb').read()
            iv4_image_vector = iv4_sess.run(iv4_bottleneck, {'DecodeJpeg/contents:0': iv4_image})
            labels = [line.rstrip() for line in tf.gfile.GFile('graph/output_labels.txt')]
            prediction = iv4_sess.run(logits, {'DecodeJpeg/contents:0': iv4_image})
            s_label = heapq.nlargest(3, range(len(prediction[0])), prediction[0].__getitem__)
            s_label = [labels[idx] for idx in s_label]
            selected_list = [v for v in iv4_vector_list if v[0].split('/')[0].split('\\')[1] in s_label]

            for vec in selected_list:
                dist = similarity_func(iv4_image_vector, vec[1])
                iv4_img_list[vec[0]] = dist

            keys_sorted = heapq.nsmallest(5, iv4_img_list, key=iv4_img_list.get)

            products = []
            for result in keys_sorted:
                product = image_db.retrieve_info_by_PRODUCT_CD(PRODUCT_CD=str('/' + result.replace('\\', '/')).split('/')[-1].split('.jpg')[0])
                if product is None:
                    continue
                products.append(product)

            result_set['products'] = products
            return JsonResponse({'success': True, 'result': result_set, 'message': ''})

        except Exception as exp:
            return JsonResponse({'success': False, 'message': exp})

    else:
        return JsonResponse({'success': False, 'message': '이 method는 POST 만 지원합니다.'})


@csrf_exempt
def upload_image(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            logger.debug("INPUT %s", data)

            # Get Variable for Meta data Info
            brand = data.get('brand')
            class_type = data.get('class_type')
            if not os.path.isdir(UPLOAD_FOLDER + class_type):
                os.mkdir(UPLOAD_FOLDER + class_type)
            image_dir = UPLOAD_FOLDER + class_type
            file = request.files.get('image')
            if file is None:
                return JsonResponse({'success': False, 'message': 'IMAGE는 필수 입니다.'})

            img_path = image_dir + '/' + file.filename
            if allowed_file(img_path):
                file.save(img_path)
                return JsonResponse({'success': True, 'message': '성공!'})
            else:
                return JsonResponse({'success': False, 'message': '[%s] 파일 형식 확인해 주세요' % file.filename})

        except Exception as exp:
            return JsonResponse({'success': False, 'message': exp})

    else:
        return JsonResponse({'success': False, 'message': '이 method는 POST 만 지원합니다.'})


@csrf_exempt
def recommend_product(request):
    if request.method == 'GET':
        try:
            result_set = dict()
            page = request.args.get('page')
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


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_FORMAT

