import os
import tensorflow as tf
import pickle
import timeit
import logging
import heapq
import json
from .models import Label

from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from image_searcher.settings import BASE_DIR
from image.engine.utils.configs import ARGS
from image.engine.utils.vector_file_handler import save_vec2list
from image.engine.Incept_v4_Trainer import Incept_v4_Trainer
from image.engine.utils import configs
from image.engine.utils.ops import get_similarity_func

logging.basicConfig(
    format="[%(name)s][%(asctime)s] %(message)s",
    handlers=[logging.StreamHandler()],
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)
similarity_func = get_similarity_func()
ALLOWED_FORMAT = ['png', 'jpg', 'jpeg', 'gif', 'JPG', 'PNG']
args = ARGS()
IV4_vec2list_path = os.path.join(BASE_DIR, 'image/engine/vectors/vectors_i4_app/vec2list.pickle')

# IV4_vec2list_path = os.path.join(BASE_DIR, 'image/engine/vectors/vectors_i4_app/vec2list.pickle')
#
# with tf.gfile.FastGFile(os.path.join(BASE_DIR, configs.output_graph), 'rb') as fp:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(fp.read())
#     tf.import_graph_def(graph_def, name='')
# config = tf.ConfigProto(allow_soft_placement=True)
# iv4_sess = tf.Session(config=config)
# iv4_bottleneck = iv4_sess.graph.get_tensor_by_name('input/BottleneckInputPlaceholder:0')
#
# with open(IV4_vec2list_path, 'rb') as handle:
#     iv4_vector_list = pickle.load(handle)


@csrf_exempt
def train(request):
    logger.debug(request)
    try:
        trainer = Incept_v4_Trainer(image_dir=args.image_dir, output_graph=args.output_graph,
                                    output_labels=args.output_labels, vector_path=args.vector_path,
                                    summaries_dir=args.summaries_dir,
                                    how_many_training_steps=args.how_many_training_steps,
                                    learning_rate=args.learning_rate, testing_percentage=args.testing_percentage,
                                    eval_step_interval=args.eval_step_interval,
                                    train_batch_size=args.train_batch_size, test_batch_size=args.test_batch_size,
                                    validation_batch_size=args.validation_batch_size,
                                    print_misclassified_test_images=args.print_misclassified_test_images,
                                    model_dir=args.model_dir,
                                    bottleneck_dir=args.bottleneck_dir, final_tensor_name=args.final_tensor_name,
                                    flip_left_right=args.flip_left_right, random_crop=args.random_crop,
                                    random_scale=args.random_scale, random_brightness=args.random_brightness,
                                    check_point_path=args.check_point_path, max_ckpts_to_keep=args.max_ckpts_to_keep,
                                    gpu_list=args.gpu_list, validation_percentage=args.validation_percentage)
        trainer.do_train_with_GPU(gpu_list=['/gpu:0'])
        vector_actual_path = trainer.vectorize_with_GPU(gpu_list=['/gpu:0'])
        save_vec2list(vector_actual_path=vector_actual_path)

        return render(request, 'image/list_label.html')

    except Exception as exp:
        logger.exception(exp)
        return render({'result': False, 'reason': 'INTERNAL SERVER ERROR'})


@csrf_exempt
def predict(request):
    logger.debug(request)
    try:
        if request.method == 'POST':
            start = timeit.default_timer()
            with tf.gfile.FastGFile(os.path.join(BASE_DIR, configs.output_graph), 'rb') as fp:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(fp.read())
                tf.import_graph_def(graph_def, name='')
            config = tf.ConfigProto(allow_soft_placement=True)
            iv4_sess = tf.Session(config=config)
            iv4_bottleneck = iv4_sess.graph.get_tensor_by_name('input/BottleneckInputPlaceholder:0')

            with open(IV4_vec2list_path, 'rb') as handle:
                iv4_vector_list = pickle.load(handle)

            file = request.FILES.get('image')
            if file is None:
                return render(request, 'image/display_prediction.html')
            if allowed_file(str(file)):
                img_path = save_file(file)
            else:
                return render(request, 'image/display_prediction.html')

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
                tmp['img'] = str('/static/' + result.replace('\\', '/'))
                tmp['label'] = str('/static/' + result.replace('\\', '/')).split('/')[-2]
                iv4_images.append(tmp)

            end = timeit.default_timer()
            print('Time to load : ', end - start)
            print('ICEPTION : ', iv4_images)
            return render(request, 'image/display_prediction.html', {'images': iv4_images})
    except Exception as exp:
        logger.exception(exp)
        return render({'result': False, 'reason': 'INTERNAL SERVER ERROR'})


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_FORMAT


def save_file(file):
    filename = file._get_name()
    fd = open('image/static/upload/%s' % str(filename), 'wb')
    for chunk in file.chunks():
        fd.write(chunk)
    fd.close()
    return 'image/static/upload/%s' % str(filename)
