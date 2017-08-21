import argparse
# universal
METADATA_FILE_NAME = 'metadata.pkl'

# vector_file_handler
ATUO_VECTORS_FOLDER_NAME = 'vectors_auto_app'
I4_VECTORS_FOLDER_NAME = 'vectors_i4_app'
VECTORS_FILE_NAME = 'vectors.tfrecord'
TYPE_FILE_NAME = 'type.txt'

# autoencoder_training
IMAGE_INPUT_SIZE = 224
TENSORBOARD_NAME = 'tb'
AUTO_MODEL_CHECKPOINT_NAME = 'auto_model{}.ckpt'
I4_MODEL_CHECKPOINT_NAME = 'i4_model{}.ckpt'

# vectorize_pretrained
INCEPTION_IMAGE_SIZE = 299
VGG_IMAGE_SIZE = 224
BATCH_SIZE = 128
output_graph = 'project/engine/graph/'


class ARGS(object):
    def __init__(self):
        self.image_dir = 'media/images'
        self.output_graph = 'project/engine/graph'
        self.output_labels = 'project/engine/graph'
        self.summaries_dir = 'project/engine/tmp/retrain_logs'
        self.how_many_training_steps = 100
        self.learning_rate = 0.01
        self.testing_percentage = 20
        self.validation_percentage = 20
        self.eval_step_interval = 25
        self.train_batch_size = 32
        self.test_batch_size = -1
        self.validation_batch_size = 32
        self.print_misclassified_test_images = False
        self.model_dir = 'project/engine/models'
        self.bottleneck_dir = 'project/engine/bottleneck'
        self.final_tensor_name = 'final_result'
        self.flip_left_right = False
        self.random_crop = 0
        self.random_scale = 0
        self.random_brightness = 0
        self.check_point_path = 'project/engine/check_point'
        self.max_ckpts_to_keep = 5
        self.vector_path = 'project/engine/vectors'
        self.gpu_list = ['/gpu:0']
