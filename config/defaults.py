from yacs.config import CfgNode as CN
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

_C = CN()
_C.PATH = CN()
_C.MODEL = CN()
_C.DEVICE = CN()
_C.DATA = CN()

_C.DEVICE.GPU = 0 # <gpu_id>
_C.DEVICE.CUDA = True # use gpu or not

_C.PATH.TRAIN_SET = "C:\\Users\\user\\Desktop\\ML\\HW\\HW02\\AIMango_sample (1)\\train\\" # <path_to_trainset>
_C.PATH.TEST_SET = "C:\\Users\\user\\Desktop\\ML\\HW\\HW02\\AIMango_sample (1)\\test\\" # <path_to_testset>

_C.MODEL.OUTPUT_PATH = "C:\\Users\\user\\Desktop\\hw2\\" # <weight_output_path>
_C.MODEL.LR = 0.01 # <learning_rate>
_C.MODEL.EPOCH = 100 # <train_epochs>

# -----------------------------------------------
# normalization parameters(suggestion)

_C.DATA.PIXEL_MEAN = [0.485, 0.456, 0.406] 
_C.DATA.PIXEL_STD = [0.229, 0.224, 0.225]

# -----------------------------------------------

_C.DATA.RESIZE = [720, 720] # picture size after resizing
_C.DATA.NUM_WORKERS = 0 # use how many processors
_C.DATA.TRAIN_BATCH_SIZE = 32 # <train_batch_size>
_C.DATA.TEST_BATCH_SIZE = 16 # <test_batch_size>
_C.DATA.VALIDATION_SIZE = 0.1

_C.merge_from_file(os.path.join(BASE_PATH, "config.yml"))