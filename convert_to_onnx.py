import tensorflow as tf
import os
import tf2onnx
import onnxmltools
from onnxmltools.utils.float16_converter import convert_float_to_float16
import onnx
import onnx_graphsurgeon as gs
import numpy as np
import onnxruntime

## tensorflow saved_model.pb -> Onnx
'''
https://morioh.com/p/7be063227a64
--saved-model : input path
--output : output path
--inputs : 입력 Tensor의 이름과 shape. tensor이름:index 이 형태여야함
'''

os.system(
    'python -m tf2onnx.convert --saved-model C://Users/jsyoon/PycharmProjects/Tensorflow/venv/twinnet_org/RESULTS/B32_LR1e-04_EP30_DE30_DR1_P15_cw0.4/fin --output twinnet_testmodel.onnx  --inputs input_1:0[4,120,120,1],input_2:1[4,120,120,1] --opset 10 --verbose')

#중간 Tensor들의 shape을 알기위한 shape inference 기능
# os.system(
#     'python -m onnxruntime.tools.symbolic_shape_infer --input ./model.onnx --output ./cvt_model4.onnx --auto_merge')
