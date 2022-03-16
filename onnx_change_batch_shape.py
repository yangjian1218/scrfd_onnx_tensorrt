# -*- coding: utf-8 -*-
"""
@File : onnx_change_batch
@Description: scrfd人脸检测,原始onnx规定了batch=1,故进行改为动态batch
@Author: Yang Jian
@Contact: lian01110@outlook.com
@Time: 2022/2/25 10:31
@IDE: PYTHON
@REFERENCE: https://github.com/yangjian1218
"""

import onnx
change_batch = True  # 修改batch维度
change_shape = False  # 修改shape
onnx_path = 'models/scrfd/scrfd_2.5g_bnkps_shape640x640.onnx'
split_index = onnx_path.rfind(".")
# output_path =onnx_path[:split_index] + "n.onnx"
output_path ='models/scrfd/scrfd_2.5g_1kps_shape640x640.onnx'
model = onnx.load(onnx_path)
graph = model.graph
print(graph.input[0].type.tensor_type.shape)

# print(graph.node[0].attribute[1].i)
if change_batch:
    graph.input[0].type.tensor_type.shape.dim[0].dim_param = '1'
if change_shape:
    graph.input[0].type.tensor_type.shape.dim[2].dim_param = '?'
    graph.input[0].type.tensor_type.shape.dim[3].dim_param = '?'

# print(graph.input[0].type.tensor_type.shape.dim[0])
# graph.input[0].type.tensor_type.shape.dim[0].dim_value = 'None'
# graph.node[0].attribute[1].i = graph.input[0].type.tensor_type.shape.dim[0].dim_param
# graph.node[0].attribute[1].i = 2

onnx.save(model, output_path)
# try:
#     onnx.checker.check_model(output_path,full_check=True)
#     #onnx_model可是是加载后的模型,也可以是模型地址
# except onnx.checker.ValidationError as e:
#     print("model is invalid: %s"%(e))