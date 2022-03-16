# -*- coding: utf-8 -*-
"""
@File : scrfd_trt
@Description: 描述信息
@Author: Yang Jian
@Contact: lian01110@outlook.com
@Time: 2022/3/14 10:16
@IDE: PYTHON
@REFERENCE: https://github.com/yangjian1218
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import threading
import cv2
import face_align
import os

def load_onnx(onnx_file_path):
    """Read the ONNX file."""
    with open(onnx_file_path, 'rb') as f:
        return f.read()


def set_net_batch(network, batch_size):
    """Set network input batch size.
    The ONNX file might have been generated with a different batch size,
    say, 64.
    """
    if trt.__version__[0] >= '7':
        shape = list(network.get_input(0).shape)
        shape[0] = batch_size
        network.get_input(0).shape = shape
    return


class HostDeviceMem(object):
    """Simple helper data class that's a little nicer to use than a 2-tuple."""

    def __init__(self, host_mem, device_mem):
        """
        host_mem: cpu memory
        device_mem: gpu memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return 'Host:\n' + str(self.host) + '\nDevice:\n' + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine, context):
    """[summary]

    Arguments:
        engine {[type]} -- [description]
        context {[type]} -- tensorRT7以后改用这个读取一些数据

    Returns:
        [type] -- [description]
    """
    inputs, outputs, bindings = [], [], []
    assert len(engine) == 10 and engine[0] == 'input'
    # print("max_batch_size:", engine.max_batch_size)

    for i in range(len(engine)):
        binding = engine[i]
        print("bindging:",binding)
        dims = context.get_binding_shape(i)
        size = trt.volume(dims) * engine.max_batch_size  # volume 计算可地带变量的空间,指元素个数
        # size = trt.volume([1, 3, 112, 112]) * 2 if i == 0 else trt.volume([1, 512]) * 2
        # if dims[0] < 0:
        #     size *= -1
        dtype = trt.nptype(engine.get_binding_dtype(binding))  # get_binding_dytpe  获取binding的数据类型
        # allocate host and device buffers  host即内存  device即显存
        host_mem = cuda.pagelocked_empty(size, dtype)  # 创建锁页内存
        device_mem = cuda.mem_alloc(host_mem.nbytes)  # cuda分配空间
        # print(int(device_men))    # binding在计算图中的缓冲地址
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings


def do_inference(context, bindings, inputs, outputs, stream, batch_dynamic=False):
    # htod ost to device把数据从cpu 移到GPU

    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference
    if batch_dynamic:
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
    else:
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # 当创建network时显式指定了batchsize,则使用execute_async_v2, 否则使用execute_async
    # 将预测结果从GPU 返回给CPU
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize同步 the stream
    stream.synchronize()
    # Return only the host outputs
    return [out.host for out in outputs]


def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]  # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]  # dito
    return e_x / div


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

def preprocess_data(image, input_size=(640, 640), swap_rb=False):
    if not isinstance(image, list):
        image = [image]  # 如果不是列表,则转为列表
    image = [cv2.resize(img, dsize=input_size, interpolation=cv2.INTER_AREA) for img in image]
    if swap_rb:
        image = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in image]
    image = np.transpose(image, (0, 3, 1, 2)).astype(np.float32)  # (b, channel, height, width)
    image = (image - 127.5) / 128
    return image

class SCRFD_trt(object):
    def __init__(self, max_batch_size=1, onnx_file_path="", engine_file_path="", fp16_mode=True, int8_mode=False, cuda_ctx=None, verbose=False):
        """

        :param max_batch_size: 最大batch,
        :param onnx_file_path: onnx模型地址
        :param engine_file_path: wts模型
        :param fp16_mode: 是否用fp16,用了速度加快,但精度会一定的下降
        :param int8_mode: 是否用int8,需要设置支持
        :param cuda_ctx:
        :param verbose:
        """
        if cuda_ctx:
            self.cuda_ctx = cuda.Device(cuda_ctx).make_context()
        self.trt_logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
        self.engine = self.get_engine(max_batch_size=max_batch_size, onnx_file_path=onnx_file_path,
                                      engine_file_path=engine_file_path, fp16_mode=fp16_mode, int8_mode=int8_mode, save_engine=True)
        assert self.engine.get_binding_dtype('input') == trt.tensorrt.DataType.FLOAT  # 输入应该为浮点型
        try:
            self.context = self.engine.create_execution_context()
            input_dims = self.context.get_binding_shape(0)  # 'input'
            # print("input_dims:", input_dims)
            self.stream = cuda.Stream()
            self.inputs, self.outputs, self.bindings = allocate_buffers(self.engine, self.context)
        except Exception as e:
            raise RuntimeError('fail to allocate CUDA resources') from e
        finally:
            if self.cuda_ctx:
                self.cuda_ctx.pop()
        self.input_shape = (input_dims[2], input_dims[3])
        self.center_cache = {}
        self.nms_thresh = 0.4
        self.det_thresh = 0.5
        self._init_vars()

    def _init_vars(self):
        self.input_mean = 127.5
        self.input_std = 128.0
        self.use_kps = False
        self._anchor_ratio = 1.0
        self._num_anchors = 1
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]  # 基本执行这个
        self._num_anchors = 2
        self.use_kps = True

    def get_engine(self, max_batch_size=1, onnx_file_path="", engine_file_path="", fp16_mode=True, int8_mode=False,
                   save_engine=False):
        """
        params max_batch_size:      预先指定大小好分配显存
        params onnx_file_path:      onnx文件路径
        params engine_file_path:    待保存的序列化的引擎文件路径
        params fp16_mode:           是否采用FP16,可以加速,但减精度
        paramsint8_mode:            是否采用int8,需要设备支持
        params save_engine:         是否保存引擎
        returns:                    ICudaEngine
        """
        # 如果已经存在序列化之后的引擎,则直接反序列化得到cudaEngine
        if os.path.exists(engine_file_path):
            print("Reading engine from file :{}".format(engine_file_path))
            with open(engine_file_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
                return runtime.deserialize_cuda_engine(f.read())  # 反序列化
        else:

            # 解析onnx文件,填充计算图
            if not os.path.exists(onnx_file_path):
                quit("ONNX file {} not found!".format(onnx_file_path))
            print('loading onnx file from path : {}...'.format(onnx_file_path))
            onnx_data = load_onnx(onnx_file_path)
            if onnx_data is None:
                return None
            # parse.parse_from_file(onnx_file_path)  # parse解析onnx文件的另一种方法

            # 由onnx创建cudaEngine
            # builder 创建一个计算图 INetworkDefinition
            explicit_batch = [] if trt.__version__[0] < '7' else [
                1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)]

            # In TensorRT7.0 the parser only supports full-dimensions mode,meanning that your network \
            # must be created with the explicitBatch flag set(即TensorRT7.0 不支持动态batch).

            with trt.Builder(self.trt_logger) as builder, \
                    builder.create_network(*explicit_batch) as network, \
                    trt.OnnxParser(network, self.trt_logger) as parser:  # 使用onnx的解析器绑定计算图,后续将通过解析填充计算图
                if int8_mode and not builder.platform_has_fast_int8:
                    raise RuntimeError('INT8 not supported on this platform')  # 判断是否支持int8
                if not parser.parse(onnx_data):
                    print('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None

                print(network.get_layer(network.num_layers - 1).get_output(0).shape)
                # network.mark_output(network.get_layer(network.num_layers-1).get_output(0))
                # 重新设置网络输入batch
                network = set_net_batch(network, max_batch_size)
                # 填充计算图完成后, 则使用builder从计算图中创建CudaEngine
                print("Building an engine from file: {},this may take a while ...".format(onnx_file_path))
                builder.max_batch_size = max_batch_size  # 执行时最大可以使用的batchsize
                if trt.__version__[0] < '7':  # older API: build_cuda_engine()
                    builder.fp16_mode = fp16_mode
                    builder.max_workspace_size = max_batch_size << 30
                    if int8_mode:
                        builder.int8_mode = int8_mode
                    engine = builder.build_cuda_engine(network)
                else:
                    config = builder.create_builder_config()
                    config.max_workspace_size = max_batch_size << 30  # 预先分配的工作空间大小,即ICudaEngine执行时GPU最大需要的空间
                    config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
                    if fp16_mode:
                        config.set_flag(trt.BuilderFlag.FP16)
                    if int8_mode:
                        config.set_flag(trt.BuilderFlag.INT8)
                    profile = builder.create_optimization_profile()
                    profile.set_shape(
                        'input',  # input tensor name
                        (1, 3, 640, 640),  # min shape
                        (max_batch_size, 3, 640, 640),  # opt shape
                        (max_batch_size, 3, 640, 640))  # max shape
                    config.add_optimization_profile(profile)

                    engine = builder.build_engine(network, config)  # 注意,这里的network是INetworkDefinition类型,即填充后的计算图
                if engine is None:
                    print("build trt engine fail")
                else:
                    print("Completed creating Engine")
                    if save_engine:
                        # 保存engine供以后直接反序列化使用
                        with open(engine_file_path, 'wb') as f:
                            f.write(engine.serialize())  # 序列化
                        print("engine have saved in {}".format(engine_file_path))
                return engine

    def prepare(self, ctx_id, **kwargs):
        if ctx_id < 0:
            self.session.set_providers(['CPUExecutionProvider'])
        # else:
        #     self.session.set_providers(['CUDAExecutionProvider'], [{'device_id': ctx_id}])
        nms_thresh = kwargs.get('nms_thresh', None)
        if nms_thresh is not None:
            self.nms_thresh = nms_thresh
        det_thresh = kwargs.get('det_thresh', None)
        if det_thresh is not None:
            self.det_thresh = det_thresh
        input_size = kwargs.get('input_size', None)
        if input_size is not None:
            if self.input_size is not None:
                print('warning: det_size is already set in scrfd model, ignore')
            else:
                self.input_size = input_size
        img_tmp = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
        img_tmp = np.asfarray(img_tmp, dtype="float32")
        blob = cv2.dnn.blobFromImage(img_tmp, 1.0 / self.input_std, input_size,
                                     (self.input_mean, self.input_mean, self.input_mean))
        self.inputs[0].host = np.ascontiguousarray(blob)  # 这一步非常重要,不然会报错ndarray不是连续数据
        if self.cuda_ctx:
            self.cuda_ctx.push()
        do_inference(context=self.context, bindings=self.bindings, inputs=self.inputs,
                     outputs=self.outputs, stream=self.stream)

    def init_det_threshold(self, det_threshold):
        """
        单独设置人脸检测阈值
        :param det_threshold: 人脸检测阈值
        :return:
        """
        self.det_thresh = det_threshold

    def forward(self, img, threshold=0.6, swap_rb=True):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(img.shape[0:2][::-1])
        # print('input_size:',input_size)
        # blob = cv2.dnn.blobFromImages([img], 1.0 / self.input_std, input_size,
        #                               (self.input_mean, self.input_mean, self.input_mean), swapRB=swap_rb)
        blob = preprocess_data(img, swap_rb=swap_rb)
        self.inputs[0].host = np.ascontiguousarray(blob)  # 这一步非常重要,不然会报错ndarray不是连续数据
        if self.cuda_ctx:
            self.cuda_ctx.push()
        net_outs = do_inference(context=self.context, bindings=self.bindings, inputs=self.inputs,
                                   outputs=self.outputs, stream=self.stream)
        if self.cuda_ctx:
            self.cuda_ctx.pop()

        # net_outs = self.session.run(self.output_names, {self.input_name: blob})
        # print("net_outs:::",net_outs[0])
        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc  # 3
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx]
            # print("scores:",scores)
            bbox_preds = net_outs[idx + fmc]
            bbox_preds = bbox_preds * stride
            if self.use_kps:
                kps_preds = net_outs[idx + fmc * 2] * stride
            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                # solution-3:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                # print(anchor_centers.shape)

                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers
            # print(anchor_centers.shape,bbox_preds.shape,scores.shape,kps_preds.shape)
            pos_inds = np.where(scores >= threshold)[0]
            # print("pos_inds:",pos_inds)
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                # kpss = kps_preds
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        # print("....:",bboxes_list)
        return scores_list, bboxes_list, kpss_list

    def detect(self, img, input_size=None, max_num=0, det_thresh=None, metric='default', swap_rb=True):
        """

        :param img: 原始图像
        :param input_size:  输入尺寸,元组或者列表
        :param max_num: 返回人脸数量, 如果为0,表示所有,
        :param det_thresh: 人脸检测阈值,
        :param metric: 排序方式,默认为面积+中心偏移, "max"为面积最大排序
        :param swap_rb: 是否进行r b通道转换, 如果传入的是bgr格式图片,则需要为True
        :return:
        """
        assert input_size is not None or self.input_size is not None
        input_size = self.input_size if input_size is None else input_size
        # resize方法选择,缩小选择cv2.INTER_AREA , 放大选择cv2.INTER_LINEAR
        resize_interpolation = cv2.INTER_AREA if img.shape[0] >= input_size[0] else cv2.INTER_LINEAR
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=resize_interpolation)
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img
        if det_thresh == None:
            det_thresh = self.det_thresh
        scores_list, bboxes_list, kpss_list = self.forward(det_img, det_thresh, swap_rb)
        # print("====",len(scores_list),len(bboxes_list),len(kpss_list))
        # print("scores_list:",scores_list)
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] -
                                              det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == 'max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0  # some extra weight on the centering
            bindex = np.argsort(
                values)[::-1]  # some extra weight on the centering
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        return det, kpss

    def get_align(self, image, kpss):
        """
        从图像中生成align后的人脸图像
        :param image: nparray, 原始图
        :param kpss:  人脸关键点坐标列表
        :return: aligned 人脸 112x112
        """
        aligns = []
        for pts in kpss:
            align = face_align.norm_crop(image, pts)  # 得到112x112的对齐图像
            aligns.append(align)
        return aligns

    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep


def main():
    max_batch_size = 1
    onnx_model_path = "models/scrfd/scrfd_2.5g_kps.onnx"
    model_name = onnx_model_path[:-5]
    fp16_mode = False   # 设置为True ,速度提高,但精度也会比原始onnx低
    int8_mode = False
    trt_engine_path = model_name + '_b{}_fp16{}_int8{}.trt'.format(max_batch_size, fp16_mode, int8_mode)
    img_path = "data/test2.jpg"
    img = cv2.imread(img_path)

    detector = SCRFD_trt(max_batch_size, onnx_model_path, trt_engine_path, fp16_mode, int8_mode)
    # detector.prepare(-1)
    # ta = datetime.datetime.now()
    # cycle = 100
    # for i in range(cycle):
    #     bboxes, kpss = detector.detect(img, input_size=(640, 640))  # 得到box跟关键点
    # # print("bboxes:",bboxes,"\nkpss:",kpss)
    # tb = datetime.datetime.now()
    # print('all cost:', (tb - ta).total_seconds() * 1000)
    # print(img_path, bboxes.shape)
    # if kpss is not None:
    #     print(kpss.shape)
    # # todo 画图
    # for i in range(bboxes.shape[0]):
    #     bbox = bboxes[i]
    #     x1, y1, x2, y2, score = bbox.astype(np.int)
    #     cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    #     if kpss is not None:
    #         kps = kpss[i]
    #         for kp in kps:
    #             kp = kp.astype(np.int)
    #             cv2.circle(img, tuple(kp), 1, (0, 0, 255), 2)
    # cv2.namedWindow("img", 2)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)

if __name__ == '__main__':
    main()