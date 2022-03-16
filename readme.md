# scrfd的tensorrt-onnxparser推理

# 一 、环境:

​	centos7

​	tensorrt8.2.1.8

​	onnx==1.9.0

​	cudnn==8.0.2.39

- 提示安装的版本很有关系,一开始安装的是tensorrt7.0, 在创建引擎时报错,

- onnx也推荐用tensorrt官网推荐的1.9.0.

- cudnn必须也要用8系列的. 由于系统还用onnxruntime-gpu ==1.6.0,故cudnn就采用cudnnn8.0.2.39. 这样可以兼容,只是在创建engine时会有警报,cudnn版不对. 推荐使用的是cudnn版本为8.2.1.32.

  ```shell
  [03/15/2022-15:03:37] [TRT] [W] TensorRT was linked against cuBLAS/cuBLAS LT 10.2.3 but loaded cuBLAS/cuBLAS LT 10.2.2   
  [03/15/2022-15:03:37] [TRT] [W] TensorRT was linked against cuDNN 8.2.1 but loaded cuDNN 8.0.2
  ```

  

# 二、pth转onnx

git clone insightface项目:[deepinsight/insightface: State-of-the-art 2D and 3D Face Analysis Project (github.com)](https://github.com/deepinsight/insightface)

scrfd在 ./detection/scrfd中. 

修改 ./detection/scrfd/modesl/hense_heads/scrfd_head.py中333-335行代码

```python
# 原代码
# cls_score = cls_score.permute(2, 3, 0, 1).reshape(-1, self.cls_out_channels).sigmoid()
# bbox_pred = bbox_pred.permute(2, 3, 0, 1).reshape(-1, 4)
# kps_pred = kps_pred.permute(2,3,0,1).reshape(-1, 10)

#todo 上面三行代码修改如下,去除-1
cls_score_shape = cls_score.shape
bbox_pred_shape = bbox_pred.shape
kps_pred_shape = kps_pred.shape
print("cls_score_shape",cls_score_shape," bbox_pred_shape:",bbox_pred_shape," kps_pred_shape:",kps_pred_shape)

#todo 如果要batch,用此三行
# cls_score = cls_score.permute(2, 3, 0, 1).reshape(int(cls_score_shape[0]),int(2*cls_score_shape[2]*cls_score_shape[3]), self.cls_out_channels).sigmoid()
# bbox_pred = bbox_pred.permute(2, 3, 0, 1).reshape(int(bbox_pred_shape[0]),int(2*bbox_pred_shape[2]*bbox_pred_shape[3]), 4)
# kps_pred = kps_pred.permute(2,3,0,1).reshape(int(kps_pred_shape[0]),int(2*kps_pred_shape[2]*kps_pred_shape[3]), 10)            
#todo如果直接输入batch=1,取消batch维度,用此三行
cls_score = cls_score.permute(2, 3, 0, 1).reshape(int(2*cls_score_shape[2]*cls_score_shape[3]), self.cls_out_channels).sigmoid()
bbox_pred = bbox_pred.permute(2, 3, 0, 1).reshape(int(2*bbox_pred_shape[2]*bbox_pred_shape[3]), 4)
kps_pred = kps_pred.permute(2,3,0,1).reshape(int(2*kps_pred_shape[2]*kps_pred_shape[3]), 10)  
```

**原因是如果按源代码转的onnx, 即使我使用固定batch和shape,但是输出的维度还是有?, 而不是固定的值.因为原代码用了-1来reshape,那么转onnx后并不认识. 另外记住用reshape里面的值,最好加上int**

然后使用detection/scrfd/tools/scrfd2onnx.py运行, 因为使用tensorrt, 我之前试过动态batch,失败,就放弃了,

# 三、onnx转trt

  代码如下：

```python
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

                # 解析onnx文件,填充计算图
                if not os.path.exists(onnx_file_path):
                    quit("ONNX file {} not found!".format(onnx_file_path))
                print('loading onnx file from path : {}...'.format(onnx_file_path))
                success = parser.parse_from_file(onnx_file_path)
                if not success:
                    print('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None

                # 重新设置网络输入batch
                # network = set_net_batch(network, max_batch_size)
                # 填充计算图完成后, 则使用builder从计算图中创建CudaEngine
                print("Building an engine from file: {},this may take a while ...".format(onnx_file_path))
                builder.max_batch_size = max_batch_size  # 执行时最大可以使用的batchsize

                config = builder.create_builder_config()
                # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)  # 1 MiB
                config.max_workspace_size = max_batch_size << 30  # 预先分配的工作空间大小,即ICudaEngine执行时GPU最大需要的空间
                config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
                if fp16_mode:
                    config.set_flag(trt.BuilderFlag.FP16)
                if int8_mode:
                    config.set_flag(trt.BuilderFlag.INT8)
                # profile = builder.create_optimization_profile()
                # profile.set_shape(
                #     'input',  # input tensor name
                #     (1, 3, 640, 640),  # min shape
                #     (max_batch_size, 3, 640, 640),  # opt shape
                #     (max_batch_size, 3, 640, 640))  # max shape
                # config.add_optimization_profile(profile)

                engine = builder.build_engine(network，config)  # 没有序列化,<class 'tensorrt.tensorrt.ICudaEngine'>
                #engine = builder.build_serialized_network(network, config)  # 已经序列化,类型为:<class 'tensorrt.tensorrt.IHostMemory'
                if engine is None:
                    print("build trt engine fail")
                else:
                    print("Completed creating Engine")
                    if save_engine:
                        # 保存engine供以后直接反序列化使用
                        with open(engine_file_path, 'wb') as f:
                            f.write(engine.serialize())  # 序列化
                            # f.write(engine)  
                        print("engine have saved in {}".format(engine_file_path))
                return engine
```

然后会生产trt文件， 后缀起trt或者engine都可以没有区别。 

**如果采用engine = builder.build_engine(network，config)  # 没有序列化 那么保存的时候需要f.write(engine.serialize())**

**如果采用engine = builder.build_serialized_network(network, config)  # 已经序列化, 那么保存的时候直接f.write(engine)**

如果代码后续直接使用返回的engine,建议用未序列化的,因为序列化返回来的engine,类型不对会报错:

```python
    self.context = self.engine.create_execution_context()
AttributeError: 'tensorrt.tensorrt.IHostMemory' object has no attribute 'create_execution_context'
```

也可以使用trtexec 工具来实现onnx转trt/engine模型.例如:

```shell
trtexec --onnx=models/scrfd/scrfd_2.5g_kps.onnx --saveEngine=models/scrfd/scrfd_2.5g_kps.engine --workspace=256 --explicitBatch --minShapes=input.1:1x3x112x112 --optShapes=input.1:1x3x640x640 --maxShapes=input.1:1x3x1280x1280 --fp16
```

注意:

- 虽然这里有--explicitBatch参数,但是并不支持动态batch
- 输入名称可以先打开onnx模型查看,这里一定要一直,如我这里为"input.1"
- maxShapes会影响trt模型大小
- --fp16 可以缩小模型,加快推理速度,但精度会有一定的损失

# 三、后处理

由于tensorrt返回的结果顺序跟原始onnx不同,而且结果是连续数据, 但人脸检测的结果是是二维的,所以需要reshape.

原始onnx的输出跟shape为:

```
['score_8', 'score_16', 'score_32', 'bbox_8',  'bbox_16', 'bbox_32',  'kps_8',   'kps_16',    'kps_32']
(12800,1),  (3200, 1),  (800, 1),  (12800, 4), (3200, 4),  (800, 4), (12800, 10), (3200, 10), (800, 10)
```

trt输出和shape如下:

```shell
score_8     bbox_8     kps_8     score_16     bbox_16     kps_16	  score_32     bbox_32     kps_32 
(12800,)   (51200,)   (128000,)  (3200,)     (12800,)    (32000,)     (800,)        (3200,)    (8000,)
```

在输出结果后面加上:

```python
trt_outs = do_inference(context=self.context, bindings=self.bindings, inputs=self.inputs,
                                outputs=self.outputs, stream=self.stream)
net_outs = []
reshape_list = [1, 4, 10]  # reshape 第二位的数
k = 0
i = 0
while len(net_outs) < 9:     
    out = trt_outs[i]
    out = out.reshape(out.shape[0]//reshape_list[k],reshape_list[k])
    net_outs.append(out)
    i += 3
    if i >8:
        k += 1
        i = k     
```

# 四、与onnxruntime-gpu比较

1张图片,推理1000次，scrfd.py跟scrfd_trt.py

| 框架            | 总耗时     | time/pcs |
| --------------- | ---------- | -------- |
| onnxruntime-gpu | 11030.383  | 11.0ms   |
| tensorrt8       | 9103.422ms | 9.1ms    |



