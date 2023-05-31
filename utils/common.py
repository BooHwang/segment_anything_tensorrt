#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File      : common.py
@Time      : 2023/04/20 16:30:58
@Author    : Huang Bo
@Contact   : cenahwang0304@gmail.com
@Desc      : None
'''

'''
TensorRT common file
'''

import tensorrt as trt
import numpy as np
import os
import pycuda.driver as cuda
# import pycuda.autoinit      # if you want use other diff gpu rather gpu 0, you should commit this line

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()
    
class TrtModel():
    def __init__(self, engine_path, gpu_id=0, max_batch_size=1):
        print("TensorRT version: %s" % trt.__version__)
        cuda.init()
        self.cfx = cuda.Device(gpu_id).make_context()
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.context = self.engine.create_execution_context()
        self.max_batch_size = max_batch_size
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()

    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")             
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine
    
    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in self.engine:
            if binding in  ["point_coords", "point_labels"]:
                size = abs(trt.volume(self.engine.get_binding_shape(binding))) * self.max_batch_size
            else:
                size = abs(trt.volume(self.engine.get_binding_shape(binding))) 
            # print(f"binding: {binding}, size: {size}")
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream
    
    def __call__(self, inf_in_list, binding_shape_map=None):
        self.cfx.push()
        
        if binding_shape_map:
            self.context.set_optimization_profile_async
            for binding_name, shape in binding_shape_map.items():
                binding_idx = self.engine[binding_name]
                # print(f"binding_name: {binding_name}, binding_idx: {binding_idx}, shape: {shape}")
                self.context.set_binding_shape(binding_idx, shape)
        
        for i in range(len(self.inputs)):
            self.inputs[i].host = inf_in_list[i]
            cuda.memcpy_htod_async(self.inputs[i].device, self.inputs[i].host, self.stream)
        
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        for i in range(len(self.outputs)):
            cuda.memcpy_dtoh_async(self.outputs[i].host, self.outputs[i].device, self.stream)
        
        self.stream.synchronize()
        self.cfx.pop()
        return [out.host.copy() for out in self.outputs]
    
    def __del__(self):
        self.cfx.pop()
        del self.cfx