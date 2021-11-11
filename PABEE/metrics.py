import importlib
import time
import timeit

import numpy as np
from memory_profiler import memory_usage
from thop import profile as thop_profile
from torch.utils import benchmark


class Benchmark(object):
    def __init__(
        self, model, input_constructor, num_threads=1, use_cuda=False, device_idx=0
    ):
        self.model = model
        self.input_constructor = input_constructor

        self.num_threads = num_threads
        self.use_cuda = use_cuda
        self.device_idx = device_idx

    def get_wallclock(self, iters=100):
        timer = benchmark.Timer(
            stmt="model(input_tensor)",
            setup="input_tensor=input_constructor()",
            num_threads=self.num_threads,
            globals={"model": self.model, "input_constructor": self.input_constructor},
        )
        wallclock_mean = timer.timeit(iters).mean
        return wallclock_mean

    def get_memory(self):
        if self.use_cuda and importlib.util.find_spec("py3nvml"):
            from py3nvml import py3nvml as nvml

            nvml.nvmlInit()
            _ = self.model(self.input_constructor())
            handle = nvml.nvmlDeviceGetHandleByIndex(self.device_idx)
            meminfo = nvml.nvmlDeviceGetMemoryInfo(handle)
            memory_bytes = meminfo.used
        else:
            memory_bytes = memory_usage(
                (self.model, self.input_constructor().unsqueeze(0))
            )
        return memory_bytes

    def get_flops_count(self):
        macs, _ = thop_profile(self.model, (self.input_constructor(),), verbose=False)
        return macs

    def get_param_count(self, trainable_only=False):
        if trainable_only:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.model.parameters())
