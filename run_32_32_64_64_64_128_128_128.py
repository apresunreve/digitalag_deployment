import numpy as np
import tvm
from tvm.contrib import graph_executor

target = tvm.target.Target("llvm -mcpu=core-avx2")
dev = tvm.device(str(target), 0)
dtype = "float32"

model = '32_32_64_64_64_128_128_128.so'
# Load saved .so file
lib = tvm.runtime.load_module(model)
module = graph_executor.GraphModule(lib["default"](dev))

# TODO preprocess real image data
input_shape = (224, 224, 3)
data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
module.set_input("data", data_tvm)

module.run()
n_outputs = module.get_num_outputs()
# get output classification
out = module.get_output(0)
print("Evaluate inference time cost...")
line = str(module.benchmark(dev, repeat=3, min_repeat_ms=500))
print(line)
