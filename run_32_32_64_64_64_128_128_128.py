import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
import tvm
from tvm.contrib import graph_executor

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)
def eval_tvm(path, model, img_size=108, threshold_range=[0.5]):
    trans = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    dataset = ImageFolder(root=path, transform=trans)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, num_workers=8, shuffle=False, pin_memory=True, drop_last=False)
    pre = []
    tar = []
    correct = 0
    for s, (im,target) in enumerate(loader):
        im = im.numpy()
        target = target.numpy()
        data_tvm = tvm.nd.array(im)
        module.set_input("data", data_tvm)
        module.run()
        logit = module.get_output(0).numpy()
        pred = np.argmax(logit,1)
        correct += (pred==target).sum()
        pre.append(pred)
        tar.append(target)
    tar = np.concatenate(tar)
    acc = correct/len(tar)
    return acc

target = tvm.target.Target("llvm -mcpu=core-avx2")
dev = tvm.device(str(target), 0)
dtype = "float32"

model = '32_32_64_64_64_128_128_128.so'
# Load saved .so file
lib = tvm.runtime.load_module(model)
module = graph_executor.GraphModule(lib["default"](dev))

# TODO preprocess real image data
input_shape = (1, 3, 108, 108)
#input_shape = (108, 108, 3)
data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
module.set_input("data", data_tvm)
module.run()
n_outputs = module.get_num_outputs()
# get output 
out = module.get_output(0)
print("Evaluate inference time cost...")
line = str(module.benchmark(dev, repeat=3, min_repeat_ms=500))
print(line)

# load test set
test_dir = '/scratch/general/nfs1/u1320844/dataset/corn_224/images/val'
print('Evaluate accuracy...')
acc_test = eval_tvm(test_dir, model, 108)
print(f'Accuracy on test set: {acc_test*100:.2f}%')