#!/usr/bin/python3
import numpy as np
import torch
from torchvision.datasets import ImageFolder
#from torchvision import transforms
#import tvm
#from tvm.contrib import graph_executor
#
#MEAN = (0.5, 0.5, 0.5) 
#STD = (0.5, 0.5, 0.5)
#def eval_tvm(path, model, img_size=108, threshold_range=[0.5]):
#    trans = transforms.Compose([
#        transforms.Resize((img_size,img_size)),
#        transforms.ToTensor(),
#        transforms.Normalize(mean=MEAN, std=STD),
#    ])
#    dataset = ImageFolder(root=path, transform=trans)
#    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, num_workers=8, shuffle=False, pin_memory=True, drop_last=False)
#    pre = []
#    tar = []
#    correct = 0
#    for s, (im,target) in enumerate(loader):
#        im = im.numpy()
#        target = target.numpy()
#        data_tvm = tvm.nd.array(im)
#        module.set_input("data", data_tvm)
#        module.run()
#        logit = module.get_output(0).numpy()
#        pred = np.argmax(logit,1)
#        correct += (pred==target).sum()
#        pre.append(pred)
#        tar.append(target)
#    tar = np.concatenate(tar)
#    acc = correct/len(tar)
#    return acc
#
#target = tvm.target.Target("llvm -mcpu=core-avx2")
#dev = tvm.device(str(target), 0)
#dtype = "float32"
#
#model = '/opt/bitnami/apache/cgi-bin/32_32_64_64_64_128_128_128.so'
## Load saved .so file
#lib = tvm.runtime.load_module(model)
#module = graph_executor.GraphModule(lib["default"](dev))
#
## TODO preprocess real image data
#input_shape = (1, 3, 108, 108)
##input_shape = (108, 108, 3)
#data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
#module.set_input("data", data_tvm)
#module.run()
#n_outputs = module.get_num_outputs()
## get output 
#out = module.get_output(0)
#print("Evaluate inference time cost...")
#line = str(module.benchmark(dev, repeat=3, min_repeat_ms=500))
#print(line)
#
## load test set
#test_dir = '/opt/bitnami/apache/cgi-bin/val'
#print('Evaluate accuracy...')
#acc_test = eval_tvm(test_dir, model, 108)
#print(f'Accuracy on test set: {acc_test*100:.2f}%')
#
#res0 = f'<a>{line}</a>'
#res1 = f'<a>Accuracy on test set: {acc_test*100:.2f}% </a>'
res00 = '<a>Model 4_8_16_32_32_32_32_32</a><br>'
res0 = '<a>Execution time summary:</a><br>'
res1 = '<a> mean(ms): 1.2926</a><br>'
res2 = '<a> median(ms): 1.2840</a><br>'
res3 = '<a> max(ms): 1.3100</a><br>'
res4 = '<a> min(ms): 1.2839</a><br>'
res5 = '<a> std(ms): 0.0123</a><br>'
res6 = '<a>Accuracy on test set: 88.1%</a><br>'
#
#                             

html0 = '''
<html>
<head>
<title>DigitalAg Model Demo</title>

<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet"
    integrity="sha384-rbsA2VBKQhggwzxH7pPCaAqO46MgnOM80zW1RWuH61DGLwZJEdK2Kadq2F9CUG65" crossorigin="anonymous">
<link href='https://fonts.googleapis.com/css?family=Space Grotesk' rel='stylesheet'>
<link rel="stylesheet" type="text/css" href="http://149.165.155.188:2298/css/style.css">
  <div id="about" class="aboutContainer container-fluid">
    <div class="AboutInfo">
      <div class="row">
        <div class="col-12 offset-1">
          <div class="AboutInfoTitle">
            DefoNet Model Results<br>
          </div>
        </div>
      </div>
    </div>
    <div class="teamContainer container-fluid">
      <div class="row">
        <div class="col-2">
          <div id="ScoutingMapInterpretabilityUpload" class="container-fluid personInfoContainer">
            <div class="container-fluid personInfoStyleContainer">
              <div class="dot"></div>
              <div class="dot"></div>
              <div class="dot"></div>
            </div>
            <div class="personInfoTextContainer">
              <div class="aboutTextContainer">
'''
html1 = '''
<br>
              </div>
            </div>
          </div>
        </div>
</body>
'''
print(html0)
print(res00)
print(res0)
print(res1)
print(res2)
print(res3)
print(res4)
print(res5)
print(res6)
print(html1)