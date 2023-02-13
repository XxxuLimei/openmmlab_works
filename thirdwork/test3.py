import numpy as np
import matplotlib.pyplot as plt
from mmseg.apis import init_model, inference_model, show_result_pyplot
import mmcv
import cv2
from mmengine import Config

cfg = Config.fromfile('new_cfg.py')
from mmengine.runner import Runner
from mmseg.utils import register_all_modules

# register all modules in mmseg into the registries
# do not init the default scope here because it will be init in the runner

register_all_modules(init_default_scope=False)
runner = Runner.from_cfg(cfg)

# 初始化模型
checkpoint_path = './work_dirs/iter_800.pth'
model = init_model(cfg, checkpoint_path, 'cuda:0')

img = mmcv.imread('./data/Glomeruli-dataset/images/VUHSK_1702_39.png')
result = inference_model(model, img)

pred_mask = result.pred_sem_seg.data[0].cpu().numpy()

# 可视化预测结果
visualization = show_result_pyplot(model, img, result, opacity=0.7, out_file='pred.jpg')
plt.imshow(mmcv.bgr2rgb(visualization))
plt.show()