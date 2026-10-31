# -*- coding: utf-8 -*-
"""Count parameters for the YOLOv5 detector and the LPRNet recognizer."""
import os
import sys
import types
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# cv2 stub so repo modules import on a machine without OpenCV
try:
    import cv2  # noqa: F401
except ImportError:
    cv2 = types.ModuleType('cv2')
    for k, v in dict(IMREAD_COLOR=1, IMREAD_GRAYSCALE=0, IMREAD_UNCHANGED=-1,
                     INTER_NEAREST=0, INTER_LINEAR=1, INTER_CUBIC=2, INTER_AREA=3,
                     INTER_LANCZOS4=4, BORDER_CONSTANT=0, BORDER_REPLICATE=1,
                     FONT_HERSHEY_SIMPLEX=0, LINE_AA=16, COLOR_BGR2RGB=4,
                     COLOR_RGB2BGR=4, CAP_PROP_FPS=5, CAP_PROP_FRAME_WIDTH=3,
                     CAP_PROP_FRAME_HEIGHT=4).items():
        setattr(cv2, k, v)
    for name in ('imread', 'imwrite', 'resize', 'cvtColor', 'putText', 'rectangle',
                 'imshow', 'namedWindow', 'waitKey', 'destroyAllWindows', 'VideoCapture',
                 'VideoWriter', 'fillPoly', 'polylines', 'circle', 'getTextSize',
                 'addWeighted', 'setNumThreads', 'getNumThreads', 'setUseOptimized',
                 'useOptimized', 'getBuildInformation'):
        setattr(cv2, name, lambda *a, **k: None)
    cv2.VideoWriter_fourcc = lambda *a: 0
    sys.modules['cv2'] = cv2

from lprr.LPRNet import CHARS, build_lprnet  # noqa: E402


def n_params(module):
    return sum(p.numel() for p in module.parameters())


def n_buffers(module):
    return sum(b.numel() for b in module.buffers())


# ---------------- YOLOv5s ----------------
ckpt_path = os.path.join(ROOT, 'runs/train/exp/weights/best.pt')
ckpt = torch.load(ckpt_path, map_location='cpu')
yolo = ckpt['model'].float()
ema = ckpt.get('ema')
y_params = n_params(yolo)
y_buf = n_buffers(yolo)
print('YOLOv5s model params      : %d (%.3f M)' % (y_params, y_params / 1e6))
print('YOLOv5s model buffers     : %d' % y_buf)
if ema is not None:
    e_params = n_params(ema.float())
    print('YOLOv5s EMA params        : %d (%.3f M)' % (e_params, e_params / 1e6))

# ---------------- LPRNet ----------------
net = build_lprnet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0.5)
state = torch.load(os.path.join(ROOT, 'lprr/Final_LPRNet_model.pth'), map_location='cpu')
net.load_state_dict(state)
l_params = n_params(net)
l_buf = n_buffers(net)
print('LPRNet params             : %d (%.3f M)' % (l_params, l_params / 1e6))
print('LPRNet buffers            : %d' % l_buf)
print('LPRNet classes            : %d' % len(CHARS))

print('--- LPRNet per block ---')
for name, mod in net.named_children():
    print('  %-10s %8d' % (name, n_params(mod)))

total = y_params + l_params
print('--- totals ---')
print('Total params              : %d (%.3f M)' % (total, total / 1e6))
print('FP32 weights              : %.2f MB' % (total * 4 / 1024 ** 2))
print('FP16 weights              : %.2f MB' % (total * 2 / 1024 ** 2))
print('YOLO .pt file             : %.2f MB (includes optimizer/EMA)' % (os.path.getsize(ckpt_path) / 1024 ** 2))
print('LPRNet .pth file          : %.2f MB' % (os.path.getsize(os.path.join(ROOT, 'lprr/Final_LPRNet_model.pth')) / 1024 ** 2))

# ---------------- FLOPs if thop is available ----------------
try:
    from thop import profile
    x = torch.zeros(1, 3, 640, 640)
    macs, _ = profile(yolo, inputs=(x,), verbose=False)
    print('YOLOv5s MACs @640         : %.2f G' % (macs / 1e9))
    x2 = torch.zeros(1, 3, 24, 94)
    macs2, _ = profile(net, inputs=(x2,), verbose=False)
    print('LPRNet MACs @94x24        : %.3f G' % (macs2 / 1e9))
except Exception as e:
    print('FLOPs skipped:', type(e).__name__, str(e)[:80])
