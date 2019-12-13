import numpy as np
import json
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

def pad_to_square(c, h, w, pad_value):
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    del_h = pad[2] + pad[3]
    del_w = pad[0] + pad[1]
    padded_h = h + del_h
    padded_w = w + del_w
    return padded_h, padded_w, pad

def convert_bbox(size, bbox):
   dw = 1./size[0]
   dh = 1./size[1]
   x = (bbox[0] + bbox[1])/2.0
   y = (bbox[2] + bbox[3])/2.0
   w = bbox[1] - bbox[0]
   h = bbox[3] - bbox[2]
   x = x*dw
   w = w*dw
   y = y*dh
   h = h*dh
   return (x,y,w,h)

def check_annotation(o, padded_h, padded_w, pad):
      x1 = o['x']
      y1 = o['y']
      x2 = x1 + o['w']
      y2 = y1 + o['h']
      #print ('x1 y1 x2 y2 ', x1, y1, x2, y2)
      x1 += pad[0]
      y1 += pad[2]
      x2 += pad[1]
      y2 += pad[3]
      x = ((x1 + x2)/2)/padded_w
      y = ((y1 + y2)/2)/padded_h
      w = o['w'] / padded_w
      h = o['h'] / padded_h
      if x >= 1 or y >= 1:
         return True
      else:
         return False
'''
gqa_dir = '..'
scenegraph_data = json.load(open(gqa_dir+'/data/raw/scenegraphs/train_sceneGraphs.json'))
gqa_image_path = gqa_dir+'/data/images/'
for k in scenegraph_data:
   #k = '150344'
   height = scenegraph_data[k]['height']
   width = scenegraph_data[k]['width']
   #print ('height width ', height, width)
   padding_info = None
   error = False
   padded_h, padded_w, pad = pad_to_square(3, height, width, 0)
   print ('padded_h, padded_w, pad  ', padded_h, padded_w, pad )
   for o in scenegraph_data[k]['objects'].values():
      error = check_annotation(o, padded_h, padded_w, pad)
      if error:
          break
   if error:
      print ('SOMETHING WRONG ', k)
   sys.exit(1)
'''
