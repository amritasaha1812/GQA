#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 15:58:04 2019

@author: amrita
"""
import json 
import pickle as pkl
import os
from check_bad_annotations import *
import torchvision.transforms as transforms
from PIL import Image

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

gqa_dir = '/dccstor/cssblr/amrita/GQA/'
gqa_train_scenegraph_file = gqa_dir+'/data/raw/scenegraphs/train_sceneGraphs.json'
gqa_train_scenegraph = json.load(open(gqa_train_scenegraph_file))
gqa_val_scenegraph_file = gqa_dir+'/data/raw/scenegraphs/val_sceneGraphs.json'
gqa_val_scenegraph = json.load(open(gqa_val_scenegraph_file))
gqa_images_dir = gqa_dir+'/data/images/'
gqa_object_hclusters = json.load(open(gqa_dir+'/data/preprocessed/gqa_object_hierarchical_clusters.json'))
gqa_object_categories = []
gqa_objects = {} 
count = 0
gqa_objects_hierarchy = {}
for k, vs in gqa_object_hclusters.items():
     supercategory = str(k)
     for v in vs:
        name = ",".join(v)
        id = count
        gqa_object_categories.append({"supercategory":supercategory, "id":id, "name":name})
        for vi in v:
            print (vi)
            gqa_objects[vi] = count
        gqa_objects_hierarchy[count] = str(k)
        count += 1
gqa_object_cluster_dict = {}
for k in gqa_object_hclusters.keys():
     gqa_object_cluster_dict[str(k)] = count
     count += 1
gqa_objects_hierarchy_todump = []
for k in sorted(gqa_objects_hierarchy):
     gqa_objects_hierarchy_todump.append([k, gqa_object_cluster_dict[gqa_objects_hierarchy[k]]])

dump_dir = gqa_dir+'/data/preprocessed/object_coco_format_data'
if not os.path.exists(dump_dir):
    os.mkdir(dump_dir)
pkl.dump(gqa_objects_hierarchy_todump, open(dump_dir+'/hierarchy_names.pkl', 'wb'))
fw = open(dump_dir+'/category_names.txt','w')
for k in gqa_object_categories:
   fw.write(k['name'].strip()+'\n')
fw.close()
sys.exit(1)
def get_preprocessed_data(gqa_scenegraph, labels=None):
  object_annotations = []
  images = []
  image_names = []
  if labels is None:
     labels = {}
  for id in gqa_scenegraph:
     graph = gqa_scenegraph[id]
     image_path = gqa_images_dir+id+'.jpg'
     width, height =Image.open(image_path).convert('RGB').size
     height1 = graph['height']
     width1 = graph['width']
     if height != height1 or width != width1:
         print ('h,w ', height, width, ' h1,w1', height1, width1)
     padded_h, padded_w, pad = pad_to_square(3, height, width, 0)
     file_name = str(id)+'.jpg'
     if not os.path.exists(gqa_images_dir+file_name):
         raise Exception('Image missing ', gqa_images_dir+file_name)
     if id not in labels:
        labels[id] = []
     for object_data in graph['objects'].values():
        error = check_annotation(object_data, padded_h, padded_w, pad)
        if error:
            print ('Ignored data because of error in annotation in image', id)
            continue
        objects = [object_data['name']]
        bbox = [object_data['x'], object_data['y'], object_data['w'], object_data['h']]
        converted_bbox = convert_bbox((width, height), [object_data['x'], object_data['x']+object_data['w'], object_data['y'], object_data['y']+object_data['h']])
        object_ids = [gqa_objects[x] for x in objects]
        for attr_id in object_ids:
            object_annotations.append({"category_id": attr_id, "id":len(object_annotations), "bbox":bbox, "area":bbox[2]*bbox[3], "image_id":id})
            labels[id].append(str(attr_id)+ " " +" ".join([str(a) for a in converted_bbox]))  
     if len(labels[id])>0:
        images.append({'height':height, 'width':width, 'file_name':file_name, 'id':id})
        image_names.append(file_name) 
  return object_annotations, images, image_names, labels

train_object_annotations, train_images, train_image_names, all_labels = get_preprocessed_data(gqa_train_scenegraph)
valid_object_annotations, valid_images, valid_image_names, all_labels = get_preprocessed_data(gqa_val_scenegraph, all_labels)  
train_object_coco_format_data = {'annotations':train_object_annotations, 'categories':gqa_object_categories, 'images':train_images}
valid_object_coco_format_data = {'annotations':valid_object_annotations, 'categories':gqa_object_categories, 'images':valid_images}

dump_dir = gqa_dir+'/data/preprocessed/object_coco_format_data'
if not os.path.exists(dump_dir):
    os.mkdir(dump_dir)
pkl.dump(gqa_objects_hierarchy, open(dump_dir+'/hierarchy_names.pkl', 'wb')) 
fw = open(dump_dir+'/category_names.txt','w')
for k in gqa_object_categories:
   fw.write(k['name'].strip()+'\n')
fw.close()
sys.exit(1)
if not os.path.exists(dump_dir+'/annotations'):
    os.mkdir(dump_dir+'/annotations')
json.dump(train_object_coco_format_data, open(dump_dir+'/annotations/train_instances.json', 'w'))
json.dump(valid_object_coco_format_data, open(dump_dir+'/annotations/valid_instances.json', 'w'))
fw = open(dump_dir+'/train_images.txt', 'w')
for name in train_image_names:
   fw.write(name.strip()+'\n')
fw = open(dump_dir+'/valid_images.txt', 'w')
for name in valid_image_names:
   fw.write(name.strip()+'\n')
if not os.path.exists(dump_dir+'/labels'):
   os.mkdir(dump_dir+'/labels/')
for id in all_labels:
   fw = open(dump_dir+'/labels/'+str(id)+'.txt', 'w')
   for entry in all_labels[id]:
       fw.write(entry.strip()+'\n')
   fw.close()
