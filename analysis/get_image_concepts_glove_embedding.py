#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 17:43:40 2019

@author: amrita
"""
import json
import pickle as pkl
import numpy as np

image_phrase_dict = {}
gqa_dir = '..'
glove_clustering_dir = '../../GloVe_Clustering'
split = 'val'
scenegraph_data = json.load(open(gqa_dir+'/data/raw/scenegraphs/'+split+'_sceneGraphs.json', 'rb'))
glove_embedding = {x.strip().split(' ')[0]:[float(xi) for xi in x.strip().split(' ')[1:]] for x in open(glove_clustering_dir+'/data/glove/glove.6B.100d.txt').readlines()}
count=0
for image_id, image_data in scenegraph_data.items():
    objects = image_data['objects']
    region_words = set([])
    for object_id, object_val in objects.items():
        bbox = [object_val['x'], object_val['y'], object_val['h'], object_val['w']]
        bbox_str = ' '.join([str(x) for x in bbox])
        region_words.add(object_val['name'])
        region_words.update(object_val['attributes'])
        region_words.update([rel['name'] for rel in object_val['relations']])
    #region_wordembs = {w:glove_embedding[w] for w in region_words if w in glove_embedding}
    region_wordembs = {}
    for w in region_words:
        if w in glove_embedding:
             x = glove_embedding[w] 
        elif ' ' in w:
           l = []
           for wi in w.split(' '):
             if wi in glove_embedding:
                l.append(glove_embedding[wi])
           if len(l)>0:
             x = np.mean(np.asarray(l), axis=0).tolist() 
           else:
             x = None
        if x:
           region_wordembs[w] = x  
    image_phrase_dict[image_id] = region_wordembs
pkl.dump(image_phrase_dict, open(gqa_dir+'/data/preprocessed/'+split+'_image_concepts_glove_embedding.pkl', 'wb'))
