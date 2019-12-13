#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 17:02:54 2019

@author: amrita
"""
import pickle as pkl
import json

gqa_dir = '..'
attribute_region_data = {}
synset_region_data = {}
relation_region_data = {}
split = 'train'
gqa_scenegraph_file = gqa_dir+'/data/raw/scenegraphs/'+split+'_sceneGraphs.json'
gqa_data = json.load(open(gqa_scenegraph_file))
for image_id, image_data in gqa_data.items():
    objects = image_data['objects']
    for object_id, object_val in objects.items():
        object = object_val['name']
        h = object_val['h']
        w = object_val['w']
        x = object_val['x']
        y = object_val['y']
        bbox = [x, y, h, w]
        attributes = object_val['attributes']
        for rel in object_val['relations']:
            rel_name = rel['name']
            h2 = objects[rel['object']]['h']
            w2 = objects[rel['object']]['w']
            x2 = objects[rel['object']]['x']
            y2 = objects[rel['object']]['y']
            bbox2 = [x2, y2, h2, w2]
            if rel_name not in relation_region_data:
                relation_region_data[rel_name] = {}
            if object not in relation_region_data[rel_name]:
                relation_region_data[rel_name][object] = {}
            object2 =objects[rel['object']]['name']
            if object2 not in relation_region_data[rel_name][object]:
                relation_region_data[rel_name][object][object2] = [] 
            relation_region_data[rel_name][object][object2].append({'image_id':image_id, 'bbox1':bbox, 'bbox2':bbox2})    
        if object not in synset_region_data:
            synset_region_data[object] = []
        synset_region_data[object].append({'image_id':image_id, 'bbox':bbox})    
        for attribute in attributes:
            if attribute not in attribute_region_data:
                attribute_region_data[attribute] = {}
            if object not in attribute_region_data[attribute]:
                attribute_region_data[attribute][object] = []
            attribute_region_data[attribute][object].append({'image_id':image_id, 'bbox':bbox})
pkl.dump(synset_region_data, open(gqa_dir+'/data/preprocessed/'+split+'_synset_regions.pkl', 'wb'))
pkl.dump(attribute_region_data, open(gqa_dir+'/data/preprocessed/'+split+'_attribute_regions.pkl', 'wb'))
pkl.dump(relation_region_data, open(gqa_dir+'/data/preprocessed/'+split+'_relation_regions.pkl', 'wb'))
            
      

