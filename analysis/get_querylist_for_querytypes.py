#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 19:57:26 2019

@author: amrita
"""
import json
import os
import pickle as pkl

gqa_data_questions_file = '/dccstor/cssblr/amrita/GQA/data/raw/questions/train_balanced_questions.json'
querytype_querylist_map = {}
#for file in os.listdir(gqa_data_questions_path):
data = json.load(open(gqa_data_questions_file))
for k,v in data.items():
        query_type = v['types']['detailed']
        if query_type not in querytype_querylist_map:
            querytype_querylist_map[query_type] = []
        question = v['question']
        semantic = v['semantic']
        querytype_querylist_map[query_type].append({'question':question, 'program':semantic})
pkl.dump(querytype_querylist_map, open('querytype_querylist.pkl', 'wb'))        
