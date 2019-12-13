#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 21:08:42 2019

@author: amrita
"""
import pickle as pkl
import numpy as np
from annoy import AnnoyIndex
from nltk.corpus import wordnet as wn
import spacy
import os

glove_clustering_dir = '../../GloVe_Clustering'
glove_emb = {x.strip().split(' ')[0].lower():[float(xi) for xi in x.strip().split(' ')[1:]] for x in open(glove_clustering_dir+'/data/glove/glove.6B.100d.txt').readlines()}
nlp = spacy.load("en_core_web_sm")
def create_annoy_index(gqa_object_categories, index_dir):
    ann_index = AnnoyIndex(100, 'euclidean')
    ann_keys = []
    i=0
    for object in gqa_object_categories:
        emb = get_object_emb(object)
        if emb is not None:
            ann_index.add_item(i, emb)
            ann_keys.append(object)
            i=i+1
    ann_index.build(100)
    ann_index.save(index_dir+'/annoy_index.ann')
    pkl.dump(ann_keys, open(index_dir+'/annoy_keys.pkl', 'wb'))
    return ann_index, ann_keys

def load_annoy_index(index_dir):
    ann_index = AnnoyIndex(100, 'euclidean')
    ann_index.load(index_dir+'/annoy_index.ann')
    ann_keys = pkl.load(open(index_dir+'/annoy_keys.pkl', 'rb'))
    return ann_index, ann_keys

def get_obj_categories_hyponyms(gqa_object_categories):
    all_hyponyms = set([])
    hyponyms_dict = {}
    hypernym_dict = {}
    for obj in gqa_object_categories:
        hyponyms = get_hyponym_tree(obj, 'n')
        if len(hyponyms)==0:
           hyponyms = get_hyponym_tree(obj, 'a')
           if len(hyponyms)==0:
               hyponyms = get_hyponym_tree(obj, 'r')
        print ('[[Obj Category ---> hyponyms]] ' , obj, '--->', hyponyms, '\n')
        all_hyponyms.update(hyponyms)
        hyponyms_dict[obj] = hyponyms
        for x in hyponyms:
            if x not in hypernym_dict:
                hypernym_dict[x] = set([])
            hypernym_dict[x].add(obj)
    return all_hyponyms, hyponyms_dict, hypernym_dict

def get_obj_hyponyms_hypernyms(gqa_objects):
    all_hypohypernyms = set([])
    hypohypernyms_dict = {}
    avg_hypohyper_emb_dict = {}
    for obj in gqa_objects:
       hypohypers = set([])
       for synset in wn.synsets(obj, 'n'):
           hypernyms = synset.hypernyms()
           #hyponyms = synset.hyponyms()
           hypohypers.update(set([x.lemmas()[0].name().lower() for x in hypernyms]))
           #hypohypers.update(set([x.lemmas()[0].name().lower() for x in hyponyms]))
           all_hypohypernyms.update(hypohypers)
       avg_hypohyper_emb = []
       for x in hypohypers:
           emb = get_object_emb(x)
           if emb is not None:
              avg_hypohyper_emb.append(emb)
       if len(avg_hypohyper_emb)>0:
           avg_hypohyper_emb = np.mean(np.asarray(avg_hypohyper_emb), axis=0).tolist()
       hypohypernyms_dict[obj] = hypohypers
       avg_hypohyper_emb_dict[obj] = avg_hypohyper_emb
       print ('[[Obj ---> hypo/hypernyms]] ', obj, '---->', hypohypers, '\n')    
    return all_hypohypernyms, hypohypernyms_dict, avg_hypohyper_emb_dict
       
def get_hyponym_tree(word, pos):
    synsets = wn.synsets(word, 'n')
    hypos_tree = set([])
    for synset in synsets:
       hypos = lambda s:s.hyponyms()
       hypos = set([x.lemmas()[0].name().lower() for x in synset.closure(hypos)])
       hypos = hypos.intersection(set(glove_emb))
       hypos_tree.update(hypos)
       #hypos_tree.update([x.lemmas()[0].name().lower() for x in synset.closure(hypos)])
    return hypos_tree

def get_word_lemma(word):
    return nlp(word)[0].lemma_

def get_object_emb(object):
    emb = None
    if object in glove_emb:
       emb = glove_emb[object]
    else:
       embs = []
       for object in object.replace('_',' ').split(' '):
           if object in glove_emb:
               embs.append(glove_emb[object])
           else:
               lemma = get_word_lemma(object)
               if lemma in glove_emb:
                   embs.append(glove_emb[lemma])
       if len(embs)>0:
           emb = np.mean(np.asarray(embs), axis=0).tolist()
    return emb


gqa_object_categories_file = '../data/preprocessed/gqa_object_categories.txt'
gqa_object_categories = [x.strip() for x in open(gqa_object_categories_file, 'r').readlines()]
gqa_object_category_hyponyms, hyponyms_dict, hypernyms_dict = get_obj_categories_hyponyms(gqa_object_categories)
gqa_objects = pkl.load(open('../data/preprocessed/object_vocab.pkl', 'rb'))
gqa_object_hypohypernyms, hypohypernyms_dict, avg_hypohyper_emb_dict = get_obj_hyponyms_hypernyms(gqa_objects)
all_objects = set(gqa_objects)
all_objects.update(gqa_object_categories)
obj_cat_index_dir = '../data/preprocessed/gqa_object_category_annoy_index'
if not os.path.exists(obj_cat_index_dir):
    os.mkdir(obj_cat_index_dir)
if not os.path.exists(obj_cat_index_dir+'/annoy_index.ann'):
    create_annoy_index(gqa_object_categories, obj_cat_index_dir)
ann_index, ann_keys = load_annoy_index(obj_cat_index_dir)
obj_cat_hyponyms_index_dir = '../data/preprocessed/gqa_object_category_hyponyms_annoy_index'
if not os.path.exists(obj_cat_hyponyms_index_dir):
    os.mkdir(obj_cat_hyponyms_index_dir)
if not os.path.exists(obj_cat_hyponyms_index_dir+'/annoy_index.ann'):
    create_annoy_index(gqa_object_category_hyponyms, obj_cat_hyponyms_index_dir)
ann_hypo_index, ann_hypo_keys = load_annoy_index(obj_cat_hyponyms_index_dir)
print ('ann_hypo_keys ', ann_hypo_keys)
print ('ann_keys ', ann_keys)
print ('ann_index ', ann_index.get_n_items())
fw_lines = []
fw_lines.append('Word\tGlove based Cluster\tWN Hyponym based Cluster\tWN Hypernym based Cluster\t'+'\t'.join(sorted(gqa_object_categories)))
for object in sorted(gqa_objects):
    object_emb = get_object_emb(object)
    if object_emb is None:
       print (object, '::: None Embedding')
       continue
    avg_hypohyper_emb = avg_hypohyper_emb_dict[object]
    nn_object_category = ann_keys[ann_index.get_nns_by_vector(object_emb, 1)[0]]
    nn_obj_category_hyponyms = ann_hypo_keys[ann_hypo_index.get_nns_by_vector(object_emb, 1)[0]]
    nn_object_category2 = hypernyms_dict[nn_obj_category_hyponyms]
    if len(avg_hypohyper_emb)>0:
        nn_object_category3 = ann_keys[ann_index.get_nns_by_vector(avg_hypohyper_emb, 1)[0]]
    else:
        nn_object_category3 = ''
    print (object, '\n\t\tGlove based parent:', nn_object_category,'\n\t\tWN based parent:',nn_object_category2, '(', nn_obj_category_hyponyms, ')','\n\t\tHypoHyper based parent:', nn_object_category3)
    nn_object_category2 = ','.join(list(nn_object_category2))
    fw_lines.append(object+'\t'+nn_object_category +'\t'+nn_object_category2+'\t'+nn_object_category3+'\t' + '\t'.join(['']*len(gqa_object_categories)))
fw = open('../data/preprocessed/gqa_object_clustering_for_annotation.tsv', 'w')
for line in fw_lines:
    fw.write(line+'\n')
fw.close()

   



     
