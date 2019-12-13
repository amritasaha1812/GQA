#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 17:00:40 2019

@author: amrita
"""
import json
import pickle as pkl
import spacy
nlp = spacy.load("en_core_web_sm")
from annoy import AnnoyIndex
import os
import numpy as np

# Compute overlap between our custom object, relation, attribute and GQA object, relation, attribute lexicons
gqa_dataset_file = '/dccstor/cssblr/amrita/GQA/data/train_sceneGraphs.json'
vqa_experiment_dir = '/dccstor/cssblr/amrita/NSVQA_Pipeline/preprocessed_data/vqa/cyear_2014_csplit_train_bbox_gold_obj_gold_attr_gold_dataset_vg_intersection_coco_max_qcon_10_qcon_emb_dim_100'
vg_object_lexicon_file = vqa_experiment_dir+'/object_vocab.pkl'
vg_attribute_lexicon_file = vqa_experiment_dir+'/attribute_vocab.pkl'
glove_embedding_file = '/dccstor/cssblr/amrita/GloVe_Clustering/data/glove/glove_6B.100d.pkl'
if not os.path.exists(glove_embedding_file):
    glove_embedding_file = '/dccstor/cssblr/amrita/GloVe_Clustering/data/glove/glove.6B.100d.txt'
    glove_embedding = {}
    for line in open(glove_embedding_file).readlines():
        words = line.split()
        word = words[0]
        vector = np.asarray([float(x) for x in words[1:]])
        glove_embedding[word] = vector
    pkl.dump(glove_embedding, open('/dccstor/cssblr/amrita/GloVe_Clustering/data/glove/glove_6B.100d.pkl', 'wb'))
else:
    glove_embedding = pkl.load(open(glove_embedding_file, 'rb'))

vg_object_lexicon = set(pkl.load(open(vqa_experiment_dir+'/object_vocab.pkl', 'rb')).keys())
vg_attribute_lexicon = set(pkl.load(open(vqa_experiment_dir+'/attribute_vocab.pkl', 'rb')).keys())
vg_object_lexicon.update(vg_attribute_lexicon)
vg_attribute_lexicon.update(vg_object_lexicon)
vg_concept_annoy_index = AnnoyIndex(100, 'euclidean')
vg_concept_annoy_index.load('/dccstor/cssblr/amrita/VisualGenome/data/preprocessed/annoy_indices/object_attr_annoy_index/annoy_glove_index.ann')
#print (vg_concept_annoy_index.get_n_items())
#print (vg_concept_annoy_index.get_nns_by_item(0, 1))
vg_concepts_dict = pkl.load(open('/dccstor/cssblr/amrita/VisualGenome/data/preprocessed/annoy_indices/object_attr_annoy_index/annoy_glove_words.pkl', 'rb'))
vg_attributes_glove_embedding = {}
for k in vg_attribute_lexicon:
    if k in glove_embedding:
        vg_attributes_glove_embedding[k] = glove_embedding[k]

gqa_data = json.load(open(gqa_dataset_file))
gqa_objects = set([])
gqa_relations = set([])
gqa_attributes = set([])
for k in gqa_data:
    for o in gqa_data[k]['objects']:
        gqa_objects.add(gqa_data[k]['objects'][o]['name'])
        gqa_relations.update(set([gqa_data[k]['objects'][o]['relations'][i]['name'] for i in range(len(gqa_data[k]['objects'][o]['relations']))]))
        gqa_attributes.update(gqa_data[k]['objects'][o]['attributes'])
objects_intersection = vg_object_lexicon.intersection(gqa_objects)
for o in gqa_objects:
    nlp_o = nlp(o)
    o_toks = [tok.text for tok in nlp_o]
    o_lemma = [tok.lemma_ for tok in nlp_o]
    if any([(o_toks[i] in vg_object_lexicon or o_lemma[i] in vg_object_lexicon) for i in range(len(o_toks))]):
        objects_intersection.add(o)
    else:
        print ('did not find ', o, [(o, o in vg_object_lexicon) for o in o_toks], [(o, o in vg_object_lexicon) for o in o_lemma])
        for i in range(len(o_toks)):
           if o_toks[i] in glove_embedding:
               o_i_emb = list(glove_embedding[o_toks[i]])
           elif o_lemma[i] in glove_embedding:
               o_i_emb = list(glove_embedding[o_lemma[i]])
           else:
               continue
           nn_vg_concept = vg_concept_annoy_index.get_nns_by_vector(o_i_emb, 1)[0]
           print (o_toks[i], '---->', vg_concepts_dict[nn_vg_concept])

           
    
print ('intersecting objects ', len(objects_intersection),' out of ', len(gqa_objects))

attributes_intersection = vg_attribute_lexicon.intersection(gqa_attributes)
for a in gqa_attributes:
    nlp_a = nlp(a)
    a_toks = [tok.text for tok in nlp_a]
    a_lemma = [tok.lemma_ for tok in nlp_a]
    if any([(a_toks[i] in vg_attribute_lexicon or a_lemma[i] in vg_attribute_lexicon) for i in range(len(a_toks))]):
        attributes_intersection.add(a)
    else:
        print ('did not find ', a, [(a, a in vg_attribute_lexicon) for a in a_toks], [(a, a in vg_attribute_lexicon) for a in a_lemma])
        for i in range(len(a_toks)):
            if a_toks[i] in glove_embedding:
                a_i_emb = list(glove_embedding[a_toks[i]])
            elif a_lemma[i] in glove_embedding:
                a_i_emb = list(glove_embedding[a_lemma[i]])
            else:
                continue
            nn_vg_concept = vg_concept_annoy_index.get_nns_by_vector(a_i_emb, 1)[0]            
            print (a_toks[i], '---->', vg_concepts_dict[nn_vg_concept])

