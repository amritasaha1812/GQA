#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 11:54:50 2019

@author: amrita
"""
import re
import json
import pickle as pkl
import itertools
import os
import spacy
from pattern.en import singularize

class ConvertGQAProgramToCustom():
    
    def __init__(self):
        if not os.path.exists('object_vocab.pkl') or  not os.path.exists('relation_vocab.pkl') or  not os.path.exists('attribute_vocab.pkl'):
            self.gqa_objects = set([])
            self.gqa_relations = set([])
            self.gqa_attributes = set([])
            for gqa_file in os.listdir('../data/raw/scenegraphs/'):
                gqa_data = json.load(open('../data/raw/scenegraphs/'+gqa_file))
                for k in gqa_data:
                    for o in gqa_data[k]['objects']:
                        self.gqa_objects.add(gqa_data[k]['objects'][o]['name'])
                        self.gqa_relations.update(set([gqa_data[k]['objects'][o]['relations'][i]['name'] for i in range(len(gqa_data[k]['objects'][o]['relations']))]))
                        self.gqa_attributes.update(gqa_data[k]['objects'][o]['attributes'])
            pkl.dump(self.gqa_objects, open('object_vocab.pkl', 'wb'))
            pkl.dump(self.gqa_relations, open('relation_vocab.pkl', 'wb'))
            pkl.dump(self.gqa_attributes, open('attribute_vocab.pkl', 'wb'))            
        else:
            self.gqa_objects = pkl.load(open('object_vocab.pkl', 'rb'))
            self.gqa_relations = pkl.load(open('relation_vocab.pkl', 'rb'))
            self.gqa_attributes = pkl.load(open('attribute_vocab.pkl', 'rb'))
        self.nlp = spacy.load("en_core_web_sm")    
        self.spatial_positions = ['top', 'bottom', 'left', 'right', 'center', 'middle']
        self.gender_pronouns = ['he', 'she', 'it', 'they', 'we', 'her', 'his']
        self.property_types = ["activity", "age", "appearance", "arrangement", "brightness", "cleanliness", "cleanliness", "clothes", "color", "company", "density", "depth", "digital", "face expression", "fatness", "fertility", "fit", "flavor", "food", "fruit", "hardness", "height", "length", "liquid", "location", "material", "nature", "occasion", "opaqness", "orientation", "pattern", "place", "pose", "position", "quantity", "race", "realism", "room", "shape", "signal", "size", "sport", "sportActivity", "state", "strength", "temperature_state", "texture", "tone", "way of cut", "weather", "width", "gender", "weight", "direction", "object", "cake slice", "name", "-", "_", "hposition", "thickness", "this", "scene", "furniture", "fast food", "clothing", "watercraft", "type", "over"]
        print ('finished loading gqa objects, relations, attributes')
        
    def get_argument_types(self, arguments):
        argtypes = []
        for arg in arguments:
            if '|' in arg:
                arguments_i = [self.get_argument_types([x]) for x in arg.split('|')]
                if all([x=='object' for x in arguments_i]):
                    argtype = ['object']*len(arguments_i)
                elif all([x=='relation' for x in arguments_i]):
                    argtype = ['relation']*len(arguments_i)
                elif all([x=='attribute' for x in arguments_i]):
                    argtype = ['attribute']*len(arguments_i)
                else:
                    argtype = arguments_i
                argtype = '|'.join(list(itertools.chain(*argtype)))
            else:  
                arg_parsed = self.nlp(arg)
                arg_lemma = ' '.join([tok.lemma_ for tok in arg_parsed])
                arg_sing = ' '.join([singularize(tok.text) for tok in arg_parsed])
                if arg in self.gqa_objects or arg_lemma in self.gqa_objects or arg_sing in self.gqa_objects:
                    argtype = 'object'
                elif arg in self.gqa_attributes or arg_lemma in self.gqa_attributes or arg_sing in self.gqa_attributes:
                    argtype = 'attribute'
                elif arg in self.gqa_relations or arg_lemma in self.gqa_relations or arg_sing in self.gqa_relations:
                    argtype = 'relation'
                elif arg in ['s', 'o']:
                    argtype = 'scene'
                elif arg in self.spatial_positions:
                    argtype = 'spatial_positions'
                elif arg in self.gender_pronouns:
                    argtype = 'gender_pronouns'
                elif arg in self.property_types:
                    argtype = 'property_type'
                elif arg.startswith('not'):
                   arg = arg.replace('not', '', 1).strip()
                   argtype = self.get_argument_types([arg])         
                   if argtype != 'none':
                      if type(argtype)==list:
                          argtype = '_'.join(argtype)
                      argtype = 'not_'+argtype
                   else:
                      #print ('cannot find arg ', arg)
                      argtype = 'none'
                elif arg.startswith('same '):
                    arg = arg.replace('same ', '', 1).strip()
                    argtype = self.get_argument_types([arg])
                    if argtype != 'none':
                       if type(argtype)==list:
                          argtype = '_'.join(argtype)
                       argtype = 'same_'+argtype   
                    else:
                       #print ('cannot find arg ', arg)
                       argtype = 'none'
                else:
                    #print ('cannot find arg ', arg)
                    argtype = 'none'
            argtypes.append(argtype)
        return argtypes
    
    def convert_all_programs(self):
        data = pkl.load(open('../data/preprocessed/querytype_querylist.pkl', 'rb'))
        #ignore_questypes = set(['existAttrC', 'relVerifyCo', 'weather', 'weatherVerify', 'placeVerifyC', 'weatherChoose', 'weatherVerifyC', 'exist', 'existOr', 'existRelS', 'existRelSC'])
        new_data = pkl.load(open('querytype_query_converted_programs_list.pkl', 'rb'))
        querytype_index_map = {k:len(v) for k,v in new_data.items()}#'chooseAttr':9000}
        #while True: 
        #  if all([querytype_index_map[query_type]>=len(data[query_type]) for query_type in querytype_index_map]):
        #    break
        for query_type in data:
            if query_type not in new_data:
               new_data[query_type] = []
            #if query_type in ignore_questypes:
            #    continue  
            if query_type not in querytype_index_map:
                querytype_index_map[query_type] = 0
            if querytype_index_map[query_type]>=len(data[query_type]):
                continue
            for i in range(querytype_index_map[query_type], len(data[query_type])):#min(querytype_index_map[query_type]+3000, len(data[query_type]))):
                entry = data[query_type][i]
                question = entry['question']
                program = entry['program']
                converted_program = self.convert_program(program)
                new_entry = {'question':question, 'program':program, 'converted_program':converted_program}
                new_data[query_type].append(new_entry)
                if i%1000==0:
                   print ('finished ', i, 'out of ', len(data[query_type]) , 'of type ', query_type)
            querytype_index_map[query_type] = i
            #if len(new_data[query_type])%1000==0:
            pkl.dump(new_data, open('querytype_query_converted_programs_list.pkl', 'wb'))
            print ('dumped data')
            #if all([querytype_index_map[query_type]>=len(data[query_type]) for query_type in querytype_index_map]):
            #    break
          #if all([querytype_index_map[query_type]>=len(data[query_type]) for query_type in querytype_index_map]):
          #  break
        return new_data        
                
    def convert_program(self, program):
        converted_program = []
        converted_i = 0
        i_converted_i_map = {'none':'scene_none'}
        #print ('original program ', program)    
        for line_i,line in enumerate(program):
            operation = line['operation']
            argument = [re.sub(r"[0-9\-, ]*\)", "", re.sub(r"\([0-9\-, ]*", "", re.sub(r"\([0-9\-, ]*\)", "", x))).strip() for x in line['argument'].replace('?','').split(',')]
            if '' in argument:
                argument.remove('')
            for x in argument:
                if x.isdigit() and len(x)>4:
                    argument.remove(x)
            argument_types = self.get_argument_types(argument)
            if 'none' in set(argument_types):
                print ('cannot find ', [argument[i] for i in range(len(argument)) if argument_types[i]=='none'])
            if len(line['dependencies'])==0:
                line['dependencies'] = ['none']
            if not all([x in i_converted_i_map for x in line['dependencies']]):
                print ('dependency not found in map')
                print ('i_converted_i_map', i_converted_i_map)
                print ('dependencies', line['dependencies'])
            if operation.endswith("rel") or operation=='relate':
                converted_operation = operation
                converted_arguments = []
                converted_argument_types = []
                dependencies = line['dependencies']
                for i in range(len(argument)):
                    arg = argument[i]
                    arg_type = argument_types[i]
                    if arg=='s' or arg=='o':
                        converted_arguments.extend([i_converted_i_map[d] for d in dependencies])
                        converted_argument_types.extend(['scene']*len(dependencies))
                    else:
                        converted_arguments.append(arg)
                        converted_argument_types.append(arg_type)
                if operation.startswith('verify'):
                    converted_operation = 'verify_rel'
                elif operation.startswith('choose'):
                    converted_operation = 'choose_rel'
            else:        
                if operation == 'select':
                    converted_operation = "filter_object"
                    dependencies = line['dependencies']
                    converted_arguments = [i_converted_i_map[d] for d in dependencies]+argument
                    converted_argument_types = ['scene']*len(dependencies)+argument_types
                elif operation.startswith("filter"):
                    if set(argument_types)==set(['attribute']):
                        converted_operation = "filter_attribute"
                    elif set(argument_types)==set(['object']):
                        converted_operation = "filter_object"
                    property_type = '_'.join(operation.split(' ')[1:])
                    if len(property_type)=='':
                        property_type = None
                    dependencies = line['dependencies']
                    if property_type:
                        converted_arguments = [i_converted_i_map[d] for d in dependencies]+[property_type] + argument
                        converted_argument_types = ['scene']*len(dependencies)+['property_type']+argument_types
                    else:
                        converted_arguments = [i_converted_i_map[d] for d in dependencies] + argument
                        converted_argument_types = ['scene']*len(dependencies)+argument_types
                elif operation.startswith("choose"):
                    converted_operation = "choose"
                    property_type = '_'.join(operation.split(' ')[1:])
                    if len(property_type)=='':
                        property_type = None
                    dependencies = line['dependencies']
                    if property_type:
                        converted_arguments = [i_converted_i_map[d] for d in dependencies]+[property_type] + argument
                        converted_argument_types = ['scene']*len(dependencies)+['property_type']+argument_types
                    else:
                        converted_arguments = [i_converted_i_map[d] for d in dependencies] + argument
                        converted_argument_types = ['scene']*len(dependencies)+argument_types
                elif operation.startswith("verify"):
                    converted_operation = "verify"
                    property_type = '_'.join(operation.split(' ')[1:])
                    if len(property_type)=='':
                        property_type = None
                    dependencies = line['dependencies']
                    if property_type:
                        converted_arguments = [i_converted_i_map[d] for d in dependencies]+[property_type] + argument
                        converted_argument_types = ['scene']*len(dependencies)+['property_type']+argument_types
                    else:
                        converted_arguments = [i_converted_i_map[d] for d in dependencies] + argument
                        converted_argument_types = ['scene']*len(dependencies)+argument_types
                else:
                    converted_operation = operation
                    property_type = '_'.join(operation.split(' ')[1:])
                    if len(property_type)=='':
                        property_type = None
                    dependencies = line['dependencies']
                    if property_type:
                        converted_arguments = [i_converted_i_map[d] for d in dependencies]+[property_type] + argument
                        converted_argument_types = ['scene']*len(dependencies)+['property_type']+argument_types
                    else:
                        converted_arguments = [i_converted_i_map[d] for d in dependencies] + argument
                        converted_argument_types = ['scene']*len(dependencies)+argument_types
            converted_output = {'operation':converted_operation, 'arguments': converted_arguments, 'argument_types': converted_argument_types}
            converted_program.append(converted_output)
            i_converted_i_map[line_i] = 'scene_'+str(converted_i)
            converted_i += 1
        #print ('converted program ', converted_program)
        #print ('\n')        
        return converted_program

if __name__=="__main__":
    program_converter = ConvertGQAProgramToCustom()
    new_data = program_converter.convert_all_programs()            
    pkl.dump(new_data, open('querytype_query_converted_programs_list.pkl', 'wb'))        
    
    
