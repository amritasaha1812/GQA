#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 10:53:26 2019

@author: amrita
"""
import pickle as pkl
import itertools

operator_argument_type_map = {}
argument_type_values_map = {}
d = pkl.load(open('../data/preprocessed/querytype_query_converted_programs_list.pkl', 'rb'))

for query_type, queries in d.items():
    for query in queries:
        converted_program = query['converted_program']
        for line in converted_program:
            operator = line['operation']
            arguments = line['arguments']
            argument_types = line['argument_types']
            if (any([isinstance(argument_types[i], list) and '|' in arguments[i] for i in range(len(argument_types))])):
                   for i in range(len(argument_types)):
                       if isinstance(argument_types[i], list):
                           argument_types[i] = '|'.join(list(itertools.chain(*argument_types[i])))
            if (any([isinstance(argument_types[i], list)  for i in range(len(argument_types))])):
                   print ('argument types ', argument_types , 'for ', operator)
                   print ('arguments ', arguments)
            if (any([argument_types[i]=='none' and arguments[i].isdigit() for i in range(len(argument_types))])):
               new_arg_types = []
               new_args = []
               for i in range(len(argument_types)):
                   if argument_types[i]!='none' or not arguments[i].isdigit():
                      new_arg_types.append(argument_types[i])
                      new_args.append(arguments[i])
               argument_types = new_arg_types
               arguments = new_args
            if (any('|' in x for x in argument_types)):
               new_arg_types = []
               new_args = []
               for arg_type, arg in zip(argument_types, arguments):
                    if '|' in arg_type:
                       new_arg_types.extend(arg_type.split('|'))
                       new_args.extend(arg.split('|'))
                    else:
                       new_arg_types.append(arg_type)
                       new_args.append(arg)
               argument_types = new_arg_types
               arguments = new_args 
            argument_types_str = ','.join(argument_types)
            arguments_type_value_dict = {k:v for k,v in zip(argument_types, arguments)}
            sorted_argument_types = sorted(argument_types)
            sorted_arguments = [arguments_type_value_dict[k] for k in sorted_argument_types]
            sorted_argument_types_str = ','.join(sorted_argument_types)
            if operator not in operator_argument_type_map:
                operator_argument_type_map[operator] = {}
            if sorted_argument_types_str not in operator_argument_type_map[operator]:
                operator_argument_type_map[operator][sorted_argument_types_str] = []
            operator_argument_type_map[operator][sorted_argument_types_str].append(','.join(sorted_arguments))
            for arg, arg_type in zip(arguments, argument_types):
                if arg_type not in argument_type_values_map:
                    argument_type_values_map[arg_type] = set([])
                argument_type_values_map[arg_type].add(arg)
                
pkl.dump(operator_argument_type_map, open('../data/preprocessed/operator_argument_type_map.pkl', 'wb'))
pkl.dump(argument_type_values_map, open('../data/preprocessed/argument_type_values_map.pkl', 'wb'))
                
                    
