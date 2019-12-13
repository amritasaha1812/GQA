#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 12:53:03 2019

@author: amrita
"""
import pickle as pkl

def get_operator_stats():
    operator_stats_data = {}
    program_data = pkl.load(open('../data/preprocessed/querytype_query_converted_programs_list.pkl', 'rb'))
    for query_type in program_data:
        program_data_for_qtype = program_data[query_type]
        for data in program_data_for_qtype:
            converted_program = data['converted_program']
            num_lines = len(converted_program)-1
            for num_line,line in enumerate(converted_program):
                operator = line['operation']
                if operator not in operator_stats_data:
                    operator_stats_data[operator] = {'last-line':0, 'not-last-line':0}
                if num_line==num_lines:
                    operator_stats_data[operator]['last-line']+=1
                else:
                    operator_stats_data[operator]['not-last-line']+=1
                    if operator =='verify' or operator=='exists':
                        print ('Query ', data['question'])
                        print ('Program ')
                        for k in converted_program:
                            print (k['operation'], k['argument_types'], k['arguments'])
                        print ('\n')
    pkl.dump(operator_stats_data, open('../data/preprocessed/operator_stats.pkl', 'wb'))                

if __name__=="__main__":
    get_operator_stats()                    
