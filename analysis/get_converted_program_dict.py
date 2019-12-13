import pickle as pkl

gqa_dir = '..'
converted_program_data = pkl.load(open(gqa_dir+'/data/preprocessed/querytype_query_converted_programs_list.pkl', 'rb'))
query_data_dict = {}
for query_type in converted_program_data:
    query_data = converted_program_data[query_type]
    if query_type not in query_data_dict:
        query_data_dict[query_type] = {}
    for data in query_data:
        q = data['question']
        converted_program_str = []
        for line in data['converted_program']:
            line = line['operation']+'('+','.join([str(x)+':'+str(y) for x,y in zip(line['argument_types'],line['arguments'])])+')'
            converted_program_str.append(line)
        if q not in query_data_dict[query_type]:
            query_data_dict[query_type][q] = []
        query_data_dict[query_type][q].append('\n'.join(converted_program_str))
pkl.dump(query_data_dict, open(gqa_dir+'/data/preprocessed/querytype_query_converted_programs_dict.pkl', 'wb'))        
