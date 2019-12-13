import pickle as pkl

simplifications  = {'dressed in':'wearing', 'looking for':'looking_for', 'looking at':'looking_at', 'looking':'looking_at', 'pointing':'pointing_at', 'in the middle of':'in_between', 'in the center of':'in_between', 'in mirror':'reflected_in', 'light blue':'light_blue'}

print (simplifications.items())
gqa_objects = {k.replace(' ','_'):v for k,v in pkl.load(open('../data/preprocessed/object_vocab.pkl', 'rb')).items()}
gqa_attributes = {k.replace(' ','_'):v for k,v in pkl.load(open('../data/preprocessed/attribute_vocab.pkl', 'rb')).items()}
gqa_relations = {k.replace(' ','_'):v for k,v in pkl.load(open('../data/preprocessed/relation_vocab.pkl', 'rb')).items()}
gqa_relations['looking_for'] = len(gqa_relations)
to_add_to_rels = ["shaking hands","brushing teeth","taking bath","taking a photo","taking a picture","making a face","sticking out of", "looking_for", "taking pictures", "looking", "taking a photo"]
to_add_to_attrs = ["the same direction","across","along","beyond","backwards","ahead","away", "at camera","front","upward","upwards","downwards","downward", "forward","outside"]
to_add_to_objs = ["hand towel","cast on shadow","eraser","the camera","frying pan","dish towel", "thing"]
for k in to_add_to_rels:
    gqa_relations[k] = len(gqa_relations)
for k in to_add_to_attrs:
    gqa_attributes[k] = len(gqa_attributes)
for k in to_add_to_objs:
    gqa_objects[k] = len(gqa_objects)

data=pkl.load(open('../data/preprocessed/querytype_query_converted_programs_list.pkl', 'rb'))
for q_type in data:
    for i in range(len(data[q_type])):
       for j in range(len(data[q_type][i]['converted_program'])):
          arguments = data[q_type][i]['converted_program'][j]['arguments']
          argument_types = data[q_type][i]['converted_program'][j]['argument_types']
          new_args = []
          new_arg_types = []
          for arg, arg_type in zip(arguments, argument_types):
             arg_orig = arg
             if arg_type=='none':
                for k,v in simplifications.items():
                    if k in arg:
                      arg = arg.replace(k, v)
                      break
                types = []
                words = []
                last_type = None
                for word in arg.split(' '):
                    if word in gqa_objects:
                       if last_type=='object':
                          words[-1] = words[-1] +' '+word
                       else:
                          types.append('object')
                          words.append(word)
                       last_type = 'object'
                    elif word in gqa_relations:
                       if last_type=='relation':
                          words[-1] = words[-1] +' '+word 
                       else:
                          types.append('relation')
                          words.append(word)
                       last_type = 'relation'
                    elif word in gqa_attributes:
                       if last_type=='attribute':
                          words[-1] = words[-1]+' '+word
                       else:
                          types.append('attribute')
                          words.append(word)
                       last_type = 'attribute'
                #if len(types)>0:
                #    print (arg_orig, '--->', types)
                if len(types)==0 and not arg.isdigit():
                    print (arg_orig , '--->????????')
                new_args.extend(words)
                new_arg_types.extend(types)
             else:
                new_args.append(arg)
                new_arg_types.append(arg_type)
          data[q_type][i]['converted_program'][j]['arguments'] = new_args
          data[q_type][i]['converted_program'][j]['argument_types'] = new_arg_types

pkl.dump(data, open('../data/preprocessed/querytype_query_converted_programs_list.pkl', 'wb'))           
