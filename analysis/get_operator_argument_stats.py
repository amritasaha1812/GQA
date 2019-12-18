import pickle as pkl
from collections import Counter
operatorwise_argtype_arg_counter = {}
argtype_arg_counter = {}
operator_counter = Counter()
d=pkl.load(open('../data/preprocessed/querytype_query_converted_programs_dict.pkl', 'rb'))
for prog_type in d:
	instances = d[prog_type]
	for question in instances:
		programs = instances[question]
		for program in programs:
			program = program.split('\n')
			for line in program:
				operation = line.split('(')[0].strip()
				operator_counter.update([operation])
				arguments = [x.split(':') for x in line.split(')')[0].split('(')[1].split(',')]
				for arg_type, arg in arguments:
					if arg_type not in argtype_arg_counter:
						argtype_arg_counter[arg_type] = Counter()
					argtype_arg_counter[arg_type].update([arg])	
				argument_types = ','.join([x[0] for x in arguments])
				arguments = [x[1] for x in arguments]
				if operation not in operatorwise_argtype_arg_counter:
					operatorwise_argtype_arg_counter[operation] = {}
				if argument_types not in operatorwise_argtype_arg_counter[operation]:
					operatorwise_argtype_arg_counter[operation][argument_types] = []	
					for arg in arguments:
						operatorwise_argtype_arg_counter[operation][argument_types].append(Counter())
				for i, arg in enumerate(arguments):
					operatorwise_argtype_arg_counter[operation][argument_types][i].update([arg])
pkl.dump(operator_counter, open('../data/preprocessed/operator_counter.pkl', 'wb'))
pkl.dump(argtype_arg_counter, open('../data/preprocessed/argtype_arg_counter.pkl', 'wb'))
pkl.dump(operatorwise_argtype_arg_counter, open('../data/preprocessed/operatorwise_argtype_arg_counter.pkl', 'wb'))
print ('Operator_counter: ', operator_counter)
print ('Argtype_arg_counter: ', argtype_arg_counter)
print ('Operatorwise_argtype_arg_counter: ', operatorwise_argtype_arg_counter)


