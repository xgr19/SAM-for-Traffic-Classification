# -*- coding: utf-8 -*-
# @Author: xiegr
# @Date:   2019-08-05 12:54:10
# @Last Modified by:   xiegr
# @Last Modified time: 2019-08-19 21:53:28
import torch
import random
import argparse
import os

def split_src_tgt(max_sent_len, data_tgt_src):
	'''data_tgt_src:[[0,1,2],[1,1,2]], 
	first one (0 and 1 here) is the class_number'''
	max_sent_len += 1
	src_insts, tgt_insts = [], []

	for tmp in data_tgt_src:
		tgt_insts += [tmp[:1]]
		src = tmp[1:max_sent_len]
		if len(src) < max_sent_len:
			src += [0]*(max_sent_len - len(src))
		src_insts += [src[1:max_sent_len]]

	return src_insts, tgt_insts

def read_txt_file(data_dir, max_sent_len, train_ratio, 
	valid_ratio, test_ratio):
	'''data.txt: 0 1 2 3\n1 1 2 3\n the first one is class_number'''
	# a list of .txt, [.txt, .txt]
	all_files = os.listdir(data_dir)
	all_int_data = []
	for file in all_files:
		with open(os.path.join(data_dir,file)) as f:
			all_lines = f.readlines()
			for line in all_lines:
				# every line, a list of integer [int, int,]
				line = list(map(int, line.split()))
				# every line is a packet
				all_int_data += [line]

	random.shuffle(all_int_data)
	# all_int_data = [packet, packet,], packet = [int, int,]
	train = all_int_data[:int(len(all_int_data)*train_ratio)+1]
	valid = all_int_data[int(len(all_int_data)*train_ratio):
	int(len(all_int_data)*(train_ratio+valid_ratio))+1]
	test = all_int_data[int(len(all_int_data)*(train_ratio+valid_ratio)):]

	train_src,train_tgt = split_src_tgt(max_sent_len, train)
	valid_src,valid_tgt = split_src_tgt(max_sent_len, valid)
	test_src,test_tgt = split_src_tgt(max_sent_len, test)


	# they are all like [[],[],], 2-dim
	return train_src,train_tgt,\
	valid_src,valid_tgt,\
	test_src,test_tgt

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('-data_dir', type=str, default='data/')
	parser.add_argument('-train_ratio', type=float, default=0.6)
	parser.add_argument('-valid_ratio', type=float, default=0.2)
	parser.add_argument('-test_ratio', type=float, default=0.2)
	parser.add_argument('-save', type=str, default='data.map')
	parser.add_argument('-max_sent_len', type=int, default=40)
	parser.add_argument('-classes', type=int, default=9)

	opt = parser.parse_args()

	# Training set
	train_src,train_tgt,\
	valid_src,valid_tgt, \
	test_src,test_tgt, = read_txt_file(\
	    opt.data_dir, opt.max_sent_len, \
	    opt.train_ratio, opt.valid_ratio, opt.test_ratio)

	print('all instances in train: %d'%len(train_tgt))

	data = {
	    'settings': opt,
	    'train': {
	        'src': train_src,
	        'tgt': train_tgt},
	    'valid': {
	        'src': valid_src,
	        'tgt': valid_tgt},
	    'test': {
	        'src': test_src,
	        'tgt': test_tgt}
	        }

	print('[Info] Dumping the processed data to pickle file', opt.save)
	torch.save(data, opt.save)
	print('[Info] Finish.')