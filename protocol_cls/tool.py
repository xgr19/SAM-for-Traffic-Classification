# -*- coding: utf-8 -*-
# @Author: xiegr
# @Date:   2020-08-30 20:57:53
# @Last Modified by:   xiegr
# @Last Modified time: 2021-06-01 22:28:57
import pickle
import dpkt
import random
import numpy as np
from preprocess import protocols
from tqdm import tqdm, trange

ip_features = {'hl':1,'tos':1,'len':2,'df':1,'mf':1,'ttl':1,'p':1}
tcp_features = {'off':1,'flags':1,'win':2}
udp_features = {'ulen':2}
max_byte_len = 50

def mask(p):
	p.src = b'\x00\x00\x00\x00'
	p.dst = b'\x00\x00\x00\x00'
	p.sum = 0
	p.id = 0
	p.offset = 0

	if isinstance(p.data, dpkt.tcp.TCP):
		p.data.sport = 0
		p.data.dport = 0
		p.data.seq = 0
		p.data.ack = 0
		p.data.sum = 0

	elif isinstance(p.data, dpkt.udp.UDP):
		p.data.sport = 0
		p.data.dport = 0
		p.data.sum = 0

	return p

def pkt2feature(data, k):
	flow_dict = {'train':{}, 'test':{}}

	# train->protocol->flowid->[pkts]
	for p in protocols:
		flow_dict['train'][p] = []
		flow_dict['test'][p] = []
		all_pkts = []
		p_keys = list(data[p].keys())

		for flow in p_keys:
			pkts = data[p][flow]
			all_pkts.extend(pkts)
		random.Random(1024).shuffle(all_pkts)

		for idx in range(len(all_pkts)):
			pkt = mask(all_pkts[idx])
			raw_byte = pkt.pack()

			byte = []
			pos = []
			for x in range(min(len(raw_byte),max_byte_len)):
				byte.append(int(raw_byte[x]))
				pos.append(x)

			byte.extend([0]*(max_byte_len-len(byte)))
			pos.extend([0]*(max_byte_len-len(pos)))
			# if len(byte) != max_byte_len or len(pos) != max_byte_len:
			# 	print(len(byte), len(pos))
			# 	input()
			if idx in range(k*int(len(all_pkts)*0.1), (k+1)*int(len(all_pkts)*0.1)):
				flow_dict['test'][p].append((byte, pos))
			else:
				flow_dict['train'][p].append((byte, pos))
	return flow_dict

def load_epoch_data(flow_dict, train='train'):
	flow_dict = flow_dict[train]
	x, y, label = [], [], []

	for p in protocols:
		pkts = flow_dict[p]
		for byte, pos in pkts:
			x.append(byte)
			y.append(pos)
			label.append(protocols.index(p))

	return np.array(x), np.array(y), np.array(label)[:, np.newaxis]


if __name__ == '__main__':
	# f = open('flows.pkl','rb')
	# data = pickle.load(f)
	# f.close()

	# print(data.keys())

	# dns = data['dns']
	# # print(list(dns.keys())[:10])

	# # wide dataset contains payload
	# print('================\n',
	# 	len(dns['203.206.160.197.202.89.157.51.17.53.51648'][0]))

	# print('================')
	# flow_dict = pkt2feature(data)
	# x, y, label = train_epoch_data(flow_dict)
	# print(x.shape)
	# print(y.shape)
	# print(label[0])
	with open('pro_flows.pkl','rb') as f:
		data = pickle.load(f)

	for i in trange(10, mininterval=2, \
		desc='  - (Building fold dataset)   ', leave=False):
		flow_dict = pkt2feature(data, i)
		with open('pro_flows_%d_noip_fold.pkl'%i, 'wb') as f:
			pickle.dump(flow_dict, f)