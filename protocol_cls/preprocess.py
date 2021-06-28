# -*- coding: utf-8 -*-
# @Author: xiegr
# @Date:   2020-08-28 21:26:15
# @Last Modified by:   xiegr
# @Last Modified time: 2020-09-16 21:19:38
import numpy as np
import dpkt
import random
import pickle

protocols = ['dns', 'smtp', 'ssh', 'ftp', 'http', 'https']
ports = [53, 25, 22, 21, 80, 443]

def gen_flows(pcap):
	flows = [{} for _ in range(len(protocols))]

	if pcap.datalink() != dpkt.pcap.DLT_EN10MB:
		print('unknow data link!')
		return

	xgr = 0
	for _, buff in pcap:
		eth = dpkt.ethernet.Ethernet(buff)
		xgr += 1
		if xgr % 500000 == 0:
			print('The %dth pkt!'%xgr)
			# break

		if isinstance(eth.data, dpkt.ip.IP) and (
		isinstance(eth.data.data, dpkt.udp.UDP)
		or isinstance(eth.data.data, dpkt.tcp.TCP)):
			# tcp or udp packet
			ip = eth.data

			# loop all protocols
			for name in protocols:
				index = protocols.index(name)
				if ip.data.sport == ports[index] or \
				ip.data.dport == ports[index]:
					if len(flows[index]) >= 10000:
						# each class has at most 1w flows
						break
					# match a protocol
					key = '.'.join(map(str, map(int, ip.src))) + \
					'.' + '.'.join(map(str, map(int, ip.dst))) + \
					'.' + '.'.join(map(str, [ip.p, ip.data.sport, ip.data.dport]))

					if key not in flows[index]:
						flows[index][key] = [ip]
					elif len(flows[index][key]) < 1000:
						# each flow has at most 1k flows
						flows[index][key].append(ip)
					# after match a protocol quit
					break

	return flows


# def split_train_test(flows, name, k):
# 	keys = list(flows.keys())

# 	test_keys = keys[k*int(len(keys)*0.1):(k+1)*int(len(keys)*0.1)]
# 	test_min = 0xFFFFFFFF
# 	test_flows = {}
# 	for k in test_keys:
# 		test_flows[k] = flows[k]
# 		test_min = min(test_min, len(flows[k]))

# 	train_keys = set(keys) - set(test_keys)
# 	train_min = 0xFFFFFFFF
# 	train_flows = {}
# 	for k in train_keys:
# 		train_flows[k] = flows[k]
# 		train_min = min(train_min, len(flows[k]))

# 	print('============================')
# 	print('Generate flows for %s'%name)
# 	print('Total flows: ', len(flows))
# 	print('Train flows: ', len(train_flows), ' Min pkts: ', train_min)
# 	print('Test flows: ', len(test_flows), ' Min pkts: ', test_min)

# 	return train_flows, test_flows


def closure(flows):
	flow_dict = {}
	for name in protocols:
		index = protocols.index(name)
		flow_dict[name] = flows[index]
		print('============================')
		print('Generate flows for %s'%name)
		print('Total flows: ', len(flows[index]))
		cnt = 0
		for k, v in flows[index].items():
			cnt += len(v)
		print('Total pkts: ', cnt)

	with open('pro_flows.pkl', 'wb') as f:
		pickle.dump(flow_dict, f)

if __name__ == '__main__':
	pcap = dpkt.pcap.Reader(open('/data/xgr/sketch_data/wide/202006101400.pcap', 'rb'))
	flows = gen_flows(pcap)
	closure(flows)


