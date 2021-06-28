# -*- coding: utf-8 -*-
# @Author: xiegr
# @Date:   2020-09-06 18:07:51
# @Last Modified by:   xiegr
# @Last Modified time: 2021-06-03 17:16:25
import torch
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import argparse
import time
from tqdm import tqdm, trange
from SAM import SAM
from tool import protocols, load_epoch_data, max_byte_len
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import pickle
import numpy as np

class Dataset(torch.utils.data.Dataset):
	"""docstring for Dataset"""
	def __init__(self, x, y, label):
		super(Dataset, self).__init__()
		self.x = x
		self.y = y
		self.label = label

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		return self.x[idx], self.y[idx], self.label[idx]

def paired_collate_fn(insts):
	x, y, label = list(zip(*insts))
	return torch.LongTensor(x), torch.LongTensor(y), torch.LongTensor(label)

def cal_loss(pred, gold, cls_ratio=None):
	gold = gold.contiguous().view(-1)
	# By default, the losses are averaged over each loss element in the batch. 
	loss = F.cross_entropy(pred, gold)

	# torch.max(a,0) 返回每一列中最大值的那个元素，且返回索引
	pred = F.softmax(pred, dim = -1).max(1)[1]
	# 相等位置输出1，否则0
	n_correct = pred.eq(gold)
	acc = n_correct.sum().item() / n_correct.shape[0]

	return loss, acc*100

def test_epoch(model, test_data):
	''' Epoch operation in training phase'''
	model.eval()

	total_acc = []
	total_pred = []
	total_score = []
	total_time = []
	# tqdm: 进度条库
	# desc ：进度条的描述
	# leave：把进度条的最终形态保留下来 bool
	# mininterval：最小进度更新间隔，以秒为单位
	for batch in tqdm(
		test_data, mininterval=2,
		desc='  - (Testing)   ', leave=False):

		# prepare data
		src_seq, src_seq2, gold = batch
		src_seq, src_seq2, gold = src_seq.cuda(), src_seq2.cuda(), gold.cuda()
		gold = gold.contiguous().view(-1)

		# forward
		torch.cuda.synchronize()
		start = time.time()
		pred, score = model(src_seq, src_seq2)
		torch.cuda.synchronize()
		end = time.time()
		# 相等位置输出1，否则0
		n_correct = pred.eq(gold)
		acc = n_correct.sum().item()*100 / n_correct.shape[0]
		total_acc.append(acc)
		total_pred.extend(pred.long().tolist())
		total_score.append(torch.mean(score, dim=0).tolist())
		total_time.append(end - start)

	return sum(total_acc)/len(total_acc), np.array(total_score).mean(axis=0), \
	total_pred, sum(total_time)/len(total_time)

def train_epoch(model, training_data, optimizer):
	''' Epoch operation in training phase'''
	model.train()

	total_loss = []
	total_acc = []
	# tqdm: 进度条库
	# desc ：进度条的描述
	# leave：把进度条的最终形态保留下来 bool
	# mininterval：最小进度更新间隔，以秒为单位
	for batch in tqdm(
		training_data, mininterval=2,
		desc='  - (Training)   ', leave=False):

		# prepare data
		src_seq, src_seq2, gold = batch
		src_seq, src_seq2, gold = src_seq.cuda(), src_seq2.cuda(), gold.cuda()

		optimizer.zero_grad()
		# forward
		pred = model(src_seq, src_seq2)
		loss_per_batch, acc_per_batch = cal_loss(pred, gold)
		# update parameters
		loss_per_batch.backward()
		optimizer.step()

		# 只有一个元素，可以用item取而不管维度
		total_loss.append(loss_per_batch.item())
		total_acc.append(acc_per_batch)

	return sum(total_loss)/len(total_loss), sum(total_acc)/len(total_acc)

def main(i, flow_dict):
	f = open('results/results_%d.txt'%i, 'w')
	f.write('Train Loss Time Test\n')
	f.flush()

	model = SAM(num_class=len(protocols), max_byte_len=max_byte_len).cuda()
	optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()))
	loss_list = []
	# default epoch is 3
	for epoch_i in trange(5, mininterval=2, \
		desc='  - (Training Epochs)   ', leave=False):

		train_x, train_y, train_label = load_epoch_data(flow_dict, 'train')
		training_data = torch.utils.data.DataLoader(
				Dataset(x=train_x, y=train_y, label=train_label),
				num_workers=0,
				collate_fn=paired_collate_fn,
				batch_size=128,
				shuffle=True
			)
		train_loss, train_acc = train_epoch(model, training_data, optimizer)

		test_x, test_y, test_label = load_epoch_data(flow_dict, 'test')
		test_data = torch.utils.data.DataLoader(
				Dataset(x=test_x, y=test_y, label=test_label),
				num_workers=0,
				collate_fn=paired_collate_fn,
				batch_size=128,
				shuffle=False
			)
		test_acc, score, pred, test_time = test_epoch(model, test_data)
		with open('results/atten_%d.txt'%i, 'w') as f2:
			f2.write(' '.join(map('{:.4f}'.format, score)))

		# write F1, PRECISION, RECALL
		with open('results/metric_%d.txt'%i, 'w') as f3:
			f3.write('F1 PRE REC\n')
			p, r, fscore, _ = precision_recall_fscore_support(test_label, pred)
			for a, b, c in zip(fscore, p, r):
				# for every cls
				f3.write('%.2f %.2f %.2f\n'%(a, b, c))
				f3.flush()
			if len(fscore) != len(protocols):
				a = set(pred)
				b = set(test_label[:,0])
				f3.write('%s\n%s'%(str(a), str(b)))

		# write Confusion Matrix
		with open('results/cm_%d.pkl'%i, 'wb') as f4:
			pickle.dump(confusion_matrix(test_label, pred, normalize='true'), f4)


		# write ACC
		f.write('%.2f %.4f %.6f %.2f\n'%(train_acc, train_loss, test_time, test_acc))
		f.flush()

		# # early stop
		# if len(loss_list) == 5:
		# 	if abs(sum(loss_list)/len(loss_list) - train_loss) < 0.005:
		# 		break
		# 	loss_list[epoch_i%len(loss_list)] = train_loss
		# else:
		# 	loss_list.append(train_loss)

	f.close()


if __name__ == '__main__':
	for i in range(10):
		with open('pro_flows_%d_noip_fold.pkl'%i, 'rb') as f:
			flow_dict = pickle.load(f)
		print('====', i, ' fold validation ====')
		main(i, flow_dict)