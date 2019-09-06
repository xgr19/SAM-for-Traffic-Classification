# -*- coding: utf-8 -*-
# @Author: xiegr
# @Date:   2019-08-05 14:02:14
# @Last Modified by:   xiegr
# @Last Modified time: 2019-08-18 16:02:19
import torch
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import argparse
import time
from tqdm import tqdm
from SAN import make_model, Dataset

tmp_pred = []
tmp_tgt = []

def paired_collate_fn(insts):
	src_insts, tgt_insts = list(zip(*insts))
	# return (torch.Tensor(src_insts), torch.LongTensor(tgt_insts))
	return (torch.LongTensor(src_insts), torch.LongTensor(tgt_insts))

def prepare_dataloaders(data, opt):
	train_loader = torch.utils.data.DataLoader(
		Dataset(src_insts=data['train']['src'],
			tgt_insts=data['train']['tgt']),
		num_workers=0,
		collate_fn=paired_collate_fn,
		batch_size=opt.batch_size,
		shuffle=True)

	valid_loader = torch.utils.data.DataLoader(
		Dataset(src_insts=data['valid']['src'],
			tgt_insts=data['valid']['tgt']),
		num_workers=0,
		collate_fn=paired_collate_fn,
		batch_size=opt.batch_size)

	test_loader = torch.utils.data.DataLoader(
		Dataset(src_insts=data['test']['src'],
			tgt_insts=data['test']['tgt']),
		num_workers=0,
		collate_fn=paired_collate_fn,
		batch_size=opt.batch_size)

	return train_loader, valid_loader, test_loader

def cal_loss(pred, gold):
	''' Calculate cross entropy loss'''
	gold = gold.contiguous().view(-1)
	loss = F.cross_entropy(pred, gold)

	return loss

def cal_performance(pred, gold, test=False):
	loss = cal_loss(pred, gold)

	# torch.max(outputs.data,1)
	# 找出每行的最大值
	# (值，idx)：(tensor([ 1., 2., 3.]), tensor([ 0, 0, 0]))
	pred = F.softmax(pred, dim = -1)
	pred = pred.max(1)[1]
	gold = gold.contiguous().view(-1)
	# 相等位置输出1，否则0
	n_correct = pred.eq(gold)
	n_correct = n_correct.sum().item()

	if test:
		global tmp_pred
		global tmp_tgt
		tmp_pred += pred.tolist()
		tmp_tgt += gold.tolist()

	return loss, n_correct

def train_epoch(model, training_data, optimizer):
	''' Epoch operation in training phase'''
	# 使用PyTorch进行训练和测试时一定注意要把实例化的model指定train/eval，eval时框架会自动把BN和DropOut固定住，
	# 不会取平均，而是用训练好的值，不然的话，一旦test的batch_size过小，很容易就会被BN层导致生成图片颜色失真极大
	model.train()

	total_loss = 0
	total_acc = 0
	steps = 0
	# tqdm: 进度条库
	# desc ：进度条的描述
	# leave：把进度条的最终形态保留下来 bool
	# mininterval：最小进度更新间隔，以秒为单位
	for batch in tqdm(
		training_data, mininterval=2,
		desc='  - (Training)   ', leave=False):

		# prepare data
		src_seq, tgt_seq = map(lambda x: x.to(torch.device('cuda')), batch)
		gold = tgt_seq

		# forward
		optimizer.zero_grad()
		pred = model(src_seq)
		loss_per_batch, n_correct_per_batch = cal_performance(pred, gold)
		
		# update parameters
		loss_per_batch.backward()
		optimizer.step_and_update_lr()

		# 只有一个元素，可以用item取而不管维度
		total_loss += loss_per_batch.item()

		num_per_batch = gold.size(0)
		accuracy_per_batch = n_correct_per_batch / num_per_batch
		total_acc += accuracy_per_batch

		steps += 1

	loss_avg = total_loss/steps
	accuracy_avg = total_acc/steps
	return loss_avg, accuracy_avg

def valid_epoch(model, training_data, test=False):
	''' Epoch operation in training phase'''
	# 使用PyTorch进行训练和测试时一定注意要把实例化的model指定train/eval，eval时框架会自动把BN和DropOut固定住，
	# 不会取平均，而是用训练好的值，不然的话，一旦test的batch_size过小，很容易就会被BN层导致生成图片颜色失真极大
	model.eval()

	total_loss = 0
	total_acc = 0
	steps = 0
	# tqdm: 进度条库
	# desc ：进度条的描述
	# leave：把进度条的最终形态保留下来 bool
	# mininterval：最小进度更新间隔，以秒为单位
	for batch in tqdm(
		training_data, mininterval=2,
		desc='  - (Validing)   ', leave=False):

		# prepare data
		src_seq, tgt_seq = map(lambda x: x.to(torch.device('cuda')), batch)
		gold = tgt_seq

		# forward
		pred = model(src_seq)
		loss_per_batch, n_correct_per_batch = cal_performance(pred, gold, test)

		# 只有一个元素，可以用item取而不管维度
		total_loss += loss_per_batch.item()

		num_per_batch = gold.size(0)
		accuracy_per_batch = n_correct_per_batch / num_per_batch
		total_acc += accuracy_per_batch

		steps += 1

	loss_avg = total_loss/steps
	accuracy_avg = total_acc/steps
	return loss_avg, accuracy_avg

def train(model, training_data, validation_data, 
	test_data ,optimizer, opt):
	''' Start training '''
	global tmp_pred
	global tmp_tgt
	best_score = -1

	log_train_file = opt.log + '.train.log'
	log_valid_file = opt.log + '.valid.log'
	train_time_file = opt.log +'.time.txt'

	with open(log_train_file, 'w') as log_tf,\
	open(log_valid_file, 'w') as log_vf,\
	open(train_time_file, 'w') as log_ttf, open('san_metrics.txt','w') as nothing:
		log_tf.write('epoch,loss,accuracy\n')
		log_vf.write('epoch,loss,accuracy\n')
		log_ttf.write('time in second for train epoch\n')
		nothing.write('===\n')

	valid_accus = []
	for epoch_i in range(0, opt.epoch):
		print('[ Epoch', epoch_i, ']')

		# train an epoch
		start = time.time()
		train_loss, train_accu = train_epoch(model, 
			training_data, optimizer)
		end = time.time()
		print('  - (Training)   loss: {loss: 8.5f}, accuracy: {accu:3.3f} %'.format(
			loss=train_loss, accu=100*train_accu))

		# valid an epoch
		tmp_pred = []
		tmp_tgt = []

		valid_loss, valid_accu = valid_epoch(model, validation_data, test=True)
		print('  - (Validation) loss: {loss: 8.5f}, accuracy: {accu:3.3f} %'.format(
			loss=valid_loss, accu=100*valid_accu))
		with open('san_metrics.txt','a') as outfile:
			outfile.write(' '.join(map(str,tmp_tgt)))
			outfile.write('\n')
			outfile.write(' '.join(map(str,tmp_pred)))
			outfile.write('\n')

		valid_accus += [valid_accu]
		if valid_accu >= max(valid_accus):
			best_score = valid_accu
			# test this model, when valid score is best
			tmp_pred = []
			tmp_tgt = []

			test_loss, test_accu = valid_epoch(model, test_data, test=True)
			print('  - (Test) loss: {loss: 8.5f}, accuracy: {accu:3.3f} %'.format(
		            loss=test_loss, accu=100*test_accu))
			with open('san_metrics_test.txt','w') as outfile:
				outfile.write(' '.join(map(str,tmp_tgt)))
				outfile.write('\n')
				outfile.write(' '.join(map(str,tmp_pred)))
			print('    - [Info] The san_metrics_test file has been updated.')


		# save model
		model_state_dict = model.state_dict()
		checkpoint = {
			'model': model_state_dict,
			'settings': opt,
			'epoch': epoch_i}
		if opt.save_model:
			if opt.save_mode == 'all':
				model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
				torch.save(checkpoint, model_name)
			elif opt.save_mode == 'best':
				model_name = opt.save_model + '.chkpt'
				if valid_accu >= max(valid_accus):
					torch.save(checkpoint, model_name)
					print('    - [Info] The checkpoint file has been updated.')

		# write log (train, valid, time)
		with open(log_train_file, 'a') as log_tf,\
		open(log_valid_file, 'a') as log_vf,\
		open(train_time_file, 'a') as log_ttf:
			log_tf.write('{epoch},{loss: 8.5f},{accu:3.3f}\n'.format(
				epoch=epoch_i, loss=train_loss,
				accu=100*train_accu))
			log_vf.write('{epoch},{loss: 8.5f},{accu:3.3f}\n'.format(
				epoch=epoch_i, loss=valid_loss,
				accu=100*valid_accu))
			log_ttf.write('time per batch in epoch: %f\n'%((end - start)/len(training_data)))

	return best_score


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('-data', type=str, default='data.map')

	parser.add_argument('-epoch', type=int, default=60)
	parser.add_argument('-batch_size', type=int, default=64)
	parser.add_argument('-log', type=str ,default='san')
	parser.add_argument('-save_model', type=str ,default='san')
	parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

	opt = parser.parse_args()

	data = torch.load(opt.data)
	opt.classes = data['settings'].classes
	opt.max_sent_len = data['settings'].max_sent_len

	training_data, validation_data, test_data = prepare_dataloaders(data, opt)

	print(opt)

	n_warmup_steps = (training_data.__len__()/opt.batch_size)*opt.epoch*0.6
	san, optimizer = make_model(classes=opt.classes, n_warmup_steps=n_warmup_steps, h_groups=4)

	train(san, training_data, validation_data, 
		test_data ,optimizer, opt)

