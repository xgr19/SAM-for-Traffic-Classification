# -*- coding: utf-8 -*-
# @Author: xiegr
# @Date:   2020-08-30 15:58:51
# @Last Modified by:   xiegr
# @Last Modified time: 2020-09-18 14:22:48
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math


torch.manual_seed(2020)
torch.cuda.manual_seed_all(2020)
np.random.seed(2020)
random.seed(2020)
torch.backends.cudnn.deterministic = True

class SelfAttention(nn.Module):
	"""docstring for SelfAttention"""
	def __init__(self, d_dim=256, dropout=0.1):
		super(SelfAttention, self).__init__()
		# for query, key, value, output
		self.dim = d_dim
		self.linears = nn.ModuleList([nn.Linear(d_dim, d_dim) for _ in range(4)])
		self.dropout = nn.Dropout(p=dropout)

	def attention(self, query, key, value):
		scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.dim)
		scores = F.softmax(scores, dim=-1)
		return scores

	def forward(self, query, key, value):
		# 1) query, key, value
		query, key, value = \
		[l(x) for l, x in zip(self.linears, (query, key, value))]

		# 2) Apply attention
		scores = self.attention(query, key, value)
		x = torch.matmul(scores, value)

		# 3) apply the final linear
		x = self.linears[-1](x.contiguous())
		# sum keepdim=False
		return self.dropout(x), torch.mean(scores, dim=-2)

class OneDimCNN(nn.Module):
	"""docstring for OneDimCNN"""
	# https://blog.csdn.net/sunny_xsc1994/article/details/82969867
	def __init__(self, max_byte_len, d_dim=256, \
		kernel_size = [3, 4], filters=256, dropout=0.1):
		super(OneDimCNN, self).__init__()
		self.kernel_size = kernel_size
		self.convs = nn.ModuleList([
						nn.Sequential(nn.Conv1d(in_channels=d_dim, 
												out_channels=filters, 
												kernel_size=h),
						#nn.BatchNorm1d(num_features=config.feature_size), 
						nn.ReLU(),
						# MaxPool1d: 
						# stride – the stride of the window. Default value is kernel_size
						nn.MaxPool1d(kernel_size=max_byte_len-h+1))
						for h in self.kernel_size
						]
						)
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, x):
		out = [conv(x.transpose(-2,-1)) for conv in self.convs]
		out = torch.cat(out, dim=1)
		out = out.view(-1, out.size(1))
		return self.dropout(out)


class SAM(nn.Module):
	"""docstring for SAM"""
	# total header bytes 24
	def __init__(self, num_class, max_byte_len, kernel_size = [3, 4], \
		d_dim=256, dropout=0.1, filters=256):
		super(SAM, self).__init__()
		self.posembedding = nn.Embedding(num_embeddings=max_byte_len, 
								embedding_dim=d_dim)
		self.byteembedding = nn.Embedding(num_embeddings=300, 
								embedding_dim=d_dim)
		self.attention = SelfAttention(d_dim, dropout)
		self.cnn = OneDimCNN(max_byte_len, d_dim, kernel_size, filters, dropout)
		self.fc = nn.Linear(in_features=256*len(kernel_size),
                            out_features=num_class)

	def forward(self, x, y):
		out = self.byteembedding(x) + self.posembedding(y)
		out, score = self.attention(out, out, out)
		out = self.cnn(out)
		out = self.fc(out)
		if not self.training:
			return F.softmax(out, dim=-1).max(1)[1], score
		return out
		
if __name__ == '__main__':
	x = np.random.randint(0, 255, (10, 20))
	y = np.random.randint(0, 20, (10, 20))
	sam = SAM(num_class=5, max_byte_len=20)
	out = sam(torch.from_numpy(x).long(), torch.from_numpy(y).long())
	print(out[0])

	sam.eval()
	out, score = sam(torch.from_numpy(x).long(), torch.from_numpy(y).long())
	print(out[0], score[0])
