# -*- coding: utf-8 -*-
# @Author: xiegr
# @Date:   2019-08-05 18:47:15
# @Last Modified by:   xiegr
# @Last Modified time: 2019-08-06 14:41:53
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from Optim import ScheduledOptim
import torch.optim as optim
import torch.utils.data

class PositionEncoding(nn.Module):
	"""docstring for """
	def __init__(self, d_dim=256, dropout=0.1, l_byte=1500):
		super(PositionEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

		pe = torch.zeros(l_byte, d_dim)
		# range(0,max,step=1) => [0,1,2,3...max], size is (max,)
		# unsqueeze size is (max,1)
		position = torch.arange(0., l_byte).unsqueeze(1)
		div_term = torch.exp(torch.arange(0., d_dim, 2) *
			-(math.log(10000.0) / d_dim))

		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)

		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
		return self.dropout(x)

class LearnedEmbedding(nn.Module):
	"""docstring for """
	def __init__(self, d_dim=256, byte_range=256, pad_idx=0):
		super(LearnedEmbedding, self).__init__()
		self.lut = nn.Embedding(num_embeddings=byte_range, 
			embedding_dim=d_dim, padding_idx=pad_idx)
		self.d_dim = d_dim

	def forward(self, x):
		return self.lut(x) * math.sqrt(self.d_dim)

class Embedding(nn.Module):
	"""docstring for Embedding"""
	def __init__(self, d_dim=256, l_byte=1500,dropout=0.1,
	 byte_range=256, pad_idx=0):
		super(Embedding, self).__init__()
		self.pos = PositionEncoding(d_dim=d_dim, dropout=dropout, l_byte=l_byte)
		self.em = LearnedEmbedding(d_dim=d_dim, byte_range=byte_range, pad_idx=pad_idx)

	def forward(self, x):
		return self.pos(self.em(x))
#######################################################################################
class SelfAttention(nn.Module):
	"""docstring for SelfAttention"""
	def __init__(self, h_groups=8, d_dim=256, dropout=0.1):
		super(SelfAttention, self).__init__()
		assert d_dim % h_groups == 0

		self.d_k = d_dim // h_groups
		self.h_groups = h_groups
		# 4 part: query, key, value and concat needs projection
		self.linears = nn.ModuleList([nn.Linear(d_dim, d_dim)
		 for _ in range(4)])
		self.attn = None
		self.dropout = nn.Dropout(p=dropout)

	def attention(self, query, key, value, dropout=None):
		scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
		scores = self.dropout(F.softmax(scores, dim = -1))
		return torch.matmul(scores, value)

	def forward(self, query, key, value):
		nbatches = query.size(0)

		# 1) query, key, value shape: Batch * h * l_byte * d_k
		query, key, value = \
		[l(x).view(nbatches, -1, self.h_groups, self.d_k).transpose(1, 2)
		for l, x in zip(self.linears, (query, key, value))]

		# 2) Apply attention on all the projected vectors in batch. 
		x = self.attention(query, key, value)

		# 3) "Concat" using a view and apply a final linear. 
		x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h_groups * self.d_k)
		return self.linears[-1](x)

class SimpleCNN(nn.Module):
	"""docstring for SimpleCNN"""
	def __init__(self, d_dim=256, dropout=0.1):
		super(SimpleCNN, self).__init__()
		self.cnn = nn.Sequential(
			nn.Conv1d(in_channels=d_dim, out_channels=d_dim,
			 kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
			)
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, x):
		x =  self.cnn(x.transpose(-2,-1)).transpose(-2,-1)
		return self.dropout(x)

# class LayerNorm(nn.Module):
# 	def __init__(self, d_dim=256, eps=1e-6):
# 		super(LayerNorm, self).__init__()
# 		self.a_2 = nn.Parameter(torch.ones(d_dim))
# 		self.b_2 = nn.Parameter(torch.zeros(d_dim))
# 		self.eps = eps

# 	def forward(self, x):
# 		mean = x.mean(-1, keepdim=True)
# 		std = x.std(-1, keepdim=True)
# 		return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class Encoder(nn.Module):
	"""docstring for Encoder"""
	def __init__(self, h_groups=8, d_dim=256, dropout=0.1):
		super(Encoder, self).__init__()
		self.atten = SelfAttention(h_groups=h_groups, d_dim=d_dim, dropout=dropout)
		self.cnn = SimpleCNN(d_dim=d_dim, dropout=dropout)
		self.norm = nn.LayerNorm(normalized_shape=d_dim)

	def forward(self, x):
		x = self.norm(x + self.atten(x, x, x))
		return self.norm(x + self.cnn(x))
#######################################################################################

class Classifier_nosoftmax(nn.Module):
	"""docstring for Classifier_nosoftmax"""
	def __init__(self, classes, d_dim=256):
		super(Classifier_nosoftmax, self).__init__()
		self.proj = nn.Linear(d_dim, classes)
	
	def forward(self, x):
		# 在pytorch中若模型使用CrossEntropyLoss这个loss函数，
		# 则不应该在最后一层再使用softmax进行激活。
		return self.proj(x)	

class SAN(nn.Module):
	def __init__(self, classes, n_encoder=2, d_dim=256, dropout=0.1, 
		l_byte=1500, byte_range=256, h_groups=8):
		super(SAN, self).__init__()
		self.encoders = nn.ModuleList(
			[Encoder(h_groups=h_groups, d_dim=d_dim, dropout=dropout) for _ in range(n_encoder)]
			)
		self.embedding = Embedding(d_dim=d_dim, l_byte=l_byte,dropout=dropout,
		 byte_range=byte_range, pad_idx=0)
		self.classifier_nosoftmax = Classifier_nosoftmax(classes=classes, d_dim=d_dim)

	def forward(self, x):
		x = self.embedding(x)
		for encoder in self.encoders:
			x = encoder(x)
		x = x.sum(dim=-2)
		return self.classifier_nosoftmax(x)

class Dataset(torch.utils.data.Dataset):
	"""docstring for Dataset"""
	def __init__(self, src_insts, tgt_insts):
		super(Dataset, self).__init__()
		self.src_insts = src_insts
		self.tgt_insts = tgt_insts

	def __len__(self):
		return len(self.src_insts)

	def __getitem__(self, idx):
		return self.src_insts[idx], self.tgt_insts[idx]

def make_model(classes, n_warmup_steps, n_encoder=2, d_dim=256, dropout=0.1, 
		l_byte=1500, byte_range=256, h_groups=8):
	# we use 40 bytes for classify, but here we define l_byte=1500 just because it is the MTU
	# in fact, Preprocess.py has make every packet to 40 bytes
	model = SAN(classes, n_encoder, d_dim, dropout, 
		l_byte, byte_range, h_groups)
	for p in model.parameters():
		if p.dim() > 1:
			nn.init.xavier_uniform_(p)

	optimizer = ScheduledOptim(optim.Adam(
		filter(lambda x: x.requires_grad, model.parameters()),
	betas=(0.9, 0.98), eps=1e-09
	), 
	d_dim, n_warmup_steps)

	return model.to(torch.device('cuda')), optimizer