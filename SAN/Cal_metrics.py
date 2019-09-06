# -*- coding: utf-8 -*-
# @Author: xiegr
# @Date:   2019-08-05 16:29:34
# @Last Modified by:   xiegr
# @Last Modified time: 2019-08-16 16:01:20
import sklearn.metrics as skm

def caculate(y_true, y_pred, class_num):
	print( "class%d Recall Precision f1_score %f %f %f"\
		%( class_num, skm.recall_score(y_true, y_pred), skm.precision_score(y_true, y_pred), \
			skm.f1_score(y_true, y_pred)))

def read_metrics(cls, file):
	with open(file) as f:
		all_lines = f.readlines()
		y_tgt = list(map(int, all_lines[0].split()))
		y_pred = list(map(int, all_lines[1].split()))

		for x in range(cls):
			a,b = [],[]
			for y in range(len(y_tgt)):
				a.append(int(y_tgt[y]==x))
				b.append(int(y_pred[y]==x))

			caculate(a,b,x)

if __name__ == '__main__':
	cls = 9
	file = 'san_metrics_test.txt'
	# file = 'san_metrics.txt'
	read_metrics(cls, file)