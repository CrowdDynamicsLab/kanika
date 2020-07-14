from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


def repackage_hidden(h):
	"""Wraps hidden states in new Variables, to detach them from their history."""
	if type(h) == Variable:
		return Variable(h.data)
	else:
		return tuple(repackage_hidden(v) for v in h)

def evaluate_GCN(gcn_model, test_loader, Adj, Adj2, Adj3, Adj4, Adj5, Features, rowsum, X, epoch):
	correct = 0
	total = 0
	for i, (X_test_1, Y_test_1) in enumerate(test_loader):
		X_test = torch.tensor(X[X["QuestionId"].isin(X_test_1)]['PairId'].values)
		Y_test = torch.tensor(X[X["QuestionId"].isin(Y_test_1)]['Credible'].values)
		gcn_model.eval()
		user_gcn_embed,user_gcn_embed2,user_gcn_embed3,user_gcn_embed4 = gcn_model(Features, Adj, Adj2, Adj3, Adj4, Adj5)
		user_gcn_embed.squeeze_()
		user_gcn_embed2.squeeze_()
		user_gcn_embed3.squeeze_()
		user_gcn_embed4.squeeze_()
		predicted = torch.max(user_gcn_embed[X_test], 1)[1].data
		predicted2 = torch.max(user_gcn_embed2[X_test], 1)[1].data
		predicted3 = torch.max(user_gcn_embed3[X_test], 1)[1].data
		predicted4 = torch.max(user_gcn_embed4[X_test], 1)[1].data

	indexes = np.where(np.not_equal(Y_test,predicted))

	acc = accuracy_score( Y_test, predicted)
	acc2 = accuracy_score( Y_test, predicted2)
	acc3 = accuracy_score( Y_test, predicted3)
	acc4 = accuracy_score( Y_test, predicted4)

	return acc, acc2, acc3 , acc4

def evaluate_basic(gcn_model, test_loader, Adj, Adj2, Adj3, Adj4, Adj5, Adj6, Features, rowsum, X):
	correct = 0
	total = 0
	#QNList = []
	for i, (X_test_1, Y_test_1) in enumerate(test_loader):
		X_test = torch.tensor(X[X["QuestionId"].isin(X_test_1)]['PairId'].values)
		Y_test = torch.tensor(X[X["QuestionId"].isin(Y_test_1)]['Credible'].values)
		#for i in X_test_1:
		#	QNList.append(float(len(X[X["QuestionId"]==i])))
		gcn_model.eval()
		user_gcn_embed = gcn_model(Features, Adj, Adj2, Adj3, Adj4, Adj5, Adj6)
		user_gcn_embed.squeeze_()
		predicted = user_gcn_embed[X_test]>0
	#print "random choose(theoretical accuracy):",sum([(i-2+2/i) for i in QNList])/sum(QNList)
	#indexes = np.where(np.not_equal(Y_test,predicted))

	acc = accuracy_score( Y_test.cpu(), predicted.cpu())

	return acc

def evaluate_relational(gcn_model, test_loader, Adj, Adj2, Adj3, Adj4, Adj5, Features, rowsum, X, epoch):
	correct = 0
	total = 0
	#QNList = []
	user = []
	for i, (X_test_1, Y_test_1) in enumerate(test_loader):
		X_test = torch.tensor(X[X["QuestionId"].isin(X_test_1)]['PairId'].values)
		Y_test = torch.tensor(X[X["QuestionId"].isin(Y_test_1)]['Credible'].values)
		#for i in X_test_1:
		#	QNList.append(float(len(X[X["QuestionId"]==i])))
		gcn_model.eval()
		user_gcn_embed = gcn_model(Features, Adj, Adj2, Adj3, Adj4, Adj5)

		predicted = user_gcn_embed.squeeze_()[X_test]>0
	#print "random choose(theoretical accuracy):",sum([(i-2+2/i) for i in QNList])/sum(QNList)

		acc = accuracy_score( Y_test, predicted.cpu())

	return acc

def evaluate_adaboost(gcn_model, test_loader, Adj, Adj2, Adj3, Adj4, Adj5, Features, rowsum, X, epoch, Target):
	correct = 0
	total = 0
	for i, (X_test_1, Y_test_1) in enumerate(test_loader):
		X_test = torch.tensor(X[X["QuestionId"].isin(X_test_1)]['PairId'].values)
		Y_test = torch.tensor(X[X["QuestionId"].isin(Y_test_1)]['Credible'].values)
		gcn_model.eval()
		_,_,user_gcn_embed2= gcn_model(Features, Adj, Adj2, Adj3, Adj4, Adj5, Target, X_test)
		user_gcn_embed2.squeeze_()
		predicted2 = user_gcn_embed2[X_test]>0
		#predicted2 = torch.max(user_gcn_embed2[X_test], 1)[1].data
	acc = accuracy_score(Y_test, predicted2.cpu())

	return acc

def evaluate_adaboost_content(gcn_model, test_loader, Adj, Adj2, Adj3, Adj4, Adj5, Adj6, Features, rowsum, X, epoch, Target):
	correct = 0
	total = 0
	for i, (X_test_1, Y_test_1) in enumerate(test_loader):
		X_test = torch.tensor(X[X["QuestionId"].isin(X_test_1)]['PairId'].values)
		Y_test = torch.tensor(X[X["QuestionId"].isin(Y_test_1)]['Credible'].values)
		gcn_model.eval()
		_,_, _, user_gcn_embed2= gcn_model(Features, Adj, Adj2, Adj3, Adj4, Adj5, Adj6, Target, X_test)
		user_gcn_embed2.squeeze_()
		predicted2 = user_gcn_embed2[X_test]>0
		#predicted2 = torch.max(user_gcn_embed2[X_test], 1)[1].data
	acc = accuracy_score(Y_test, predicted2.cpu())

	return acc

def evaluate_WWW(gcn_model, test_loader, Adj, Adj2, Adj3, Adj4, Adj5, Features, rowsum, X, epoch):
	correct = 0
	total = 0
	for i, (X_test_1, Y_test_1) in enumerate(test_loader):
		X_test = torch.tensor(X[X["QuestionId"].isin(X_test_1)]['PairId'].values)
		Y_test = torch.tensor(X[X["QuestionId"].isin(Y_test_1)]['Credible'].values)
		gcn_model.eval()
		user_gcn_embed,user_gcn_embed2,user_gcn_embed3,user_gcn_embed4 = gcn_model(Features, Adj, Adj2, Adj3, Adj4, Adj5)
		user_gcn_embed.squeeze_()
		user_gcn_embed2.squeeze_()
		user_gcn_embed3.squeeze_()
		user_gcn_embed4.squeeze_()
		predicted = user_gcn_embed[X_test]>0
		predicted2 = user_gcn_embed2[X_test]>0
		predicted3 = user_gcn_embed3[X_test]>0
		predicted4 = user_gcn_embed4[X_test]>0


	acc = accuracy_score( Y_test, predicted.cpu())
	acc2 = accuracy_score( Y_test, predicted2.cpu())
	acc3 = accuracy_score( Y_test, predicted3.cpu())
	acc4 = accuracy_score( Y_test, predicted4.cpu())

	return acc, acc2, acc3 , acc4

def evaluate_line_decay_rep2(gcn_model, test_loader, Adj, Adj2, Adj3, Adj4, Adj5, Features, rowsum, X, epoch):
	correct = 0
	total = 0
	for i, (X_test_1, Y_test_1) in enumerate(test_loader):
		X_test = torch.tensor(X[X["QuestionId"].isin(X_test_1)]['PairId'].values)
		Y_test = torch.tensor(X[X["QuestionId"].isin(Y_test_1)]['Credible'].values)
		gcn_model.eval()
		user_gcn_embed,user_gcn_embed2,user_gcn_embed3,user_gcn_embed4,user_gcn_embed5,user_gcn_embed6 = gcn_model(Features, Adj, Adj2, Adj3, Adj4, Adj5)
		user_gcn_embed.squeeze_()
		user_gcn_embed2.squeeze_()
		user_gcn_embed3.squeeze_()
		user_gcn_embed4.squeeze_()
		user_gcn_embed5.squeeze_()
		user_gcn_embed6.squeeze_()
		predicted = torch.max(user_gcn_embed[X_test], 1)[1].data
		predicted2 = torch.max(user_gcn_embed2[X_test], 1)[1].data
		predicted3 = torch.max(user_gcn_embed3[X_test], 1)[1].data
		predicted4 = torch.max(user_gcn_embed4[X_test], 1)[1].data
		predicted5 = torch.max(user_gcn_embed5[X_test], 1)[1].data
		predicted6 = torch.max(user_gcn_embed6[X_test], 1)[1].data
		"""
		file = open("contrastive","w")
		for i in predicted4.tolist():
			file.write(str(i)+'\n')
		file.close()
		file = open("Asimilarity","w")
		for i in predicted2.tolist():
			file.write(str(i)+'\n')
		file.close()
		file = open("truth.txt","w")
		for i in Y_test.tolist():
			file.write(str(i)+'\n')
		file.close()
		file = open("identity.txt","w")
		for i in predicted.tolist():
			file.write(str(i)+'\n')
		file.close()
		file = open("TSsimilarity.txt","w")
		for i in predicted3.tolist():
			file.write(str(i)+'\n')
		file.close()
		"""

	indexes = np.where(np.not_equal(Y_test,predicted))

	acc = accuracy_score( Y_test, predicted)
	acc2 = accuracy_score( Y_test, predicted2)
	acc3 = accuracy_score( Y_test, predicted3)
	acc4 = accuracy_score( Y_test, predicted4)
	acc5 = accuracy_score( Y_test, predicted5)
	acc6 = accuracy_score( Y_test, predicted6)

	return acc, acc2, acc3 , acc4, acc5, acc6

def evaluate_batch_random_gcn(rnn, gcn, gcn_item, num_items, adj, adj_item, Users, batch_size, SG, Users_all, Items_all):

	correct = 0
	hidden = rnn.init_hidden(batch_size)
	gcn.train(False)
	gcn_item.train(False)
	rnn.train(False)

	for i, sequences in enumerate(batchify(SG.test_sequences, Users, batch_size)):
		input_variable,target_variable, input_lengths, target_lengths, users_batch, items_batch, sorted_targets = sequences
		#print input_variable.data, target_variable.data
		if adj.is_cuda:
			input_variable = input_variable.cuda()
			hidden = hidden.cuda()

		user_embedding = gcn(Users_all, adj, users_batch)
		item_embedding = gcn_item(Items_all, adj_item, items_batch)
		if adj.is_cuda:
			output, _ = rnn(input_variable, hidden, input_lengths, user_embedding, item_embedding, users_batch)
		else:
			output, _ = rnn(input_variable, hidden, input_lengths, user_embedding.cpu(), item_embedding.cpu(), users_batch)

		output = output.view(max(input_lengths), batch_size, output.shape[1])
		hidden = repackage_hidden(hidden)

		for j in range(len(input_lengths)):
			items = [k for k in SG.test_negatives[Users[users_batch[j]]] if k <num_items]
			target = sorted_targets[j][-1]
			items.append(target)
			output_flat = output[input_lengths[j]-1, j, :]
			output_flat = output_flat[items]
			topv, topi = output_flat.data.topk(1)


			predicted = items[topi[0]]
			if predicted == target:
				correct+=1

	print ("Total correct predictions for users", correct, correct/float(len(SG.Users)))
	return correct
