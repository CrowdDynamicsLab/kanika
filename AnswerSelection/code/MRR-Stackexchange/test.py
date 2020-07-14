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

def evaluate_basic(gcn_model, test_loader, Adj, Adj2, Adj3, Adj4, Adj5, Features, rowsum, X, epoch):
	correct = 0
	total = 0
	#QNList = []
	MRR = []
	random = [1]
	gcn_model.eval()
	user_gcn_embed = gcn_model(Features, Adj, Adj2, Adj3, Adj4, Adj5)
	user_gcn_embed.squeeze_()
	for i, (X_test_1, Y_test_1) in enumerate(test_loader):
		temp = torch.tensor(X[X["QuestionId"]==X_test_1.cpu().numpy().tolist()[0]][['PairId','Credible']].values)
		X_test = temp[:,0]
		Y_test = temp[:,1]
		#Y_test = torch.tensor(X[X["QuestionId"]==Y_test_1.numpy().tolist()[0]]['Credible'].values)
		#for i in X_test_1:
		#	QNList.append(float(len(X[X["QuestionId"]==i])))

		result = user_gcn_embed[X_test].detach().cpu().numpy()
		groundT = Y_test.cpu().numpy().tolist().index(1)
		#print(groundT)
		#print(str(groundT) + "Adaagagagaga: "+ str(i))
		temp = result.argsort()
		#print(temp)
		ranks = np.empty_like(temp)

		ranks[temp] = len(result) - np.arange(len(result))
		#print(ranks)
		#print(groundT)
		#print(1.0/ranks[groundT])
		MRR.append(1.0/ranks[groundT])
		temp = []
		for i in range(len(result)):
			temp.append(1.0/(i+1))
	#print(MRR)
	return sum(MRR)/len(MRR), sum(random)/len(random)

def evaluate_relational(gcn_model, test_loader, Adj, Adj2, Adj3, Adj4, Adj5, Features, rowsum, X, epoch):
	correct = 0
	total = 0
	#QNList = []
	MRR = []
	random = [1]
	gcn_model.eval()
	user_gcn_embed = gcn_model(Features, Adj, Adj2, Adj3, Adj4, Adj5)
	user_gcn_embed.squeeze_()
	for i, (X_test_1, Y_test_1) in enumerate(test_loader):
		temp = torch.tensor(X[X["QuestionId"]==X_test_1.numpy().tolist()[0]][['PairId','Credible']].values)
		X_test = temp[:,0]
		Y_test = temp[:,1]
		#for i in X_test_1:
		#	QNList.append(float(len(X[X["QuestionId"]==i])))

		result = user_gcn_embed[X_test].detach().cpu().numpy()
		groundT = Y_test.numpy().tolist().index(1)
		temp = result.argsort()
		ranks = np.empty_like(temp)
		ranks[temp] = len(result) - np.arange(len(result))
		MRR.append(1.0/ranks[groundT])
		temp = []
		for i in range(len(result)):
			temp.append(1.0/(i+1))

	return sum(MRR)/len(MRR), sum(random)/len(random)

def evaluate_adaboost(gcn_model, test_loader, Adj, Adj2, Adj3, Adj4, Adj5, Features, rowsum, X, epoch, Target):
	correct = 0
	total = 0
	MRR = []
	random = [1]
	gcn_model.eval()
	temp2 = []
	for i, (X_test_1, Y_test_1) in enumerate(test_loader):
		temp2.append(X_test_1.cpu().numpy().tolist()[0])
	X_test = torch.tensor(X[X["QuestionId"].isin(temp2)]['PairId'].values).cuda()
	_,_,user_gcn_embed = gcn_model(Features, Adj, Adj2, Adj3, Adj4, Adj5, Target, X_test)
	user_gcn_embed.squeeze_()
	for i, (X_test_1, Y_test_1) in enumerate(test_loader):
		temp = torch.tensor(X[X["QuestionId"]==X_test_1.cpu().numpy().tolist()[0]][['PairId','Credible']].values)
		X_test = temp[:,0]
		Y_test = temp[:,1]
		#for i in X_test_1:
		#	QNList.append(float(len(X[X["QuestionId"]==i])))

		result = user_gcn_embed[X_test].detach().cpu().numpy()
		groundT = Y_test.numpy().tolist().index(1)
		temp = result.argsort()
		ranks = np.empty_like(temp)
		ranks[temp] = len(result) - np.arange(len(result))
		MRR.append(1.0/ranks[groundT])
		temp = []
		for i in range(len(result)):
			temp.append(1.0/(i+1))

	return sum(MRR)/len(MRR), sum(random)/len(random)

def evaluate_WWW(gcn_model, test_loader, Adj, Adj2, Adj3, Adj4, Adj5, Features, rowsum, X, epoch):
	correct = 0
	total = 0
	MRR = []
	random = [1]
	gcn_model.eval()
	user_gcn_embed,_,_,_ = gcn_model(Features, Adj, Adj2, Adj3, Adj4, Adj5)
	user_gcn_embed.squeeze_()
	for i, (X_test_1, Y_test_1) in enumerate(test_loader):
		temp = torch.tensor(X[X["QuestionId"]==X_test_1.numpy().tolist()[0]][['PairId','Credible']].values)
		X_test = temp[:,0]
		Y_test = temp[:,1]
		#for i in X_test_1:
		#	QNList.append(float(len(X[X["QuestionId"]==i])))

		result = user_gcn_embed[X_test].detach().cpu().numpy()
		groundT = Y_test.numpy().tolist().index(1)
		temp = result.argsort()
		ranks = np.empty_like(temp)
		ranks[temp] = len(result) - np.arange(len(result))
		MRR.append(1.0/ranks[groundT])
		temp = []
		for i in range(len(result)):
			temp.append(1.0/(i+1))

	return sum(MRR)/len(MRR), sum(random)/len(random)

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
		file.close()qsub -I -l nodes=nano4:ppn=1,walltime=15600
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

	print "Total correct predictions for users", correct, correct/float(len(SG.Users))
	return correct
