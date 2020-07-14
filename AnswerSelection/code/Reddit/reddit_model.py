import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.utils
from torch.nn.parameter import Parameter
from layers import GraphConvolution, SparseMM
import numpy as np
from scipy import sparse
import math
import os
USE_CUDA = False

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class GCN_decay_rep2(nn.Module):
    def __init__(self, nFeat, nhid1, nhid2, nhid3, nhid4, nclass, dropout=0.5):
        super(GCN_decay_rep2, self).__init__()

        self.gc1 = GraphConvolution(nFeat, nhid2, bias=True)
        self.gc2 = GraphConvolution(nhid2, nhid3, bias=True)
        self.gc3 = GraphConvolution(nhid3, nhid4, bias=True)
        self.gc4 = GraphConvolution(nFeat, nhid2, bias=True)
        self.gc5 = GraphConvolution(nhid2, nhid3, bias=True)
        self.gc6 = GraphConvolution(nhid3, nhid4, bias=True)
        self.gc7 = GraphConvolution(nFeat, nhid2, bias=True)
        self.gc8 = GraphConvolution(nhid2, nhid3, bias=True)
        self.gc9 = GraphConvolution(nhid3, nhid4, bias=True)
        self.gc10 = GraphConvolution(nFeat, nhid2, bias=True)
        self.gc11 = GraphConvolution(nhid2, nhid3, bias=True)
        self.gc12 = GraphConvolution(nhid3, nhid4, bias=True)
        self.sim_dense = nn.Linear(2*nhid4, nhid4, bias=True)
        self.ensemble1 = nn.Linear(3*nhid4, nhid2, bias=True)
        self.ensemble2 = nn.Linear(nhid2, nhid3, bias=True)
        self.ensemble3 = nn.Linear(nhid3, nhid4, bias=True)


        self.dropout = dropout
        self.dense1 = nn.Linear(nhid4, nclass, bias=1)
        self.dense2 = nn.Linear(nhid4, nclass, bias=1)
        self.dense3 = nn.Linear(nhid4, nclass, bias=1)
        self.densesim = nn.Linear(nhid4, nclass, bias=1)
        self.dense4 = nn.Linear(nhid4, nclass, bias=1)


    def forward(self, x, adj1, adj2, adj3, adj4, adj5):
        # adj1: Identity
        # adj2: something we try before
        # adj3: Arrival-similarity
        # adj4: TrueSkill-similarity
        # adj5: Contrastive GCN

        # Identity
        x1 = F.relu(self.gc1(x, adj5))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x1 = F.relu(self.gc2(x1, adj5))
        x1 = self.gc3(x1, adj5)
        x1_dense = F.relu(x1)
        x1_dense = self.dense1(x1_dense)

        # TS-Similarity
        x2 = F.relu(self.gc4(x, adj4))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x2 = F.relu(self.gc5(x2, adj4))
        x2 = self.gc6(x2, adj4)
        x2_dense = F.relu(x2)
        x2_dense = self.dense2(x2_dense)

        # Arrival-Similarity
        x3 = F.relu(self.gc7(x, adj3))
        x3 = F.dropout(x3, self.dropout, training=self.training)
        x3 = F.relu(self.gc8(x3, adj3))
        x3 = self.gc9(x3, adj3)
        x3_dense = F.relu(x3)
        x3_dense = self.dense3(x3_dense)

        # Similarity dense
        xsim = self.sim_dense(torch.cat((x2,x3),1))
        xsim_dense = F.relu(xsim)
        xsim_dense = self.densesim(xsim_dense)

        # Contrastive
        x4 = F.relu(self.gc10(x, adj1))
        x4 = F.dropout(x4, self.dropout, training=self.training)
        x4 = F.relu(self.gc11(x4, adj1))
        x4 = self.gc12(x4, adj1)
        x4_dense = F.relu(x4)
        x4_dense = self.dense3(x4_dense)

        # Ensemble
        x_ensemble = F.relu(self.ensemble1(torch.cat((x1,xsim,x4),1)))
        x_ensemble = F.dropout(x_ensemble, self.dropout, training=self.training)
        x_ensemble = F.relu(self.ensemble2(x_ensemble))
        #x4 = F.dropout(x4, self.dropout, training=self.training)
        x_ensemble = F.relu(self.ensemble3(x_ensemble))
        x_ensemble = self.dense4(x_ensemble)

        #x4 = F.relu(self.gc10(x, adj5))
        #x4 = F.dropout(x4, self.dropout, training=self.training)
        #x4 = F.relu(self.gc11(x4, adj5))
        #x4 = F.relu(self.gc12(x4, adj5))
        #x4 = self.dense4(x4)

        #weight = torch.nn.functional.softmax(self.dense4(x),1)
        #x4 = torch.einsum('ij,ij->ij', (weight[:,0].view(-1,1),x1))+torch.einsum('ij,ij->ij', (weight[:,1].view(-1,1),x2))+torch.einsum('ij,ij->ij', (weight[:,2].view(-1,1),x3))

        return x1_dense, x2_dense, x3_dense, x4_dense, xsim_dense, x_ensemble

class GCN_decay_rep1(nn.Module):
    def __init__(self, nFeat, nhid1, nhid2, nhid3, nhid4, nclass, dropout=0.5):
        super(GCN_decay_rep1, self).__init__()

        self.gc1 = GraphConvolution(nFeat, nhid2, bias=True)
        self.gc2 = GraphConvolution(nhid2, nhid3, bias=True)
        self.gc3 = GraphConvolution(nhid3, nhid4, bias=True)
        self.gc4 = GraphConvolution(nFeat, nhid2, bias=True)
        self.gc5 = GraphConvolution(nhid2, nhid3, bias=True)
        self.gc6 = GraphConvolution(nhid3, nhid4, bias=True)
        self.gc7 = GraphConvolution(nFeat, nhid2, bias=True)
        self.gc8 = GraphConvolution(nhid2, nhid3, bias=True)
        self.gc9 = GraphConvolution(nhid3, nhid4, bias=True)
        self.gc10 = GraphConvolution(nFeat, nhid2, bias=True)
        self.gc11 = GraphConvolution(nhid2, nhid3, bias=True)
        self.gc12 = GraphConvolution(nhid3, nhid4, bias=True)

        self.dropout = dropout
        self.dense1 = nn.Linear(nhid4, nclass, bias=1)
        self.dense2 = nn.Linear(nhid4, nclass, bias=1)
        self.dense3 = nn.Linear(nhid4, nclass, bias=1)
        self.dense4 = nn.Linear(nhid4, nclass, bias=1)
        self.dense5 = nn.Linear(nhid4, nclass, bias=1)

        self.ensemble1 = nn.Linear(4*nhid4, nhid2, bias=True)
        self.ensemble2 = nn.Linear(nhid2, nhid3, bias=True)
        self.ensemble3 = nn.Linear(nhid3, nhid4, bias=True)


    def forward(self, x, adj1, adj2, adj3, adj4, adj5):
        # adj1: Identity
        # adj2: something we try before
        # adj3: Arrival-similarity
        # adj4: TrueSkill-similarity
        # adj5: Contrastive GCN

        # Identity
        x1 = F.relu(self.gc1(x, adj5))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x1 = F.relu(self.gc2(x1, adj5))
        x1 = self.gc3(x1, adj5)
        x1_dense = F.relu(x1)
        x1_dense = self.dense1(x1_dense)

        # TS-Similarity
        x2 = F.relu(self.gc4(x, adj4))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x2 = F.relu(self.gc5(x2, adj4))
        x2 = self.gc6(x2, adj4)
        x2_dense = F.relu(x2)
        x2_dense = self.dense2(x2_dense)

        # A-Similarity
        x3 = F.relu(self.gc7(x, adj3))
        x3 = F.dropout(x3, self.dropout, training=self.training)
        x3 = F.relu(self.gc8(x3, adj3))
        x3 = self.gc9(x3, adj3)
        x3_dense = F.relu(x3)
        x3_dense = self.dense3(x3_dense)

        # Contrastive
        x4 = F.relu(self.gc10(x, adj1))
        x4 = F.dropout(x4, self.dropout, training=self.training)
        x4 = F.relu(self.gc11(x4, adj1))
        x4 = self.gc12(x4, adj1)
        x4_dense = F.relu(x4)
        x4_dense = self.dense4(x4_dense)

        # Ensemble
        x_ensemble = F.relu(self.ensemble1(torch.cat((x1,x2,x3,x4),1)))
        x_ensemble = F.dropout(x_ensemble, self.dropout, training=self.training)
        x_ensemble = F.relu(self.ensemble2(x_ensemble))
        x_ensemble = F.relu(self.ensemble3(x_ensemble))
        x_ensemble = self.dense5(x_ensemble)

        #x4 = F.relu(self.gc10(x, adj5))
        #x4 = F.dropout(x4, self.dropout, training=self.training)
        #x4 = F.relu(self.gc11(x4, adj5))
        #x4 = F.relu(self.gc12(x4, adj5))
        #x4 = self.dense4(x4)

        #weight = torch.nn.functional.softmax(self.dense4(x),1)
        #x4 = torch.einsum('ij,ij->ij', (weight[:,0].view(-1,1),x1))+torch.einsum('ij,ij->ij', (weight[:,1].view(-1,1),x2))+torch.einsum('ij,ij->ij', (weight[:,2].view(-1,1),x3))

        return x1_dense, x2_dense, x3_dense, x4_dense, x_ensemble

class GCN_WWW(nn.Module):
    def __init__(self, nFeat, nhid1, nhid2, nhid3, nhid4, nclass, dropout=0.5):
        super(GCN_WWW, self).__init__()

        self.gc1 = GraphConvolution(nFeat, nhid2, bias=True)
        self.gc2 = GraphConvolution(nhid2, nhid3, bias=True)
        self.gc3 = GraphConvolution(nhid3, nhid4, bias=True)
        self.dropout = dropout
        self.dense1 = nn.Linear(nhid4, nclass, bias=1)


    def forward(self, x, adj1, adj2, adj3, adj4, adj5):
        # adj1: Identity
        # adj2: something we try before
        # adj3: Arrival-similarity
        # adj4: TrueSkill-similarity
        # adj5: Contrastive GCN

        # Identity
        x1 = F.relu(self.gc1(x, adj5))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x1 = F.relu(self.gc2(x1, adj5))
        x1 = self.gc3(x1, adj5)
        x1_dense = F.relu(x1)
        x1_dense = self.dense1(x1_dense)

        # TS-Similarity
        x2 = F.relu(self.gc1(x, adj4))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x2 = F.relu(self.gc2(x2, adj4))
        x2 = self.gc3(x2, adj4)
        x2_dense = F.relu(x2)
        x2_dense = self.dense1(x2_dense)

        # A-Similarity
        x3 = F.relu(self.gc1(x, adj3))
        x3 = F.dropout(x3, self.dropout, training=self.training)
        x3 = F.relu(self.gc2(x3, adj3))
        x3 = self.gc3(x3, adj3)
        x3_dense = F.relu(x3)
        x3_dense = self.dense1(x3_dense)

        # Contrastive
        x4 = F.relu(self.gc1(x, adj1))
        x4 = F.dropout(x4, self.dropout, training=self.training)
        x4 = F.relu(self.gc2(x4, adj1))
        x4 = self.gc3(x4, adj1)
        x4_dense = F.relu(x4)
        x4_dense = self.dense1(x4_dense)


        return x1_dense, x2_dense, x3_dense, x4_dense

class GCN_individual(nn.Module):
    def __init__(self, nFeat, nhid1, nhid2, nhid3, nhid4, nclass, dropout=0.5):
        super(GCN_individual, self).__init__()

        self.gc1 = GraphConvolution(nFeat, nhid2, bias=True)
        self.gc2 = GraphConvolution(nhid2, nhid3, bias=True)
        self.gc3 = GraphConvolution(nhid3, nhid4, bias=True)
        self.gc4 = GraphConvolution(nFeat, nhid2, bias=True)
        self.gc5 = GraphConvolution(nhid2, nhid3, bias=True)
        self.gc6 = GraphConvolution(nhid3, nhid4, bias=True)
        self.gc7 = GraphConvolution(nFeat, nhid2, bias=True)
        self.gc8 = GraphConvolution(nhid2, nhid3, bias=True)
        self.gc9 = GraphConvolution(nhid3, nhid4, bias=True)
        self.gc10 = GraphConvolution(nFeat, nhid2, bias=True)
        self.gc11 = GraphConvolution(nhid2, nhid3, bias=True)
        self.gc12 = GraphConvolution(nhid3, nhid4, bias=True)

        self.dropout = dropout
        self.dense1 = nn.Linear(nhid4, nclass, bias=1)
        self.dense2 = nn.Linear(nhid4, nclass, bias=1)
        self.dense3 = nn.Linear(nhid4, nclass, bias=1)
        self.dense4 = nn.Linear(nhid4, nclass, bias=1)


    def forward(self, x, adj1, adj2, adj3, adj4, adj5):
        # adj1: Identity
        # adj2: something we try before
        # adj3: Arrival-similarity
        # adj4: TrueSkill-similarity
        # adj5: Contrastive GCN

        # Identity
        x1 = F.relu(self.gc1(x, adj5))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x1 = F.relu(self.gc2(x1, adj5))
        x1 = self.gc3(x1, adj5)
        x1_dense = F.relu(x1)
        x1_dense = self.dense1(x1_dense)

        # TS-Similarity
        x2 = F.relu(self.gc4(x, adj4))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x2 = F.relu(self.gc5(x2, adj4))
        x2 = self.gc6(x2, adj4)
        x2_dense = F.relu(x2)
        x2_dense = self.dense2(x2_dense)

        # A-Similarity
        x3 = F.relu(self.gc7(x, adj3))
        x3 = F.dropout(x3, self.dropout, training=self.training)
        x3 = F.relu(self.gc8(x3, adj3))
        x3 = self.gc9(x3, adj3)
        x3_dense = F.relu(x3)
        x3_dense = self.dense3(x3_dense)

        # Contrastive
        x4 = F.relu(self.gc10(x, adj1))
        x4 = F.dropout(x4, self.dropout, training=self.training)
        x4 = F.relu(self.gc11(x4, adj1))
        x4 = self.gc12(x4, adj1)
        x4_dense = F.relu(x4)
        x4_dense = self.dense4(x4_dense)

        return x1_dense, x2_dense, x3_dense , x4_dense

class GCN_relational(nn.Module):
    def __init__(self, nFeat, nhid1, nhid2, nhid3, nhid4, nclass, dropout=0.5):
        super(GCN_relational, self).__init__()

        self.gc1 = GraphConvolution(nFeat, nhid2, bias=True)
        self.gc2 = GraphConvolution(nhid2, nhid3, bias=True)
        self.gc3 = GraphConvolution(nhid3, nclass, bias=True)
        self.gc4 = GraphConvolution(nFeat, nhid2, bias=True)
        self.gc5 = GraphConvolution(nhid2, nhid3, bias=True)
        self.gc6 = GraphConvolution(nhid3, nclass, bias=True)
        self.gc7 = GraphConvolution(nFeat, nhid2, bias=True)
        self.gc8 = GraphConvolution(nhid2, nhid3, bias=True)
        self.gc9 = GraphConvolution(nhid3, nclass, bias=True)
        self.gc10 = GraphConvolution(nFeat, nhid2, bias=True)
        self.gc11 = GraphConvolution(nhid2, nhid3, bias=True)
        self.gc12 = GraphConvolution(nhid3, nclass, bias=True)

        self.dropout = dropout
        self.dense1 = nn.Linear(nhid4, nclass, bias=1)
        self.dense2 = nn.Linear(nhid4, nclass, bias=1)
        self.dense3 = nn.Linear(nhid4, nclass, bias=1)
        self.dense4 = nn.Linear(nhid4, nclass, bias=1)


    def forward(self, x, adj1, adj2, adj3, adj4, adj5):
        # adj1: Identity
        # adj2: something we try before
        # adj3: Arrival-similarity
        # adj4: TrueSkill-similarity
        # adj5: Contrastive GCN

        # Identity
        x11 = self.gc1(x, adj1)
        x21 = self.gc4(x, adj4)
        x31 = self.gc7(x, adj3)
        x41 = self.gc10(x, adj5)
        temp = F.relu(x11+x21+x31+x41)

        x12 = self.gc2(temp, adj1)
        x22 = self.gc5(temp, adj4)
        x32 = self.gc8(temp, adj3)
        x42 = self.gc11(temp, adj5)
        temp = F.relu(x12+x22+x32+x42)

        x13 = self.gc3(temp, adj1)
        x23 = self.gc6(temp, adj4)
        x33 = self.gc9(temp, adj3)
        x43 = self.gc12(temp, adj5)
        temp = (x13+x23+x33+x43)

        return temp

class GCN_basic(nn.Module):
    def __init__(self, nFeat, nhid1, nhid2, nhid3, nhid4, nclass, dropout=0.5):
        super(GCN_basic, self).__init__()

        self.gc1 = GraphConvolution(nFeat, nhid2, bias=True)
        self.gc2 = GraphConvolution(nhid2, nhid3, bias=True)
        self.gc3 = GraphConvolution(nhid3, nhid4, bias=True)

        self.dropout = dropout
        self.dense = nn.Linear(nhid4, nclass, bias=1)


    def forward(self, x, adj1, adj2, adj3, adj4, adj5):
        # adj5: Identity
        # adj2: something we try before
        # adj3: Arrival-similarity
        # adj4: TrueSkill-similarity
        # adj1: Contrastive GCN
        x1 = F.relu(self.gc1(x, adj4))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x1 = F.relu(self.gc2(x1, adj4))
        x1 = F.relu(self.gc3(x1, adj4))
        x1_dense = self.dense(x1)

        return x1_dense

class GCN_adaboost(nn.Module):
    def __init__(self, nFeat, nhid1, nhid2, nhid3, nhid4, nclass, dropout=0.5):
        super(GCN_adaboost, self).__init__()

        self.gc1 = GraphConvolution(nFeat, nhid2, bias=True)
        self.gc2 = GraphConvolution(nhid2, nhid3, bias=True)
        self.gc3 = GraphConvolution(nhid3, nhid4, bias=True)
        self.gc4 = GraphConvolution(nFeat, nhid2, bias=True)
        self.gc5 = GraphConvolution(nhid2, nhid3, bias=True)
        self.gc6 = GraphConvolution(nhid3, nhid4, bias=True)
        #self.gc7 = GraphConvolution(nFeat, nhid2, bias=True)
        #self.gc8 = GraphConvolution(nhid2, nhid3, bias=True)
        #self.gc9 = GraphConvolution(nhid3, nhid4, bias=True)
        self.gc10 = GraphConvolution(nFeat, nhid2, bias=True)
        self.gc11 = GraphConvolution(nhid2, nhid3, bias=True)
        self.gc12 = GraphConvolution(nhid3, nhid4, bias=True)


        self.dropout = dropout
        self.dense1 = nn.Linear(nhid4, nclass, bias=1)
        self.dense2 = nn.Linear(nhid4, nclass, bias=1)
        self.dense3 = nn.Linear(nhid4, nclass, bias=1)
        self.dense4 = nn.Linear(nhid4, nclass, bias=1)
        self.simdense = nn.Linear(2*nhid4, nclass, bias=1)


    def forward(self, x, adj1, adj2, adj3, adj4, adj5, y, index):
        # adj1: Identity
        # adj2: something we try before
        # adj3: Arrival-similarity
        # adj4: TrueSkill-similarity
        # adj5: Contrastive GCN

        # Identity
        x1 = F.relu(self.gc1(x, adj5))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x1 = F.relu(self.gc2(x1, adj5))
        x1 = self.gc3(x1, adj5)
        x1_dense = F.relu(x1)
        x1_dense = self.dense1(x1_dense)

        # TS-Similarity
        x2 = F.relu(self.gc4(x, adj4))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x2 = F.relu(self.gc5(x2, adj4))
        x2 = self.gc6(x2, adj4)
        x2_dense = F.relu(x2)
        x2_dense = self.dense2(x2_dense)

        # A-Similarity
        x3 = F.relu(self.gc4(x, adj3))
        x3 = F.dropout(x3, self.dropout, training=self.training)
        x3 = F.relu(self.gc5(x3, adj3))
        x3 = self.gc6(x3, adj3)
        x3_dense = F.relu(x3)
        x3_dense = self.dense3(x3_dense)

        sim_dense = self.simdense(torch.cat((x2,x3),1))
        """
        #alpha1, sub-adaboost
        temp1 = torch.exp(-torch.mul(x2_dense[index],y[index]))
        temp2 = torch.mul(x3_dense[index],y[index])
        sum1 = torch.sum(torch.masked_select(temp1,temp2.ge(0)))
        sum2 = torch.sum(temp1) - sum1
        alpha1 = 0.5*torch.log(torch.div(sum2,sum1))

        sim_dense = torch.add(x2_dense,torch.mul(x3_dense,alpha1))
        """
        # Contrastive
        x4 = F.relu(self.gc10(x, adj1))
        x4 = F.dropout(x4, self.dropout, training=self.training)
        x4 = F.relu(self.gc11(x4, adj1))
        x4 = self.gc12(x4, adj1)
        x4_dense = F.relu(x4)
        x4_dense = self.dense4(x4_dense)

        #alpha2, top-adaboost
        temp3 = torch.exp(-torch.mul(x4_dense[index],y[index]))
        temp4 = torch.mul(sim_dense[index],y[index])
        sum3 = torch.sum(torch.masked_select(temp3,temp4.ge(0)))
        sum4 = torch.sum(temp3) - sum3
        alpha2 = 0.5*torch.log(torch.div(sum4,sum3))

        part1_dense = torch.add(x4_dense,torch.mul(sim_dense,alpha2))

        #alpha3, top-adaboost
        temp5 = torch.exp(-torch.mul(part1_dense[index],y[index]))
        temp6 = torch.mul(x1_dense[index],y[index])
        sum5 = torch.sum(torch.masked_select(temp5,temp6.ge(0)))
        sum6 = torch.sum(temp5) - sum5
        alpha3 = 0.5*torch.log(torch.div(sum6,sum5))

        part2_dense = torch.add(part1_dense,torch.mul(x1_dense,alpha3))


        return x2_dense,x3_dense, part2_dense
