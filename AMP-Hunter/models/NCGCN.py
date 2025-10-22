import torch
from torch.nn import Dropout, Parameter, Softmax, Sigmoid
from torch.nn.init import xavier_uniform_, constant_, xavier_uniform_, calculate_gain
from torch_geometric.nn import GCNConv,Linear
import torch.nn.functional as F
from torch_sparse import SparseTensor, fill_diag, matmul, mul, spspmm, remove_diag
from torch_sparse import sum as sparsesum


class NCGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, params):
        super().__init__()
        self.W1L = Parameter(torch.empty(num_features, params.hidden))
        self.W1H = Parameter(torch.empty(num_features, params.hidden))
        self.W2L = Parameter(torch.empty(params.hidden, params.hidden))
        self.W2H = Parameter(torch.empty(params.hidden, params.hidden))

        self.lam = Parameter(torch.zeros(3))
        self.lam1 = Parameter(torch.zeros(2))
        self.lam2 = Parameter(torch.zeros(2))
        self.dropout = Dropout(p=params.dp1)
        self.dropout2 = Dropout(p=params.dp2)
        self.finaldp = Dropout(p=0.5)
        self.act = F.relu

        self.WX = Parameter(torch.empty(num_features, params.hidden))
        #for regression num_classes = 1
        self.lin1 = Linear(params.hidden, num_classes)

        self.args = params
        self._cached_adj_l = None
        self._cached_adj_h = None
        self._cached_adj_l_l = None
        self._cached_adj_h_h = None
        # self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        constant_(self.lam, 0)
        constant_(self.lam1, 0)
        constant_(self.lam2, 0)
        xavier_uniform_(self.W1L, gain=calculate_gain('relu'))
        xavier_uniform_(self.W1H, gain=calculate_gain('relu'))
        xavier_uniform_(self.W2L, gain=calculate_gain('relu'))
        xavier_uniform_(self.W2H, gain=calculate_gain('relu'))
        xavier_uniform_(self.WX, gain=calculate_gain('relu'))


    def agg_norm(self, adj_t, mask, mtype='target'):
        # TODO: A^2
        if mtype == 'target':
            A_tilde = mul(adj_t,mask.view(-1,1))
        elif mtype == 'source':
            A_tilde = mul(adj_t,mask.view(1,-1))
        else:
            A_tilde = SparseTensor.from_torch_sparse_coo_tensor(
                torch.sparse.mm(
                    mask, torch.sparse.mm(
                        mask, adj_t.to_torch_sparse_coo_tensor())))
        if self.args.addself:
            A_tilde = fill_diag(A_tilde, 1.)
        else:
            A_tilde = remove_diag(A_tilde)
        D_tilde = sparsesum(A_tilde, dim=1)
        D_tilde_sq = D_tilde.pow_(-0.5)
        D_tilde_sq.masked_fill_(D_tilde_sq == float('inf'), 0.)
        A_hat = mul(A_tilde, D_tilde_sq.view(-1, 1))
        A_hat = mul(A_hat, D_tilde_sq.view(1, -1))

        # A_hat = A_hat.to_torch_sparse_coo_tensor()
        return A_hat

    def forward(self, data):
        x = data.x
        cc_mask = data.cc_mask
        rev_cc_mask = torch.ones_like(cc_mask) - cc_mask
        edge_index = data.edge_index.to('cuda')
        adj_t = SparseTensor(row=edge_index[1], col=edge_index[0])

        # low_cc mask
        if data.update_cc:
            A_hat_l = self.agg_norm(adj_t, cc_mask, 'target')
            self._cached_adj_l = A_hat_l
            A_hat_l_l = self.agg_norm(adj_t, cc_mask, 'source')
            self._cached_adj_l_l = A_hat_l_l
        else:
            A_hat_l = self._cached_adj_l
            A_hat_l_l = self._cached_adj_l_l

        # high_cc mask
        if data.update_cc:
            A_hat_h = self.agg_norm(adj_t, rev_cc_mask, 'target')
            self._cached_adj_h = A_hat_h
            A_hat_h_h = self.agg_norm(adj_t, rev_cc_mask, 'source')
            self._cached_adj_h_h = A_hat_h_h
        else:
            A_hat_h = self._cached_adj_h
            A_hat_h_h = self._cached_adj_h_h

        xl = matmul(A_hat_l, x)
        xl = matmul(xl, self.W1L)
        xl = self.act(xl)
        xl = self.dropout(xl)
        xl = torch.mm(matmul(A_hat_l_l, xl), self.W2L)
        # high_cc partion
        xh = matmul(A_hat_h, x)
        xh = matmul(xh, self.W1H)
        xh = self.act(xh)
        xh = self.dropout2(xh)
        xh = torch.mm(matmul(A_hat_h_h, xh), self.W2H)

        x = matmul(x, self.WX)

        lamxl, laml = Softmax()(self.lam1)
        lamxh, lamh = Softmax()(self.lam2)
        lamx= lamxl * cc_mask + lamxh * rev_cc_mask
        xf = lamx.view(-1,1) * x + laml * xl + lamh * xh
        xf = self.act(xf)
        xf = self.finaldp(xf)

        self.embeddings = xf.detach() 
        
        xf = self.lin1(xf)

        return xf

    def get_embeddings(self, data):
        with torch.no_grad():
            _ = self.forward(data)
        return self.embeddings