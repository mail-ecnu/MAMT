import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_remaining_self_loops, degree
from torch_geometric.data import Data, DataLoader
import numpy as np

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, ccs):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        weighted_norm = ccs * norm

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=weighted_norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j


class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features, affine=False):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.affine = affine
        self.log_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if self.affine:
            self.bias = nn.Parameter(torch.Tensor(out_features,))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.log_weight)

    def forward(self, input):
        if self.affine:
            return F.linear(input, self.log_weight.exp(), self.bias)
        else:
            return F.linear(input, self.log_weight.exp())


class NegativeLinear(nn.Module):
    def __init__(self, in_features, out_features, affine=False):
        super(NegativeLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.affine = affine
        self.log_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if self.affine:
            self.bias = nn.Parameter(torch.Tensor(out_features,))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.log_weight)

    def forward(self, input):
        if self.affine:
            return F.linear(input, -self.log_weight.exp(), self.bias)
        else:
            return F.linear(input, -self.log_weight.exp())


class TRAN(nn.Module):
    """
    Trust region assignment network
    """
    def __init__(self, sa_sizes, hidden_dim=32, norm_in=True, sparse=0.05):
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per 
                                           agent
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
        """
        super(TRAN, self).__init__()
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)
        self.sparse = sparse

        self.k_sa_encoders = nn.ModuleList()
        self.k_tr_encoders = nn.ModuleList()
        self.k_encoders = nn.ModuleList()
        self.k_decoders = nn.ModuleList()

        self.lambda_sa_encoders = nn.ModuleList()
        self.lambda_tr_encoders = nn.ModuleList()
        self.lambda_encoders = nn.ModuleList()

        # iterate over agents
        for sdim, adim in sa_sizes:
            idim = sdim + adim
            odim = hidden_dim

            # k network
            k_sa_encoder = nn.Sequential()
            if norm_in:
                k_sa_encoder.add_module('enc_bn', nn.BatchNorm1d(
                     idim, affine=False))
            k_sa_encoder.add_module('enc_fc1', nn.Linear(idim, hidden_dim))
            k_sa_encoder.add_module('enc_nl', nn.LeakyReLU())
            self.k_sa_encoders.append(k_sa_encoder)

            k_tr_encoder = nn.Sequential()
            # k_tr_encoder.add_module('enc_fc1', NegativeLinear(
            #     1, hidden_dim, affine=True))
            k_tr_encoder.add_module('enc_fc1', nn.Linear(
                1, hidden_dim, bias=True))
            k_tr_encoder.add_module('enc_nl1', nn.LeakyReLU())
            # k_tr_encoder.add_module('enc_fc2', NegativeLinear(
            #     hidden_dim, hidden_dim, affine=False))
            k_tr_encoder.add_module('enc_fc2', nn.Linear(
                hidden_dim, hidden_dim, bias=False))
            k_tr_encoder.add_module('enc_nl2', nn.LeakyReLU())
            self.k_tr_encoders.append(k_tr_encoder)

            k_encoder = nn.Sequential()
            # k_encoder.add_module('enc_fc1', NegativeLinear(
            #     hidden_dim*2, hidden_dim, affine=True))
            k_encoder.add_module('enc_fc1', nn.Linear(
                hidden_dim*2, hidden_dim, bias=True))
            k_encoder.add_module('enc_nl', nn.LeakyReLU())
            self.k_encoders.append(k_encoder)

            k_decoder = nn.Sequential()
            # k_decoder.add_module('dec_fc1', NegativeLinear(
            #     hidden_dim, hidden_dim, affine=True))
            k_decoder.add_module('dec_fc1', nn.Linear(
                hidden_dim, hidden_dim, bias=True))
            k_decoder.add_module('dec_nl1', nn.LeakyReLU())
            # k_decoder.add_module('dec_fc2', NegativeLinear(
            #     hidden_dim, 1, affine=False))
            k_decoder.add_module('dec_fc2', nn.Linear(
                hidden_dim, 1, bias=False))
            k_decoder.add_module('dec_nl2', nn.LeakyReLU())
            self.k_decoders.append(k_decoder)

            # lambda network
            l_sa_encoder = nn.Sequential()
            if norm_in:
                l_sa_encoder.add_module('enc_bn', nn.BatchNorm1d(
                     idim, affine=False))
            l_sa_encoder.add_module('enc_fc1', nn.Linear(idim, hidden_dim))
            l_sa_encoder.add_module('enc_nl', nn.LeakyReLU())
            self.lambda_sa_encoders.append(l_sa_encoder)

            l_tr_encoder = nn.Sequential()
            # l_tr_encoder.add_module('enc_fc1', PositiveLinear(
            #     1, hidden_dim, affine=True))
            l_tr_encoder.add_module('enc_fc1', nn.Linear(
                1, hidden_dim, bias=True))
            l_tr_encoder.add_module('enc_nl1', nn.LeakyReLU())
            # l_tr_encoder.add_module('enc_fc2', PositiveLinear(
            #     hidden_dim, hidden_dim, affine=False))
            l_tr_encoder.add_module('enc_fc2', nn.Linear(
                hidden_dim, hidden_dim, bias=False))
            l_tr_encoder.add_module('enc_nl2', nn.LeakyReLU())
            self.lambda_tr_encoders.append(l_tr_encoder)

            l_encoder = nn.Sequential()
            # l_encoder.add_module('enc_fc1', PositiveLinear(
            #     hidden_dim*2, hidden_dim, affine=True))
            l_encoder.add_module('enc_fc1', nn.Linear(
                hidden_dim*2, hidden_dim, bias=True))
            l_encoder.add_module('enc_nl1', nn.LeakyReLU())
            # l_encoder.add_module('enc_fc2', PositiveLinear(
            #     hidden_dim, 1, affine=False))
            l_encoder.add_module('enc_fc2', nn.Linear(
                hidden_dim, 1, bias=False))
            l_encoder.add_module('enc_nl2', nn.LeakyReLU())
            self.lambda_encoders.append(l_encoder)

        self.gcn = GCNConv(hidden_dim, hidden_dim)

    def forward(self, inps, trs, ccs):
        """
        trs: n_agents
        ccs: batch_size x n_agents x n_agents
        """
        states = [s for s, a in inps]
        actions = [a for s, a in inps]
        inps = [torch.cat((s, a), dim=1) for s, a in inps]
        batch_size = len(states[0])
        n_agents = len(states)
        
        ccs_old = ccs.clone()
        ccs = torch.where(ccs >= self.sparse,
                            torch.tensor(1.).cuda(),
                            torch.tensor(0.).cuda())
        edge_indexes = [torch.nonzero(cc.fill_diagonal_(1.0)) for cc in ccs]
        ccs_indexes = torch.cat(
            [ccs_old[i][idx[:,0],idx[:,1]] \
                for i, idx in enumerate(edge_indexes)])

        # shape: n_agents x batch_size x 1
        batch_trs = torch.stack(trs).unsqueeze(0).repeat(
            batch_size, 1).t().unsqueeze(-1).cuda()
        # Step 1: extract state-action [k]-encoding for each agent
        k_sa_encodings = [encoder(inp) for encoder, inp in zip(
            self.k_sa_encoders, inps)]
        # Step 2: extract trust region [k]-encoding for each agent
        k_tr_encodings = [encoder(tr) for encoder, tr in zip(
            self.k_tr_encoders, batch_trs)]
        # Step 3: extract [k]-encoding for each agent
        k_enc_inps = [torch.cat((k_sa_enc, k_tr_enc), dim=1) \
                        for k_sa_enc, k_tr_enc in zip(k_sa_encodings, k_tr_encodings)]
        k_encodings = torch.cat([encoder(enc_inp) for encoder, enc_inp in zip(
            self.k_encoders, k_enc_inps)]).view(
                n_agents, batch_size, -1).permute(1, 0, 2)
        # Step 4: extract [k]-embedding based on gnn
        data_list = [Data(x=k_encodings[i],
                          edge_index=edge_indexes[i].t().contiguous()) \
                              for i in range(batch_size)]
        loader = DataLoader(data_list, batch_size=batch_size)
        k_embeddings = []
        for batch in loader:
            k_embed = self.gcn(batch.x, batch.edge_index, ccs_indexes).view(
                batch_size, n_agents, -1).permute(1, 0, 2)
            k_embeddings.append(k_embed)
        # Step 5: extract [k] based on gnn
        k_embeddings = k_embeddings[0]
        k = torch.stack([decoder(k_embed) for decoder, k_embed in zip(
            self.k_decoders, k_embeddings)]).permute(1, 0, 2).squeeze()
        # Step 6: extract state-action [l]-encoding for each agent
        lambda_sa_encodings = [encoder(inp) for encoder, inp in zip(
            self.lambda_sa_encoders, inps)]
        # Step 7: extract trust region [l]-encoding for each agent
        lambda_tr_encodings = [encoder(tr) for encoder, tr in zip(
            self.lambda_tr_encoders, batch_trs)]
        # Step 8: extract [l]
        lambda_enc_inps = [torch.cat((l_sa_enc, l_tr_enc), dim=1) \
                        for l_sa_enc, l_tr_enc in zip(
                            lambda_sa_encodings, lambda_tr_encodings)]
        lamb = torch.cat([encoder(enc_inp) for encoder, enc_inp in zip(
            self.lambda_encoders, lambda_enc_inps)]).view(
                n_agents, batch_size, -1).permute(1, 0, 2).squeeze()

        return lamb * k

            
