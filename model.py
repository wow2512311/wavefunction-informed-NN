import torch.nn as nn
import torch
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import AvgPooling, MaxPooling

from typing import Optional
import dgl.function as fn
import numpy as np
# from dgl.nn.functional import edge_softmax

from torch.nn import init
from collections.abc import Mapping

class PureGNN(nn.Module):
    def __init__(self, in_dim=92, hidden_dim=256, output=1, 
                 num_layers=6, conv_type='alignn'):
        super().__init__()
        self.atom_embedding = MLPLayer(in_dim, hidden_dim)
        self.edge_embedding = nn.Sequential(RBFExpansion(vmin=0,vmax=8.0,bins=80),
                                            MLPLayer(80, hidden_dim),
                                            MLPLayer(hidden_dim, hidden_dim))
        
        self.conv_layers = nn.Sequential()
        if conv_type == 'alignn':
            for i in range(num_layers):
                self.conv_layers.append(EdgeGatedGraphConv(hidden_dim,hidden_dim))
        elif conv_type == 'gcn':
            for i in range(num_layers):
                self.conv_layers.append(GraphConv_Edge_Residual(hidden_dim,hidden_dim))
        elif conv_type == 'graphsage':
            for i in range(num_layers):
                self.conv_layers.append(SAGEConv_Edge_Residual(hidden_dim,hidden_dim,'mean'))
        elif conv_type == 'cgcnn':
            for i in range(num_layers):
                self.conv_layers.append(CGCNNConv(hidden_dim,hidden_dim))
        else:
            raise ValueError('Wrong conv_type')
        self.avgpool = AvgPooling()
        self.out_layer1 = MLPLayer(hidden_dim, hidden_dim)
        self.out_layer2 = MLPLayer(hidden_dim, output)#!!!!!!!!!!!!!
        
    def forward(self, g):
        g = g.local_var()
        x = g.ndata.pop("atom_features")
        x = self.atom_embedding(x)
        
        bondlength = torch.norm(g.edata.pop("r"), dim=1)
        y = self.edge_embedding(bondlength)
        
        #print(len(self.conv_layers))
        for conv_layer in self.conv_layers:
            x, y = conv_layer(g, x, y)
        #print(x)
        x = self.avgpool(g, x)
        #print(x.shape)
        x = self.out_layer1(x)
        #x = self.out_layer2(x)
        #print(x.shape)
        return x

class GNNwithCNN(nn.Module):
    def __init__(self, hidden_dim=256, output=1):
        super().__init__()
        self.cnn_conv0_1 = nn.Sequential(nn.Conv2d(2,16,(1,1)), nn.BatchNorm2d(16), nn.LeakyReLU())  #!!!!
        
        self.cnn_conv1 = nn.Sequential(nn.Conv2d(16,16,(3,3),stride=1,padding=1),
                                       nn.BatchNorm2d(16),
                                       nn.LeakyReLU(),   #!!!!
                                       nn.Conv2d(16,16,(3,3),stride=1,padding=1),
                                       nn.BatchNorm2d(16),
                                       nn.LeakyReLU())   #!!!!
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2)
        
        self.cnn_conv0_2 = nn.Sequential(nn.Conv2d(16,32,(1,1)), nn.BatchNorm2d(32), nn.LeakyReLU())  #!!!!
        
        self.cnn_conv2 = nn.Sequential(nn.Conv2d(32,32,(3,3),stride=1,padding=1),
                                       nn.BatchNorm2d(32),
                                       nn.LeakyReLU(),  #!!!!
                                       nn.Conv2d(32,32,(3,3),stride=1,padding=1),
                                       nn.BatchNorm2d(32),
                                       nn.LeakyReLU())  #!!!!
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2)
        
        self.cnn_conv0_3 = nn.Sequential(nn.Conv2d(32,64,(1,1)), nn.BatchNorm2d(64), nn.LeakyReLU())  #!!!!
        
        self.cnn_conv3 = nn.Sequential(nn.Conv2d(64,64,(3,3),stride=1,padding=1),
                                       nn.BatchNorm2d(64),
                                       nn.LeakyReLU(),  #!!!!
                                       nn.Conv2d(64,64,(3,3),stride=1,padding=1),
                                       nn.BatchNorm2d(64),
                                       nn.LeakyReLU())  #!!!!
        self.maxpool_3 = nn.MaxPool2d(kernel_size=2)

        self.cnn_conv0_4 = nn.Sequential(nn.Conv2d(64,128,(1,1)), nn.BatchNorm2d(128), nn.LeakyReLU())  #!!!!
        
        self.cnn_conv4 = nn.Sequential(nn.Conv2d(128,128,(3,3),stride=1,padding=1),
                                       nn.BatchNorm2d(128),
                                       nn.LeakyReLU(),  #!!!!
                                       nn.Conv2d(128,128,(3,3),stride=1,padding=1),
                                       nn.BatchNorm2d(128),
                                       nn.LeakyReLU())  #!!!!
        self.maxpool_4 = nn.MaxPool2d(kernel_size=2)
        
        self.cnn_conv0_5 = nn.Sequential(nn.Conv2d(128,256,(1,1)), nn.BatchNorm2d(256), nn.LeakyReLU())  #!!!!
        
        self.maxpool = nn.AvgPool2d(kernel_size=6)
        #AvgPool2d
        self.bn14 = nn.BatchNorm1d([256]) 
        
        self.wave_embedding = MLPLayer(hidden_dim, hidden_dim)
        
        self.out_layer1 = MLPLayer(hidden_dim * 2, output)
        #self.out_layer2 = MLPLayer(hidden_dim * 2, output)
        
    def forward(self, puregnn, g, total_map):
        g = g.local_var()
        topo_features = puregnn(g)

        wave_features = self.cnn_conv0_1(total_map)  #16*16*100*100
        resdual1 = wave_features
        
        wave_features = self.cnn_conv1(wave_features) + resdual1
        wave_features = self.maxpool_1(wave_features)
        
        wave_features = self.cnn_conv0_2(wave_features)
        resdual2 = wave_features

        wave_features = self.cnn_conv2(wave_features) + resdual2
        wave_features = self.maxpool_2(wave_features)
        
        wave_features = self.cnn_conv0_3(wave_features)
        resdual3 = wave_features

        wave_features = self.cnn_conv3(wave_features) + resdual3
        wave_features = self.maxpool_3(wave_features)
        
        wave_features = self.cnn_conv0_4(wave_features)
        resdual4 = wave_features

        wave_features = self.cnn_conv4(wave_features) + resdual4
        wave_features = self.maxpool_4(wave_features)
        
        wave_features = self.cnn_conv0_5(wave_features)
        #resdual5 = wave_features

        #print(wave_features.shape)        torch.Size([16, 256, 6, 6])
        wave_features = self.maxpool(wave_features).squeeze(2).squeeze(2)
        #print(wave_features.shape)        torch.Size([16, 256, 1, 1])
        
        wave_features = self.bn14(wave_features)
        wave_features = self.wave_embedding(wave_features)

        total_features = torch.cat((topo_features, wave_features), dim=1)
        output = self.out_layer1(total_features)
        #output = self.out_layer2(self.out_layer1(total_features))
        # (resdual1,resdual2,resdual3,resdual4,resdual5)
        return output

class WaveFunctionModel(nn.Module):
    #for VB  /  CB  descriptor
    def __init__(self, in_dim=92, hidden_dim=256, output=10000, 
                 num_layers=7, conv_type='alignn'):
        super().__init__()
        self.atom_embedding = MLPLayer(in_dim, hidden_dim)
        self.edge_embedding = nn.Sequential(RBFExpansion(vmin=0,vmax=8.0,bins=80),
                                            MLPLayer(80, 64),
                                            MLPLayer(64, hidden_dim))
        
        self.conv_layers = nn.Sequential()
        if conv_type == 'alignn':
            for i in range(num_layers):
                self.conv_layers.append(EdgeGatedGraphConv(hidden_dim,hidden_dim))
        elif conv_type == 'gcn':
            for i in range(num_layers):
                self.conv_layers.append(GraphConv_Edge_Residual(hidden_dim,hidden_dim))
        elif conv_type == 'graphsage':
            for i in range(num_layers):
                self.conv_layers.append(SAGEConv_Edge_Residual(hidden_dim,hidden_dim,'mean'))
        elif conv_type == 'cgcnn':
            for i in range(num_layers):
                self.conv_layers.append(CGCNNConv(hidden_dim,hidden_dim))
        else:
            raise ValueError('Wrong conv_type')
        #self.avgpool = AvgPooling()
        self.out_layer1 = MLPLayer(hidden_dim, hidden_dim * 4)
        self.out_layer2 = MLPLayer(hidden_dim * 4, hidden_dim * 8)
        self.out_layer3 = MLPLayer(hidden_dim * 8, output)

    def forward(self, g):
        g = g.local_var()
        x = g.ndata.pop("atom_features")
        x = self.atom_embedding(x)
        
        bondlength = torch.norm(g.edata.pop("r"), dim=1)
        y = self.edge_embedding(bondlength)
        
        #print(len(self.conv_layers))
        for conv_layer in self.conv_layers:
            x, y = conv_layer(g, x, y)
        #print(x)
        #x = self.avgpool(g, x)
        x = self.out_layer1(x)
        x = self.out_layer2(x)
        x = self.out_layer3(x)
 
        return x.reshape(-1,100,100)

class RBFExpansion(nn.Module):
    def __init__(
        self, 
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 40,
        lengthscale: Optional[float] = None
    ):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer("centers", torch.linspace(self.vmin, self.vmax, self.bins))
        if lengthscale is None:
            self.lengthscale = np.diff(self.centers).mean()
            #The first difference is given by out[i] = a[i+1] - a[i] along the given axis
            self.gamma = 1 / self.lengthscale
        else:
            self.lengthscale = lengthscale
            self.gamma = 1/ (lengthscale ** 2)
    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma * (distance.unsqueeze(1) - self.centers) ** 2)

class EdgeGatedGraphConv(nn.Module):
    """Edge gated graph convolution from arxiv:1711.07553.

    see also arxiv:2003.0098.

    This is similar to CGCNN, but edge features only go into
    the soft attention / edge gating function, and the primary
    node update function is W cat(u, v) + b
    """
    def __init__(
        self, input_features: int, output_features: int, residual: bool = True
    ):
        """Initialize parameters for ALIGNN update."""
        super().__init__()
        self.residual = residual
        # CGCNN-Conv operates on augmented edge features
        # z_ij = cat(v_i, v_j, u_ij)
        # m_ij = σ(z_ij W_f + b_f) ⊙ g_s(z_ij W_s + b_s)
        # coalesce parameters for W_f and W_s
        # but -- split them up along feature dimension
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.bn_edges = nn.BatchNorm1d(output_features)

        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.bn_nodes = nn.BatchNorm1d(output_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Edge-gated graph convolution.

        h_i^l+1 = ReLU(U h_i + sum_{j->i} eta_{ij} ⊙ V h_j)
        """
        g = g.local_var()

        # instead of concatenating (u || v || e) and applying one weight matrix
        # split the weight matrix into three, apply, then sum
        # see https://docs.dgl.ai/guide/message-efficient.html
        # but split them on feature dimensions to update u, v, e separately
        # m = BatchNorm(Linear(cat(u, v, e)))

        # compute edge updates, equivalent to:
        # Softplus(Linear(u || v || e))
        g.ndata["e_src"] = self.src_gate(node_feats)
        g.ndata["e_dst"] = self.dst_gate(node_feats)
        g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
        m = g.edata.pop("e_nodes") + self.edge_gate(edge_feats)

        g.edata["sigma"] = torch.sigmoid(m)
        g.ndata["Bh"] = self.dst_update(node_feats) #Wdst * Hj
        g.update_all(
            fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h")
        )
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        g.ndata["h"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6) #formula 3
        x = self.src_update(node_feats) + g.ndata.pop("h") #formula 2

        # softmax version seems to perform slightly worse
        # that the sigmoid-gated version
        # compute node updates
        # Linear(u) + edge_gates ⊙ Linear(v)
        # g.edata["gate"] = edge_softmax(g, y)
        # g.ndata["h_dst"] = self.dst_update(node_feats)
        # g.update_all(fn.u_mul_e("h_dst", "gate", "m"), fn.sum("m", "h"))
        # x = self.src_update(node_feats) + g.ndata.pop("h")

        # node and edge updates
        #x = F.silu(x)
        #y = F.silu(m)
        x = F.silu(self.bn_nodes(x))
        y = F.silu(self.bn_edges(m))
        if self.residual:
            x = node_feats + x
            y = edge_feats + y

        return x, y

class MLPLayer(nn.Module):
    """Multilayer perceptron layer helper."""

    def __init__(self, in_features: int, out_features: int):
        """Linear, Batchnorm, SiLU layer."""
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
        )

    def forward(self, x):
        """Linear, Batchnorm, silu layer."""
        return self.layer(x)

class CGCNNConv(nn.Module):
    def __init__(
        self,
        node_features,
        edge_features,
        return_messages: bool = True,
    ):
        super().__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.return_messages = return_messages

        self.linear_src = nn.Linear(node_features, 2 * node_features)
        self.linear_dst = nn.Linear(node_features, 2 * node_features)
        self.linear_edge = nn.Linear(edge_features, 2 * node_features)
        self.bn_message = nn.BatchNorm1d(2 * node_features)

        self.bn = nn.BatchNorm1d(node_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        """CGCNN convolution defined in Eq 5.

        10.1103/PhysRevLett.120.14530
        """
        g = g.local_var()

        g.ndata["h_src"] = self.linear_src(node_feats)
        g.ndata["h_dst"] = self.linear_dst(node_feats)
        g.apply_edges(fn.u_add_v("h_src", "h_dst", "h_nodes"))
        m = g.edata.pop("h_nodes") + self.linear_edge(edge_feats)
        m = self.bn_message(m)

        # split messages into W_f and W_s terms
        # multiply output of atom interaction net and edge attention net
        # i.e. compute the term inside the summation in eq 5
        # σ(z_ij W_f + b_f) ⊙ g_s(z_ij W_s + b_s)
        h_f, h_s = torch.chunk(m, 2, dim=1)
        m = torch.sigmoid(h_f) * F.softplus(h_s)
        g.edata["m"] = m

        # apply the convolution term in eq. 5 (without residual connection)
        # storing the results in edge features `h`
        g.update_all(
            message_func=fn.copy_e("m", "z"),
            reduce_func=fn.sum("z", "h"),
        )

        # final batchnorm
        h = self.bn(g.ndata.pop("h"))

        # residual connection plus nonlinearity
        out = F.softplus(node_feats + h)

        if self.return_messages:
            return out, m

        return out

def expand_as_pair(input_, g=None):
    if isinstance(input_, tuple):
        return input_
    elif g is not None and g.is_block:
        if isinstance(input_, Mapping):
            input_dst = {
                k: F.narrow_row(v, 0, g.number_of_dst_nodes(k))
                for k, v in input_.items()}
        else:
            input_dst = F.narrow_row(input_, 0, g.number_of_dst_nodes())
        return input_, input_dst
    else:
        return input_, input_

class GraphConv_Edge_Residual(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm='both',
                 residual = True,
                 weight=True,
                 bias=True,
                 activation=None,
                 allow_zero_in_degree=False):
        super().__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self.residual = residual
        if weight:
            self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self._activation = activation
    
        self.src_gate = nn.Linear(in_feats, out_feats)
        self.dst_gate = nn.Linear(in_feats, out_feats)
        self.edge_gate = nn.Linear(in_feats, out_feats)
    def reset_parameters(self):
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, node_feats, edge_feats, weight=None, edge_weight=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    pass
            graph.ndata["e_src"] = self.src_gate(node_feats)
            graph.ndata["e_dst"] = self.dst_gate(node_feats)
            graph.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
            m = graph.edata.pop("e_nodes") + self.edge_gate(edge_feats)
            graph.edata["sigma"] = torch.sigmoid(m)
            
            aggregate_fn = fn.u_mul_e('h', 'sigma', 'm')
            
            feat_src, feat_dst = expand_as_pair(node_feats, graph)
            if self._norm in ['left', 'both']:
                degs = graph.out_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            if weight is not None:
                if self.weight is not None:
                    pass
            else:
                weight = self.weight

            if self._in_feats > self._out_feats:
                # mult W first to reduce the feature size for aggregation.
                if weight is not None:
                    feat_src = torch.matmul(feat_src, weight)
                graph.srcdata['h'] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
            else:
                # aggregate first then mult W
                graph.srcdata['h'] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
                if weight is not None:
                    rst = torch.matmul(rst, weight)

            if self._norm in ['right', 'both']:
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            if self.residual:
                rst = node_feats + rst
            return rst, m

    def extra_repr(self):
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)

class SAGEConv_Edge_Residual(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 feat_drop=0.,
                 residual = True,
                 bias=True,
                 norm=None,
                 activation=None):
        super(SAGEConv_Edge_Residual, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        self.residual = residual
        self.src_gate = nn.Linear(in_feats, out_feats)
        self.dst_gate = nn.Linear(in_feats, out_feats)
        self.edge_gate = nn.Linear(in_feats, out_feats)

        # aggregator type: mean/pool/lstm/gcn
        if aggregator_type == 'pool':
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        if aggregator_type == 'lstm':
            self.lstm = nn.LSTM(self._in_src_feats, self._in_src_feats, batch_first=True)
        if aggregator_type != 'gcn':
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=False)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        if bias:
            self.bias = nn.parameter.Parameter(torch.zeros(self._out_feats))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if self._aggre_type == 'pool':
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == 'lstm':
            self.lstm.reset_parameters()
        if self._aggre_type != 'gcn':
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def _compatibility_check(self):
        """Address the backward compatibility issue brought by #2747"""
        if not hasattr(self, 'bias'):
            pass
            bias = self.fc_neigh.bias
            self.fc_neigh.bias = None
            if hasattr(self, 'fc_self'):
                if bias is not None:
                    bias = bias + self.fc_self.bias
                    self.fc_self.bias = None
            self.bias = bias

    def _lstm_reducer(self, nodes):
        m = nodes.mailbox['m'] # (B, L, D)
        batch_size = m.shape[0]
        h = (m.new_zeros((1, batch_size, self._in_src_feats)),
             m.new_zeros((1, batch_size, self._in_src_feats)))
        _, (rst, _) = self.lstm(m, h)
        return {'neigh': rst.squeeze(0)}

    def forward(self, graph, node_feats, edge_feats):
        self._compatibility_check()
        with graph.local_scope():
            graph.ndata["e_src"] = self.src_gate(node_feats)
            graph.ndata["e_dst"] = self.dst_gate(node_feats)
            graph.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
            m = graph.edata.pop("e_nodes") + self.edge_gate(edge_feats)
            graph.edata["sigma"] = torch.sigmoid(m)
            
            msg_fn = fn.u_mul_e('h', 'sigma', 'm')

            if isinstance(node_feats, tuple):
                feat_src = self.feat_drop(node_feats[0])
                feat_dst = self.feat_drop(node_feats[1])
            else:
                feat_src = feat_dst = self.feat_drop(node_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]

            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.number_of_edges() == 0:
                graph.dstdata['neigh'] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats).to(feat_dst)

            # Determine whether to apply linear transformation before message passing A(XW)
            lin_before_mp = self._in_src_feats > self._out_feats

            # Message Passing
            if self._aggre_type == 'mean':
                graph.srcdata['h'] = self.fc_neigh(feat_src) if lin_before_mp else feat_src
                graph.update_all(msg_fn, fn.mean('m', 'neigh'))
                h_neigh = graph.dstdata['neigh']
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)

            elif self._aggre_type == 'gcn':
                graph.srcdata['h'] = self.fc_neigh(feat_src) if lin_before_mp else feat_src
                if isinstance(node_feats, tuple):  # heterogeneous
                    graph.dstdata['h'] = self.fc_neigh(feat_dst) if lin_before_mp else feat_dst
                else:
                    if graph.is_block:
                        graph.dstdata['h'] = graph.srcdata['h'][:graph.num_dst_nodes()]
                    else:
                        graph.dstdata['h'] = graph.srcdata['h']
                graph.update_all(msg_fn, fn.sum('m', 'neigh'))
                # divide in_degrees
                degs = graph.in_degrees().to(feat_dst)
                h_neigh = (graph.dstdata['neigh'] + graph.dstdata['h']) / (degs.unsqueeze(-1) + 1)
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)
                    
            elif self._aggre_type == 'pool':
                graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
                graph.update_all(msg_fn, fn.max('m', 'neigh'))
                h_neigh = self.fc_neigh(graph.dstdata['neigh'])

            elif self._aggre_type == 'lstm':
                graph.srcdata['h'] = feat_src
                graph.update_all(msg_fn, self._lstm_reducer)
                h_neigh = self.fc_neigh(graph.dstdata['neigh'])
            else:
                raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))

            # GraphSAGE GCN does not require fc_self.
            if self._aggre_type == 'gcn':
                rst = h_neigh
            else:
                rst = self.fc_self(h_self) + h_neigh

            # bias term
            if self.bias is not None:
                rst = rst + self.bias

            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)

            if self.residual:
                rst = node_feats + rst

            return rst, m

from pydantic.typing import Literal
from pydantic import BaseModel
    
class ALIGNNConfig(BaseModel):
    name: Literal["alignn"]
    alignn_layers: int = 7
    atom_input_features: int = 92
    edge_input_features: int = 80
    embedding_features: int = 64
    hidden_features: int = 256
    # fc_layers: int = 1
    # fc_features: int = 64
    final_map_size: int=10000

    # if link == log, apply `exp` to final outputs
    # to constrain predictions to be positive
    link: Literal["identity", "log", "logit"] = "identity"
    zero_inflated: bool = False
    classification: bool = False

class Model_on_ALIGNN(nn.Module):
    """Atomistic Line graph network.

    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """    

    def __init__(self, config: ALIGNNConfig = ALIGNNConfig(name="alignn")):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        # print(config)
        self.atom_embedding = MLPLayer(
            config.atom_input_features, config.hidden_features
        )

        self.edge_embedding = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=config.edge_input_features,
            ),
            MLPLayer(config.edge_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )
        
        self.alignn_layers = nn.ModuleList(
            [
                EdgeGatedGraphConv(
                    config.hidden_features,
                    config.hidden_features,
                )
                for idx in range(config.alignn_layers)
            ]
        )
        self.fc1 = nn.Sequential(
            nn.Linear(config.hidden_features, config.hidden_features*4),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(config.hidden_features*4, config.hidden_features*8),
            nn.ReLU())
        self.fc3 = nn.Sequential(
            nn.Linear(config.hidden_features*8, config.final_map_size),
            nn.ReLU())        
        
    def forward(
        self, g
    ):
        """ALIGNN : start with `atom_features`.

        x: atom features (g.ndata)
        y: bond features (g.edata)
        """
        g = g.local_var()
        x = g.ndata.pop("atom_features")
        x = self.atom_embedding(x)

        # initial bond features
        bondlength = torch.norm(g.edata.pop("r"), dim=1)
        y = self.edge_embedding(bondlength)

        for alignn_layer in self.alignn_layers:
            x, y = alignn_layer(g, x, y)
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        out = F.relu(x.reshape(-1,100,100))
        return out


