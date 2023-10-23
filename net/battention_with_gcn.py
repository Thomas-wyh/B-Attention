#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
#from dgl.nn.pytorch import GATConv
# from dgl.nn.pytorch import SAGEConv
from .optim_modules import ClusterLoss
import torch.nn.functional as F
import dgl
from dgl.data.utils import save_graphs
import math

def scaled_dot_product(q, k, v, temperature=1.0, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)

    set_version = 3

    if set_version == 1:
        set_attn = attn_logits.sum(-1, keepdim=True)
        set_attn = F.softmax(set_attn, dim=-2)
        set_attn = torch.matmul(set_attn, set_attn.transpose(-2, -1))
        attn_logits = torch.mul(attn_logits, set_attn)
    elif set_version == 2:
        set_attn = attn_logits.sum(-1, keepdim=True)
        # set_attn = F.normalize(set_attn, p=2, dim=-2)
        set_attn = set_attn / set_attn.max()
        set_attn = torch.matmul(set_attn, set_attn.transpose(-2, -1))
        attn_logits = torch.mul(attn_logits, set_attn)
    elif set_version == 3:
        set_attn = F.normalize(attn_logits, p=2, dim=-1)
        set_attn = torch.matmul(set_attn, set_attn.transpose(-2, -1))
        attn_logits = torch.mul(attn_logits, set_attn)
    elif set_version == 4:
        set_attn = F.normalize(attn_logits, p=2, dim=-1)
        set_attn = torch.matmul(set_attn, set_attn.transpose(-2, -1))
        attn_logits = set_attn
    elif set_version == 5:
        set_attn = attn_logits
        set_attn = torch.matmul(set_attn, set_attn.transpose(-2, -1))
        attn_logits = set_attn

    attn_logits = attn_logits / temperature
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

# A GCN Version of MultiheadAttention
class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads, max_ctx_k=120, temperature=1.0):
        #super(MultiheadAttention).__init__()
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.temperature = temperature
        # print("Softmax_temperature: ", self.temperature)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 2*embed_dim)
        #self.o_proj = nn.Linear(2*embed_dim, input_dim)
        self.o_proj = nn.Linear(2*input_dim, input_dim)
        #self.o_proj = nn.Linear(embed_dim, input_dim)
        # self.weight = nn.Parameter(
        #         torch.FloatTensor(embed_dim *2, input_dim))
        # self.bias = nn.Parameter(torch.FloatTensor(input_dim))
        

        # weights for context attention
        self.ctx_qk_proj = nn.Linear(max_ctx_k, 2*max_ctx_k)
        self.attn_fuse = nn.Linear(2, 1)

        self._reset_parameters()


    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        #self.qkv_proj.bias.data.fill_(0)
        #nn.init.xavier_normal_(self.qkv_proj.bias)
        nn.init.xavier_normal_(self.qkv_proj.bias.view(1,-1))

        nn.init.xavier_uniform_(self.o_proj.weight)
        #self.o_proj.bias.data.fill_(0)
        #nn.init.xavier_normal_(self.o_proj.bias)
        nn.init.xavier_normal_(self.o_proj.bias.view(1,-1))
        # nn.init.xavier_uniform_(self.weight)
        # nn.init.constant_(self.bias, 0)

        nn.init.xavier_uniform_(self.ctx_qk_proj.weight)
        #self.ctx_qk_proj.bias.data.fill_(0)
        #nn.init.xavier_normal_(self.ctx_qk_proj.bias)
        nn.init.xavier_normal_(self.ctx_qk_proj.bias.view(1,-1))

        nn.init.xavier_uniform_(self.attn_fuse.weight)
        #self.ctx_qk_proj.bias.data.fill_(0)
        #nn.init.xavier_normal_(self.ctx_qk_proj.bias)
        nn.init.xavier_normal_(self.attn_fuse.bias.view(1,-1))


    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, input_dim = x.size()
        embed_dim = self.embed_dim
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 2*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k  = qkv.chunk(2, dim=-1)
        #v = x.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        v = x.reshape(batch_size, seq_length, self.num_heads, input_dim)
        v = v.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        # v = x

        # Obtain normalized q k v vectors
        # q = F.normalize(q, p=2, dim=-1)
        # k = F.normalize(k, p=2, dim=-1)
        # v = F.normalize(v, p=2, dim=-1)

        # Obtain self-attention
        d_k = q.size()[-1]
        self_attn = torch.matmul(q, k.transpose(-2, -1))
        # self_attn = self_attn / math.sqrt(d_k) # To Confirm: 
        #self_attn = F.softmax(self_attn, dim=-1)
        
        # Determine value outputs
        # values, attention = scaled_dot_product(q, k, v, temperature=self.temperature, mask=mask)

        # obtain context-attention
        adj_mat = torch.matmul(x, x.transpose(-2, -1))
        adj_mat = adj_mat.reshape(batch_size, 1, seq_length, seq_length) # [Batch, Head, SeqLen, SeqLen]
        # adj_mat = F.normalize(adj_mat, p=2, dim=-1)
        #ctx_x = F.normalize(adj_mat, p=2, dim=-1) # To Confirm: 
        #ctx_attn = torch.matmul(ctx_x, ctx_x.transpose(-2, -1))
        ctx_qk = self.ctx_qk_proj(adj_mat) # [Batch, Head, SeqLen, 2*SeqLen]
        # ctx_qk = F.normalize(ctx_qk, p=2, dim=-1) # To Confirm: 
        ctx_q, ctx_k = ctx_qk.chunk(2, dim=-1)
        ctx_q = F.normalize(ctx_q, p=2, dim=-1) # To Confirm: 
        ctx_k = F.normalize(ctx_k, p=2, dim=-1) # To Confirm: 
        ctx_attn = torch.matmul(ctx_q, ctx_k.transpose(-2, -1))
        #ctx_attn = F.softmax(ctx_attn, dim=-1)

        # For multiple heads, where the same context attention map are used in all heads
        if self.num_heads > 1:
            ctx_attn = ctx_attn.repeat(1,self.num_heads,1,1)
        
        # obtain hybrid attention
        # attn = torch.mul(self_attn, ctx_attn)
        attn_concat = torch.cat((torch.unsqueeze(self_attn, -1), torch.unsqueeze(ctx_attn, -1)), -1)
        attn = self.attn_fuse(attn_concat)
        attn = torch.squeeze(attn, -1)
        #attn = F.relu(attn,inplace=True)

        # softmax with temperature
        attn = attn / self.temperature
        attn = F.softmax(attn, dim=-1)

        # mask the attention map
        if mask is not None:
            mask = mask.reshape(batch_size, 1, seq_length, seq_length)
            mask = mask.repeat(1,self.num_heads,1,1)
            attn = attn.masked_fill(mask == 0, 0)
            attn = F.normalize(attn, p=1, dim=-1) 

        # obtain value outputs
        values = torch.matmul(attn, v)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        #values = values.reshape(batch_size, seq_length, embed_dim)
        values = values.reshape(batch_size, seq_length, input_dim)
        values = torch.cat([x, values], dim=-1)
        o = self.o_proj(values)
        #o = F.relu(o)

        if return_attention:
            # attn = torch.cat((self_attn, ctx_attn, attn), 1) # TODO: not compatible to multi-heads
            # attn = torch.cat((F.softmax(self_attn, dim=-1), F.softmax(ctx_attn, dim=-1), attn), 1) # TODO: not compatible to multi-heads
            #attn = torch.cat((self_attn/self_attn.max(), ctx_attn/ctx_attn.max(), attn/attn.max()), 1) # TODO: not compatible to multi-heads
            #print(self_attn[0,:,0,:])
            #print(ctx_attn[0,:,0,:])
            self_attn_v = F.softmax(self_attn, dim=-1)
            ctx_attn_v = F.softmax(ctx_attn, dim=-1)
            attn = torch.cat((self_attn_v/self_attn_v.max(), ctx_attn_v/ctx_attn_v.max(), attn/attn.max()), 1) # TODO: not compatible to multi-heads
            return o, attn
        else:
            return o


class EncoderBlock(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads, max_ctx_k, dim_feedforward, dropout=0.0, temperature=1.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        # self.self_attn = MultiheadAttention(input_dim, embed_dim, num_heads)
        self.self_attn = MultiheadAttention(input_dim, embed_dim, num_heads, max_ctx_k, temperature=temperature)

        # Two-layer MLP
        # self.linear_net = nn.Sequential(
        #     nn.Linear(input_dim, dim_feedforward),
        #     nn.Dropout(dropout),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(dim_feedforward, input_dim)
        # )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        # self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        #x = attn_out
        x = self.dropout(attn_out)
        # x = x + self.dropout(attn_out)
        x = self.norm1(x)
        # x = F.normalize(x, p=2, dim=-1)

        # MLP part
        # linear_out = self.linear_net(x)
        # x = x + self.dropout(linear_out)
        # x = self.norm2(x)


        return x


class TransformerEncoder(nn.Module):

    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])


    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x


    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            # attention_maps.append(attn_map)
            attention_maps.append(attn_map.cpu())
            x = l(x)
        return attention_maps


class GCN_Transformer(nn.Module):
    def __init__(self, feature_dim, embed_dim=512, out_dim=512, nhid=2048, nhead=8, nlayer=1, dropout=0.1, temperature=1.0, losstype='allall', margin=1., pweight=4., pmargin=1.0, max_ctx_k=80):
        super(GCN_Transformer, self).__init__()

        # try:
        #     from torch.nn import TransformerEncoder, TransformerEncoderLayer
        # except:
        #     raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.max_ctx_k = max_ctx_k
        # self.pos_encoder = PositionalEncoding(ninp, dropout)
        # encoder_layer = TransformerEncoderLayer(feature_dim, nhead, nhid, dropout)
        #self.transformer_encoder1 = TransformerEncoder(encoder_layer, nlayer)
        # self.transformer_encoder2 = TransformerEncoder(encoder_layer, nlayer)

        self.transformer_encoder2 = TransformerEncoder(num_layers=nlayer,
                                              input_dim=feature_dim,
                                              embed_dim=embed_dim,
                                              num_heads=nhead,
                                              max_ctx_k=max_ctx_k,
                                              dim_feedforward=nhid,
                                              dropout=dropout,
                                              temperature=temperature)
        # self.encoder = nn.Embedding(ntoken, ninp)
        self.feat_dim = feature_dim
        # self.decoder = nn.Linear(feature_dim, nclass)

        # self.init_weights()

        #self.conv1 = GraphConv(feature_dim, nhid, MeanAggregator, dropout)
        #self.gat = GATConv(feature_dim, nhid, num_heads=4, feat_drop=0., attn_drop=0.5, \
        #        negative_slope=0.2, residual=False, activation=None, allow_zero_in_degree=False)
        #self.gat = GATConv(feature_dim, nhid, num_heads=4, feat_drop=0.6, attn_drop=0.6, \
        #        negative_slope=0.2, residual=False, activation=None, allow_zero_in_degree=False)
        #self.gat = GATConv(feature_dim, nhid, num_heads=4, feat_drop=0., attn_drop=0., \
        #        negative_slope=0.2, residual=False, activation=None, allow_zero_in_degree=False)
        #self.gat = GATConv(feature_dim, nhid, num_heads=8, feat_drop=0., attn_drop=0., \
        #        negative_slope=0.2, residual=False, activation=None, allow_zero_in_degree=False)
        #self.sage = SAGEConv(feature_dim, nhid, aggregator_type='pool', activation=F.relu)
        # self.sage1 = SAGEConv(feature_dim, nhid, aggregator_type='gcn', activation=F.relu)
        # self.sage2 = SAGEConv(nhid, nhid, aggregator_type='gcn', activation=F.relu)
         
        # self.nclass = nclass

        # self.fc = nn.Sequential(nn.Linear(feature_dim, nhid), nn.PReLU(nhid))
        self.fc = nn.Sequential(nn.Linear(feature_dim, out_dim), nn.PReLU(out_dim))
        self.loss = torch.nn.MSELoss()
        #self.bclloss = BallClusterLearningLoss()
        self.bclloss = ClusterLoss(losstype=losstype, margin=margin, alpha_pos=pweight, pmargin=pmargin)

        self.sample_attention_maps = None

    def block_to_graph(self, block):
        #g = dgl.DGLGraph()
        #induced_dst = block.dstdata[dgl.NID]
        #induced_src = block.srcdata[dgl.NID]
        src, dst = block.edges()
        #torch.save(src.cpu(),'src.pt')
        #torch.save(dst.cpu(),'dst.pt')
        #print(src)
        #print(dst)
        #print(induced_dst)
        #src_ind, dst_ind = induced_src[src], induced_dst[dst]
        #sorted_inds = torch.argsort(dst_ind)
        #sorted_dst_ind, sorted_src_ind = dst_ind[sorted_inds], src_ind[sorted_inds]
        #print(sorted_dst_ind, sorted_src_ind)
        #g.add_edges(dst_ind.cpu(), src_ind.cpu())
        #g = dgl.graph((dst_ind.cpu(), src_ind.cpu()))
        g = dgl.graph((dst.cpu(), src.cpu()))
        g = dgl.to_bidirected(g)
        #g = dgl.DGLGraph((dst_ind, src_ind))
        return g

    def extract_ctx_feats(self, x, block):
        g = self.block_to_graph(block)
        #_, seed_nodes = block.dstdata[dgl.NID]
        _, seed_nodes = block.edges()
        #print(seed_nodes)
        #seed_nodes = torch.LongTensor(range(seed_nodes.max()+1))
        seed_nodes = torch.unique(seed_nodes)
        #print(seed_nodes)
        num_seed_nodes = len(seed_nodes)
        features = list()
        x_cpu = x.cpu()
        #ctx_k = 80
        ctx_k = 120
        features = torch.zeros([num_seed_nodes, ctx_k, self.feat_dim], dtype=torch.float32)
        for i in range(num_seed_nodes):
            seed_node = seed_nodes[i].cpu()
            #print("seed: ", seed_node)
            #neighbours = dgl.sampling.sample_neighbors(dgl.block_to_graph(block), seed_node, -1, replace=False)
            #neighbours = dgl.sampling.sample_neighbors(self.block_to_graph(block), seed_node.cpu(), -1, replace=False)
            neighbours = dgl.sampling.sample_neighbors(g, seed_node, ctx_k, replace=False)
            #neighbours = dgl.transform.remove_self_loop(neighbours)
            neighbours = dgl.to_block(neighbours, seed_node)
            neighbours.create_format_()
            nb_inds = neighbours.srcdata[dgl.NID]
            nb_inds = nb_inds[:ctx_k]
            #print(nb_inds.shape)
            #print(nb_inds)
            #ctx_inds = [seed_node]
            ctx_inds = []
            ctx_inds.extend(nb_inds.tolist())
            #print(ctx_inds)
            #print(1)
            ctx_inds = torch.LongTensor(ctx_inds)
            #print(2)
            #ctx_features = x_cpu[ctx_inds]
            #print(ctx_inds.shape, ctx_inds)
            features[i,:,:] = x_cpu[ctx_inds]
            #ctx_features = x[ctx_inds].numpy()
            #print(3)
            #features.append(ctx_features)
            #print(4)
        #features = torch.FloatTensor(features).cuda()
        features = features.cuda()
        #print(5)
        return features, seed_nodes

    def extract_ctx_feats2(self, x, block):
        # g = self.block_to_graph(block)
        #_, seed_nodes = block.dstdata[dgl.NID]
        induced_src = block.srcdata[dgl.NID]
        induced_dst = block.dstdata[dgl.NID]
        #print("induced_src: ", induced_src[:128])
        #print("induced_dst: ", induced_dst[:128])
        src_nodes, seed_nodes = block.edges()

        # In case the seed_nodes are not in a order we want
        seed_nodes, inds_sorted = torch.sort(seed_nodes)
        src_nodes = src_nodes[inds_sorted]

        #print(seed_nodes)
        #seed_nodes = torch.LongTensor(range(seed_nodes.max()+1))
        seed_nodes, cnts = torch.unique(seed_nodes, return_counts=True)
        #print(seed_nodes)
        ind = 0
        num_seed_nodes = len(seed_nodes)
        x_cpu = x.cpu()

        max_ctx_k = self.max_ctx_k
        features = torch.zeros([num_seed_nodes, max_ctx_k, self.feat_dim], dtype=torch.float32)
        node_inds = torch.zeros([num_seed_nodes, max_ctx_k], dtype=torch.long)
        mask = torch.zeros([num_seed_nodes, max_ctx_k, max_ctx_k], dtype=torch.long)
        for i in range(num_seed_nodes):
            seed_node = seed_nodes[i].cpu()
            #nb_inds = src_nodes[ind:ind+80].cpu()
            nb_inds = src_nodes[ind:ind+cnts[i]].cpu()
            nb_inds_nid = induced_src[nb_inds].cpu()
            seed_node_nid = induced_dst[seed_node].cpu()
            idx = (nb_inds_nid == seed_node_nid).nonzero().flatten()
            #idx0 = (nb_inds[:80] == seed_node).nonzero().flatten()
            #if idx0 != idx:
            #    print("idx0, idx:", idx0, idx)
            tmp = nb_inds[idx]
            #print("seed_node, seed_node_nid:", seed_node, seed_node_nid)
            #print("idx, tmp:", idx, tmp)
            nb_inds[idx] = nb_inds[0]
            nb_inds[0] = tmp
            ind = ind+cnts[i]
            #print(ind, cnts[i])
            #print(nb_inds.shape)
            #print(nb_inds)
            #ctx_inds = [seed_node]
            #ctx_inds = []
            #ctx_inds.extend(nb_inds.tolist())
            #print(ctx_inds)
            #print(1)

            # ctx_inds = torch.LongTensor(nb_inds[:max_ctx_k])
            # features[i,:,:] = x_cpu[ctx_inds]
            # node_inds[i,:] = induced_src[ctx_inds].cpu()
            end_ind = min(cnts[i], max_ctx_k)
            ctx_inds = torch.LongTensor(nb_inds[:end_ind])
            features[i,:end_ind,:] = x_cpu[ctx_inds]
            node_inds[i,:end_ind] = induced_src[ctx_inds].cpu()
            mask[i,:end_ind,:end_ind] = 1
        # features = features.transpose(0,1)
        features = features.cuda()
        mask = mask.cuda()
        return features, seed_nodes, node_inds, mask

    def forward(self, data, output_feat=False, return_loss=False):
        assert not output_feat or not return_loss
        x, block_list, label, idlabel = data[0], data[1], data[2], data[3]
        #x, block_list, label_probe, label = data[0], data[1], data[2], data[3]
        #print(x.size())
        #print(block_list)
        # TransformerEncoder
        # src = self.encoder(src) * math.sqrt(self.ninp)
        # src = self.pos_encoder(src)
        #save_graphs("./block_list.bin", [self.block_to_graph(block_list[0]),self.block_to_graph(block_list[1])])
        #block = block_list[0]
        #adj_ind = dgl.khop_adj(block, 1)
        #induced_dst = block.dstdata[dgl.NID]
        #induced_src = block.srcdata[dgl.NID]
        #src, dst = block.edges()
        #src_ind, dst_ind = induced_src[src], induced_dst[dst]
        #sorted_inds = torch.argsort(dst_ind)
        #sorted_dst_ind, sorted_src_ind = dst_ind[sorted_inds], src_ind[sorted_inds]
        #print(sorted_dst_ind, sorted_src_ind)
        
        # features, seed_nodes = self.extract_ctx_feats(x, block_list[0])
        # features, seed_nodes, node_inds = self.extract_ctx_feats2(x, block_list[0])
        #x = x.view(1,x.size(0),x.size(1))
        #transfeat = self.transformer_encoder1(features, self.src_mask)
        #transfeat = self.transformer_encoder(x, self.src_mask)
        #transfeat = transfeat.view(transfeat.size(1), transfeat.size(2))
        #transfeat = transfeat[0,:,:]
        # transfeat = features[:,0,:]
        #print("transfeat size: ", transfeat.size())
        # output = self.decoder(output)
        # return F.log_softmax(output, dim=-1)

        # layer1
        # gcnfeat = self.sage1(block_list[0], x)
        #transfeat = F.normalize(transfeat, p=2, dim=1)
        #x[seed_nodes] = transfeat

        # layer2
        # gcnfeat = self.sage2(block_list[1], gcnfeat)
        #features, seed_nodes = self.extract_ctx_feats(x, block_list[1])
        # features, seed_nodes, node_inds = self.extract_ctx_feats2(transfeat, block_list[1])
        features, seed_nodes, node_inds, mask = self.extract_ctx_feats2(x, block_list[0])

        transfeat = self.transformer_encoder2(features, mask)
        transfeat = transfeat[:,0,:]
        transfeat = F.normalize(transfeat, p=2, dim=1)

        # layer3
        fcfeat = self.fc(transfeat)
        fcfeat = F.normalize(fcfeat, dim=1)

        #fcfeat = fcfeat.view(fcfeat.size(1), fcfeat.size(2))

        if output_feat:
            return fcfeat, transfeat

        if return_loss:
            #print(fcfeat.size(), label.size())
            bclloss_dict = self.bclloss(fcfeat, label)
            return bclloss_dict

        return fcfeat

    # @torch.no_grad()
    def get_attention_maps(self, data):
        """
        Function for extracting the attention matrices of the whole Transformer for a single batch.
        Input arguments same as the forward pass.
        """
        x, block_list, label, idlabel = data[0], data[1], data[2], data[3]

        # layer1
        # features, seed_nodes, node_inds = self.extract_ctx_feats2(x, block_list[0])
        #transfeat = self.transformer_encoder1(features, self.src_mask)
        #transfeat = transfeat[0,:,:]
        # transfeat = features[:,0,:]
        # gcnfeat = self.sage1(block_list[0], x)
        #transfeat = F.normalize(transfeat, p=2, dim=1)

        # layer2
        # gcnfeat = self.sage2(block_list[1], gcnfeat)
        #features, seed_nodes = self.extract_ctx_feats(x, block_list[1])
        # features, seed_nodes, node_inds = self.extract_ctx_feats2(transfeat, block_list[1])
        features, seed_nodes, node_inds, mask = self.extract_ctx_feats2(x, block_list[0])
        # transfeat = self.transformer_encoder2(features, self.src_mask)
        x = features[-1,:,:]
        x = x.view(1, x.size(0), x.size(1))
        x_inds = node_inds[-1,:]
        mask = mask[-1,:,:]
        mask = mask.view(1, mask.size(0), mask.size(1))
        attention_maps = self.transformer_encoder2.get_attention_maps(x, mask)
        
        return x.cpu(), x_inds.cpu(), attention_maps


#def gcn_v(feature_dim, nhid, nclass=1, dropout=0., **kwargs):
#    model = GCN_V(feature_dim=feature_dim,
#                  nhid=nhid,
#                  nclass=nclass,
#                  dropout=dropout)
#    return model
