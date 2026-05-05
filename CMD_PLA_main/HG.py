from torch import nn
import torch
import torch_geometric
from torch_geometric.nn import global_add_pool
from TDAW import TDAW, TDAW_E

# ========== 工具：unsorted_segment_* ==========
def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result

def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)

# ========== E(n)-GCL ==========
class E_GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(),
                 residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf)
        )

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = [nn.Linear(hidden_nf, hidden_nf), act_fn, layer]
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)
        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm
        return radial, coord_diff

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise ValueError(f"Wrong coords_agg parameter: {self.coords_agg}")
        coord = coord + agg
        return coord

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        radial, coord_diff = self.coord2radial(edge_index, coord)
        edge_feat = self.edge_model(h[edge_index[0]], h[edge_index[1]], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, _ = self.node_model(h, edge_index, edge_feat, node_attr)
        return h, coord, edge_attr


# ========== 蛋白 3D Graph 模型 ==========
class Prot3DGraphModel(nn.Module):
    def __init__(self, d_vocab=21, d_embed=20, d_dihedrals=6, d_pretrained_emb=1280, d_edge=39, d_gcn=[128, 256, 256]):
        super(Prot3DGraphModel, self).__init__()
        d_gcn_in = d_gcn[0]
        self.embed = nn.Embedding(d_vocab, d_embed)
        self.proj_node = nn.Linear(d_embed + d_dihedrals + d_pretrained_emb, d_gcn_in)
        self.proj_edge = nn.Linear(d_edge, d_gcn_in)

        gcn_layer_sizes = [d_gcn_in] + d_gcn
        layers = []
        for i in range(len(gcn_layer_sizes) - 1):
            layers.append((
                torch_geometric.nn.TransformerConv(
                    gcn_layer_sizes[i], gcn_layer_sizes[i + 1], edge_dim=d_gcn_in
                ),
                'x, edge_index, edge_attr -> x'
            ))
            layers.append(nn.LeakyReLU())
        self.gcn = torch_geometric.nn.Sequential('x, edge_index, edge_attr', layers)

    def forward(self, data):
        x, edge_index, batch = data.seq, data.edge_index, data.batch
        x = self.embed(x)
        s = data.node_s
        emb = data.seq_emb
        x = torch.cat([x, s, emb], dim=-1)
        edge_attr = data.edge_s
        x = self.proj_node(x)
        edge_attr = self.proj_edge(edge_attr)
        x = self.gcn(x, edge_index, edge_attr)
        x = torch_geometric.nn.global_mean_pool(x, batch)
        return x


# ========== 全连接层堆叠 ==========
class FC(nn.Module):
    def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(FC, self).__init__()
        self.d_graph_layer = d_graph_layer
        self.d_FC_layer = d_FC_layer
        self.n_FC_layer = n_FC_layer
        self.dropout = dropout
        self.predict = nn.ModuleList()
        for j in range(self.n_FC_layer):
            if j == 0:
                self.predict.append(nn.Linear(self.d_graph_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(self.d_FC_layer))
            if j == self.n_FC_layer - 1:
                self.predict.append(nn.Linear(self.d_FC_layer, n_tasks))
            else:
                self.predict.append(nn.Linear(self.d_FC_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(self.d_FC_layer))

    def forward(self, h):
        for layer in self.predict:
            h = layer(h)
        return h


# ========== HG 主干  ==========
class HG(nn.Module):
    def __init__(
        self,
        in_node_nf,
        hidden_nf,
        out_node_nf,
        in_edge_nf=0,
        act_fn=nn.SiLU(),
        n_layers=3,
        residual=True,
        attention=False,
        normalize=True,
        tanh=False,
        lig_emb_dim=300,
    ):
        super(HG, self).__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.lig_emb_dim = lig_emb_dim

        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.lin_node = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf), nn.SiLU())

        self.gconv1 = TDAW(self.hidden_nf, self.hidden_nf)
        self.gconv2 = TDAW(self.hidden_nf, self.hidden_nf)
        self.gconv3 = TDAW(self.hidden_nf, self.hidden_nf)

        self.gnconv1 = TDAW_E(self.hidden_nf, self.hidden_nf)
        self.gnconv2 = TDAW_E(self.hidden_nf, self.hidden_nf)
        self.gnconv3 = TDAW_E(self.hidden_nf, self.hidden_nf)

        self.fc_atom = FC(self.hidden_nf * 2, self.hidden_nf, 3, 0.1, self.hidden_nf)
        self.fc_add  = FC(self.hidden_nf, self.hidden_nf, 3, 0.1, self.hidden_nf)
        self.fc_p    = FC(self.hidden_nf * 2, self.hidden_nf, 3, 0.1, self.hidden_nf)

        self.fc = FC(self.hidden_nf * 3, self.hidden_nf, 3, 0.1, out_node_nf)

        self.mlp_node_cov  = nn.Sequential(
            nn.Linear(self.hidden_nf, self.hidden_nf),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.hidden_nf)
        )
        self.mlp_node_ncov = nn.Sequential(
            nn.Linear(self.hidden_nf, self.hidden_nf),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.hidden_nf)
        )

        # E_GCL 堆叠
        for i in range(0, n_layers):
            self.add_module(
                f"gcl_{i}",
                E_GCL(
                    self.hidden_nf, self.hidden_nf, self.hidden_nf,
                    edges_in_d=in_edge_nf,
                    act_fn=act_fn, residual=residual, attention=attention,
                    normalize=normalize, tanh=tanh
                )
            )

        # Pocket+ESM 的 3D Graph 模型（保持不变）
        self.prot_model = Prot3DGraphModel(d_pretrained_emb=1280, d_gcn=[128, 256, 256])

        # 把 ligand 的 Mol2Vec (lig_emb_dim) 投到 hidden_nf
        self.lig_proj = nn.Linear(lig_emb_dim, self.hidden_nf)

    def forward(self, drug, pock, comp, esm_fea, edge_attr=None):
        # drug
        h_l, edge_intra_l, edge_attr_l, x_l, batch_l = drug.x, drug.edge_index, drug.edge_attr, drug.pos, drug.batch
        # pock
        h_p, edge_intra_p, edge_attr_p, x_p, batch_p = pock.x, pock.edge_index, pock.edge_attr, pock.pos, pock.batch
        # comp
        h_c, edge_intra_c, edge_inter_c, x_c, batch_c = comp.x, comp.edge_index_intra, comp.edge_index_inter, comp.pos, comp.batch
        edges_c = torch.cat([edge_intra_c, edge_inter_c], dim=-1)
        pos = x_c

        # ====== comp 图：异质交互 + E_GCL + TDAW ======
        h_c = self.embedding_in(h_c)
        h_intra, h_inter = h_c, h_c
        x_intra, x_inter = x_c, x_c
        for i in range(0, self.n_layers):
            h_intra, x_intra, _ = self._modules[f"gcl_{i}"](h_intra, edge_intra_c, x_intra, edge_attr=edge_attr)
            h_inter, x_inter, _ = self._modules[f"gcl_{i}"](h_inter, edge_inter_c, x_inter, edge_attr=edge_attr)

        h_c = self.lin_node(self.mlp_node_cov(h_intra) + self.mlp_node_ncov(h_inter))
        h_c = self.gconv1(h_c, edge_intra_c, edge_inter_c, pos)
        h_c = self.gconv2(h_c, edge_intra_c, edge_inter_c, pos)
        h_c = self.gconv3(h_c, edge_intra_c, edge_inter_c, pos)
        h_c = global_add_pool(h_c, batch_c)  # (B, hidden_nf)

        # ====== pock graph transformer（ESM + 3D 口袋）======
        h_p_new = self.prot_model(esm_fea)  # (B, hidden_nf) 这里是 256

        # ====== ligand 全局 Mol2Vec 向量（来自 Data.lig_emb）======
        if hasattr(drug, "lig_emb"):
            lig_emb = drug.lig_emb  # 期望 (B, lig_emb_dim)

            if lig_emb.dim() == 1:
                B = h_c.size(0)
                lig_emb = lig_emb.view(B, -1)
        else:
            B = h_c.size(0)
            lig_emb = torch.zeros(B, self.lig_emb_dim, device=h_c.device)

        lig_feat = self.lig_proj(lig_emb)  # (B, hidden_nf)

        # ====== 三块特征拼接：comp + pocket + ligand_global  ======
        h_all = torch.cat([h_c, h_p_new, lig_feat], dim=-1)  # (B, 3 * hidden_nf)

        h = self.fc(h_all)
        return h.view(-1)


if __name__ == '__main__':
    print("hg")
    pass# %%