import torch
from torch import nn as nn
from functools import partial
from timm.models.layers import DropPath, trunc_normal_
from models.pointr.Transformer_utils import LayerScale, Mlp, Attention, DeformableLocalAttention, DynamicGraphAttention, DeformableLocalCrossAttention, improvedDeformableLocalGraphAttention, CrossAttention, index_points
from models.pointr import misc
from models.pointr.encoder import DGCNN_Grouper, knn_point
from loss_functions.chamfer_ndim import CDNLoss

class SelfAttnBlockApi(nn.Module):
    r'''
        1. Norm Encoder Block 
            block_style = 'attn'
        2. Concatenation Fused Encoder Block
            block_style = 'attn-deform'  
            combine_style = 'concat'
        3. Three-layer Fused Encoder Block
            block_style = 'attn-deform'  
            combine_style = 'onebyone'        
    '''
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, block_style='attn-deform', combine_style='concat',
            k=10, n_group=2
        ):

        super().__init__()
        self.combine_style = combine_style
        assert combine_style in ['concat', 'onebyone'], f'got unexpect combine_style {combine_style} for local and global attn'
        self.norm1 = norm_layer(dim)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()        

        # Api desigin
        block_tokens = block_style.split('-')
        assert len(block_tokens) > 0 and len(block_tokens) <= 2, f'invalid block_style {block_style}'
        self.block_length = len(block_tokens)
        self.attn = None
        self.local_attn = None
        for block_token in block_tokens:
            assert block_token in ['attn', 'rw_deform', 'deform', 'graph', 'deform_graph'], f'got unexpect block_token {block_token} for Block component'
            if block_token == 'attn':
                self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            elif block_token == 'rw_deform':
                self.local_attn = DeformableLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
            elif block_token == 'deform':
                self.local_attn = DeformableLocalCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
            elif block_token == 'graph':
                self.local_attn = DynamicGraphAttention(dim, k=k)
            elif block_token == 'deform_graph':
                self.local_attn = improvedDeformableLocalGraphAttention(dim, k=k)
        if self.attn is not None and self.local_attn is not None:
            if combine_style == 'concat':
                self.merge_map = nn.Linear(dim*2, dim)
            else:
                self.norm3 = norm_layer(dim)
                self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, pos, idx=None):
        feature_list = []
        if self.block_length == 2:
            if self.combine_style == 'concat':
                norm_x = self.norm1(x)
                if self.attn is not None:
                    global_attn_feat = self.attn(norm_x)
                    feature_list.append(global_attn_feat)
                if self.local_attn is not None:
                    local_attn_feat = self.local_attn(norm_x, pos, idx=idx)
                    feature_list.append(local_attn_feat)
                # combine
                if len(feature_list) == 2:
                    f = torch.cat(feature_list, dim=-1)
                    f = self.merge_map(f)
                    x = x + self.drop_path1(self.ls1(f))
                else:
                    raise RuntimeError()
            else: # onebyone
                x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
                x = x + self.drop_path3(self.ls3(self.local_attn(self.norm3(x), pos, idx=idx)))

        elif self.block_length == 1:
            norm_x = self.norm1(x)
            if self.attn is not None:
                global_attn_feat = self.attn(norm_x)
                feature_list.append(global_attn_feat)
            if self.local_attn is not None:
                local_attn_feat = self.local_attn(norm_x, pos, idx=idx)
                feature_list.append(local_attn_feat)
            # combine
            if len(feature_list) == 1:
                f = feature_list[0]
                x = x + self.drop_path1(self.ls1(f))
            else:
                raise RuntimeError()

        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
    
class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """
    def __init__(self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4., qkv_bias=False, init_values=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
        block_style_list=['attn-deform'], combine_style='concat', k=10, n_group=2):
        super().__init__()
        self.k = k
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(SelfAttnBlockApi(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                act_layer=act_layer, norm_layer=norm_layer,
                block_style=block_style_list[i], combine_style=combine_style, k=k, n_group=n_group
            ))

    def forward(self, x, pos):
        idx = idx = knn_point(self.k, pos, pos)
        for _, block in enumerate(self.blocks):
            x = block(x, pos, idx=idx) 
        return x

class CrossAttnBlockApi(nn.Module):
    r'''
        1. Norm Decoder Block 
            self_attn_block_style = 'attn'
            cross_attn_block_style = 'attn'
        2. Concatenation Fused Decoder Block
            self_attn_block_style = 'attn-deform'  
            self_attn_combine_style = 'concat'
            cross_attn_block_style = 'attn-deform'  
            cross_attn_combine_style = 'concat'
        3. Three-layer Fused Decoder Block
            self_attn_block_style = 'attn-deform'  
            self_attn_combine_style = 'onebyone'
            cross_attn_block_style = 'attn-deform'  
            cross_attn_combine_style = 'onebyone'    
        4. Design by yourself
            #  only deform the cross attn
            self_attn_block_style = 'attn'  
            cross_attn_block_style = 'attn-deform'  
            cross_attn_combine_style = 'concat'    
            #  perform graph conv on self attn
            self_attn_block_style = 'attn-graph'  
            self_attn_combine_style = 'concat'    
            cross_attn_block_style = 'attn-deform'  
            cross_attn_combine_style = 'concat'    
    '''
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
            self_attn_block_style='attn-deform', self_attn_combine_style='concat',
            cross_attn_block_style='attn-deform', cross_attn_combine_style='concat',
            k=10, n_group=2
        ):
        super().__init__()        
        self.norm2 = norm_layer(dim)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()      

        # Api desigin
        # first we deal with self-attn
        self.norm1 = norm_layer(dim)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.self_attn_combine_style = self_attn_combine_style
        assert self_attn_combine_style in ['concat', 'onebyone'], f'got unexpect self_attn_combine_style {self_attn_combine_style} for local and global attn'
  
        self_attn_block_tokens = self_attn_block_style.split('-')
        assert len(self_attn_block_tokens) > 0 and len(self_attn_block_tokens) <= 2, f'invalid self_attn_block_style {self_attn_block_style}'
        self.self_attn_block_length = len(self_attn_block_tokens)
        self.self_attn = None
        self.local_self_attn = None
        for self_attn_block_token in self_attn_block_tokens:
            assert self_attn_block_token in ['attn', 'rw_deform', 'deform', 'graph', 'deform_graph'], f'got unexpect self_attn_block_token {self_attn_block_token} for Block component'
            if self_attn_block_token == 'attn':
                self.self_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            elif self_attn_block_token == 'rw_deform':
                self.local_self_attn = DeformableLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
            elif self_attn_block_token == 'deform':
                self.local_self_attn = DeformableLocalCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
            elif self_attn_block_token == 'graph':
                self.local_self_attn = DynamicGraphAttention(dim, k=k)
            elif self_attn_block_token == 'deform_graph':
                self.local_self_attn = improvedDeformableLocalGraphAttention(dim, k=k)
        if self.self_attn is not None and self.local_self_attn is not None:
            if self_attn_combine_style == 'concat':
                self.self_attn_merge_map = nn.Linear(dim*2, dim)
            else:
                self.norm3 = norm_layer(dim)
                self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Then we deal with cross-attn
        self.norm_q = norm_layer(dim)
        self.norm_v = norm_layer(dim)
        self.ls4 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path4 = DropPath(drop_path) if drop_path > 0. else nn.Identity()  

        self.cross_attn_combine_style = cross_attn_combine_style
        assert cross_attn_combine_style in ['concat', 'onebyone'], f'got unexpect cross_attn_combine_style {cross_attn_combine_style} for local and global attn'
        
        # Api desigin
        cross_attn_block_tokens = cross_attn_block_style.split('-')
        assert len(cross_attn_block_tokens) > 0 and len(cross_attn_block_tokens) <= 2, f'invalid cross_attn_block_style {cross_attn_block_style}'
        self.cross_attn_block_length = len(cross_attn_block_tokens)
        self.cross_attn = None
        self.local_cross_attn = None
        for cross_attn_block_token in cross_attn_block_tokens:
            assert cross_attn_block_token in ['attn', 'deform', 'graph', 'deform_graph'], f'got unexpect cross_attn_block_token {cross_attn_block_token} for Block component'
            if cross_attn_block_token == 'attn':
                self.cross_attn = CrossAttention(dim, dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            elif cross_attn_block_token == 'deform':
                self.local_cross_attn = DeformableLocalCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
            elif cross_attn_block_token == 'graph':
                self.local_cross_attn = DynamicGraphAttention(dim, k=k)
            elif cross_attn_block_token == 'deform_graph':
                self.local_cross_attn = improvedDeformableLocalGraphAttention(dim, k=k)
        if self.cross_attn is not None and self.local_cross_attn is not None:
            if cross_attn_combine_style == 'concat':
                self.cross_attn_merge_map = nn.Linear(dim*2, dim)
            else:
                self.norm_q_2 = norm_layer(dim)
                self.norm_v_2 = norm_layer(dim)
                self.ls5 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path5 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, q, v, q_pos, v_pos, self_attn_idx=None, cross_attn_idx=None, denoise_length=None):
        # q = q + self.drop_path(self.self_attn(self.norm1(q)))

        # calculate mask, shape N,N
        # 1 for mask, 0 for not mask
        # mask shape N, N
        # q: [ true_query; denoise_token ]
        if denoise_length is None:
            mask = None
        else:
            query_len = q.size(1)
            mask = torch.zeros(query_len, query_len).to(q.device)
            mask[:-denoise_length, -denoise_length:] = 1.

        # Self attn
        feature_list = []
        if self.self_attn_block_length == 2:
            if self.self_attn_combine_style == 'concat':
                norm_q = self.norm1(q)
                if self.self_attn is not None:
                    global_attn_feat = self.self_attn(norm_q, mask=mask)
                    feature_list.append(global_attn_feat)
                if self.local_self_attn is not None:
                    local_attn_feat = self.local_self_attn(norm_q, q_pos, idx=self_attn_idx, denoise_length=denoise_length)
                    feature_list.append(local_attn_feat)
                # combine
                if len(feature_list) == 2:
                    f = torch.cat(feature_list, dim=-1)
                    f = self.self_attn_merge_map(f)
                    q = q + self.drop_path1(self.ls1(f))
                else:
                    raise RuntimeError()
            else: # onebyone
                q = q + self.drop_path1(self.ls1(self.self_attn(self.norm1(q), mask=mask)))
                q = q + self.drop_path3(self.ls3(self.local_self_attn(self.norm3(q), q_pos, idx=self_attn_idx, denoise_length=denoise_length)))

        elif self.self_attn_block_length == 1:
            norm_q = self.norm1(q)
            if self.self_attn is not None:
                global_attn_feat = self.self_attn(norm_q, mask=mask)
                feature_list.append(global_attn_feat)
            if self.local_self_attn is not None:
                local_attn_feat = self.local_self_attn(norm_q, q_pos, idx=self_attn_idx, denoise_length=denoise_length)
                feature_list.append(local_attn_feat)
            # combine
            if len(feature_list) == 1:
                f = feature_list[0]
                q = q + self.drop_path1(self.ls1(f))
            else:
                raise RuntimeError()

        # q = q + self.drop_path(self.attn(self.norm_q(q), self.norm_v(v)))
        # Cross attn
        feature_list = []
        if self.cross_attn_block_length == 2:
            if self.cross_attn_combine_style == 'concat':
                norm_q = self.norm_q(q)
                norm_v = self.norm_v(v)
                if self.cross_attn is not None:
                    global_attn_feat = self.cross_attn(norm_q, norm_v)
                    feature_list.append(global_attn_feat)
                if self.local_cross_attn is not None:
                    local_attn_feat = self.local_cross_attn(q=norm_q, v=norm_v, q_pos=q_pos, v_pos=v_pos, idx=cross_attn_idx)
                    feature_list.append(local_attn_feat)
                # combine
                if len(feature_list) == 2:
                    f = torch.cat(feature_list, dim=-1)
                    f = self.cross_attn_merge_map(f)
                    q = q + self.drop_path4(self.ls4(f))
                else:
                    raise RuntimeError()
            else: # onebyone
                q = q + self.drop_path4(self.ls4(self.cross_attn(self.norm_q(q), self.norm_v(v))))
                q = q + self.drop_path5(self.ls5(self.local_cross_attn(q=self.norm_q_2(q), v=self.norm_v_2(v), q_pos=q_pos, v_pos=v_pos, idx=cross_attn_idx)))

        elif self.cross_attn_block_length == 1:
            norm_q = self.norm_q(q)
            norm_v = self.norm_v(v)
            if self.cross_attn is not None:
                global_attn_feat = self.cross_attn(norm_q, norm_v)
                feature_list.append(global_attn_feat)
            if self.local_cross_attn is not None:
                local_attn_feat = self.local_cross_attn(q=norm_q, v=norm_v, q_pos=q_pos, v_pos=v_pos, idx=cross_attn_idx)
                feature_list.append(local_attn_feat)
            # combine
            if len(feature_list) == 1:
                f = feature_list[0]
                q = q + self.drop_path4(self.ls4(f))
            else:
                raise RuntimeError()

        q = q + self.drop_path2(self.ls2(self.mlp(self.norm2(q))))
        return q

class TransformerDecoder(nn.Module):
    """ Transformer Decoder without hierarchical structure
    """
    def __init__(self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4., qkv_bias=False, init_values=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
        self_attn_block_style_list=['attn-deform'], self_attn_combine_style='concat',
        cross_attn_block_style_list=['attn-deform'], cross_attn_combine_style='concat',
        k=10, n_group=2):
        super().__init__()
        self.k = k
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(CrossAttnBlockApi(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                act_layer=act_layer, norm_layer=norm_layer,
                self_attn_block_style=self_attn_block_style_list[i], self_attn_combine_style=self_attn_combine_style,
                cross_attn_block_style=cross_attn_block_style_list[i], cross_attn_combine_style=cross_attn_combine_style,
                k=k, n_group=n_group
            ))

    def forward(self, q, v, q_pos, v_pos, denoise_length=None):
        if denoise_length is None:
            self_attn_idx = knn_point(self.k, q_pos, q_pos)
        else:
            self_attn_idx = None
        cross_attn_idx = knn_point(self.k, v_pos, q_pos)
        for _, block in enumerate(self.blocks):
            q = block(q, v, q_pos, v_pos, self_attn_idx=self_attn_idx, cross_attn_idx=cross_attn_idx, denoise_length=denoise_length)
        return q

class PointTransformerDecoder(nn.Module):
    """ Vision Transformer for point cloud encoder/decoder
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """
    def __init__(
            self, embed_dim=256, depth=12, num_heads=4, mlp_ratio=4., qkv_bias=True, init_values=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=None, act_layer=None,
            self_attn_block_style_list=['attn-deform'], self_attn_combine_style='concat',
            cross_attn_block_style_list=['attn-deform'], cross_attn_combine_style='concat',
            k=10, n_group=2
        ):
        """
        Args:
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            init_values: (float): layer-scale init values
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        assert len(self_attn_block_style_list) == len(cross_attn_block_style_list) == depth
        self.blocks = TransformerDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth = depth,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            init_values=init_values,
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate = dpr,
            norm_layer=norm_layer, 
            act_layer=act_layer,
            self_attn_block_style_list=self_attn_block_style_list, 
            self_attn_combine_style=self_attn_combine_style,
            cross_attn_block_style_list=cross_attn_block_style_list, 
            cross_attn_combine_style=cross_attn_combine_style,
            k=k, 
            n_group=n_group
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, q, v, q_pos, v_pos, denoise_length=None):
        q = self.blocks(q, v, q_pos, v_pos, denoise_length=denoise_length)
        return q


class PointTransformerEncoder(nn.Module):
    """ Vision Transformer for point cloud encoder/decoder
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Args:
        embed_dim (int): embedding dimension
        depth (int): depth of transformer
        num_heads (int): number of attention heads
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim
        qkv_bias (bool): enable bias for qkv if True
        init_values: (float): layer-scale init values
        drop_rate (float): dropout rate
        attn_drop_rate (float): attention dropout rate
        drop_path_rate (float): stochastic depth rate
        norm_layer: (nn.Module): normalization layer
        act_layer: (nn.Module): MLP activation layer
    """
    def __init__(
            self, embed_dim=256, depth=12, num_heads=4, mlp_ratio=4., qkv_bias=True, init_values=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=None, act_layer=None,
            block_style_list=['attn-deform'], combine_style='concat',
            k=10, n_group=2, input_dim=3
        ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        assert len(block_style_list) == depth
        self.blocks = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth = depth,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            init_values=init_values,
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate = dpr,
            norm_layer=norm_layer, 
            act_layer=act_layer,
            block_style_list=block_style_list,
            combine_style=combine_style,
            k=k,
            n_group=n_group)
        self.norm = norm_layer(embed_dim) 
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos):
        x = self.blocks(x, pos)
        return x
    
class PointTransformerEncoderEntry(PointTransformerEncoder):
    def __init__(self, config, **kwargs):
        super().__init__(**dict(config))    
    

class PointTransformerDecoderEntry(PointTransformerDecoder):
    def __init__(self, config, **kwargs):
        super().__init__(**dict(config))


class Fold(nn.Module):
    def __init__(self, in_channel, step , hidden_dim=512):
        super().__init__()

        self.in_channel = in_channel
        self.step = step

        a = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(1, step).expand(step, step).reshape(1, -1)
        b = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(step, 1).expand(step, step).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).cuda()

        self.folding1 = nn.Sequential(
            nn.Conv1d(in_channel + 2, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

    def forward(self, x):
        num_sample = self.step * self.step
        bs = x.size(0)
        features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, num_sample)
        seed = self.folding_seed.view(1, 2, num_sample).expand(bs, 2, num_sample).to(x.device)

        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)

        return fd2


class PCTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        encoder_config = config.encoder_config
        decoder_config = config.decoder_config
        self.num_basis = config.num_basis
        self.center_num  = getattr(config, 'center_num', [512, 128])
        self.encoder_type = config.encoder_type
        self.noise_on = config.noise_type
        self.denoise_point_num = config.denoise_length
        assert self.encoder_type in ['graph', 'pn'], f'unexpected encoder_type {self.encoder_type}'

        in_chans = encoder_config.input_dim
        self.num_query = query_num = config.num_query
        global_feature_dim = config.global_feature_dim

        # base encoder
        self.grouper = DGCNN_Grouper(in_chans, k = 16)

        self.pos_embed = nn.Sequential(
            nn.Linear(in_chans, 128),
            nn.GELU(),
            nn.Linear(128, encoder_config.embed_dim)
        )  
        self.input_proj = nn.Sequential(
            nn.Linear(self.grouper.num_features, 512),
            nn.GELU(),
            nn.Linear(512, encoder_config.embed_dim)
        )
        # Coarse Level 1 : Encoder
        self.encoder = PointTransformerEncoderEntry(encoder_config)

        self.increase_dim = nn.Sequential(
            nn.Linear(encoder_config.embed_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, global_feature_dim))
        # query generator
        self.coarse_pred = nn.Sequential(
            nn.Linear(global_feature_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, self.num_basis * query_num)
        )
        self.mlp_query = nn.Sequential(
            nn.Linear(global_feature_dim + self.num_basis, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, decoder_config.embed_dim)
        )
        # assert decoder_config.embed_dim == encoder_config.embed_dim
        if decoder_config.embed_dim == encoder_config.embed_dim:
            self.mem_link = nn.Identity()
        else:
            self.mem_link = nn.Linear(encoder_config.embed_dim, decoder_config.embed_dim)
        # Coarse Level 2 : Decoder
        self.decoder = PointTransformerDecoderEntry(decoder_config)
 
        self.query_ranking = nn.Sequential(
            nn.Linear(self.num_basis, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, xyz, initial_feat, basis_points):
        bs = xyz.size(0)
        coor, f = self.grouper(xyz, initial_feat, self.center_num) # b n c

        eps = 1e-6
        # Use distance to basis points as feature
        dist_to_basis_sampled = torch.sqrt(torch.sum(eps +  (coor[:,:,None,:] - basis_points[:,None,:,:])**2, axis=-1))
        dist_to_basis_sampled = torch.trunc(dist_to_basis_sampled * 1e3) / 1e3

        pe =  self.pos_embed(dist_to_basis_sampled)
        x = self.input_proj(f)

        x = self.encoder(x + pe, coor) # b n c
        global_feature = self.increase_dim(x) # B 1024 N 
        global_feature = torch.max(global_feature, dim=1)[0] # B 1024
        
        coarse = self.coarse_pred(global_feature).reshape(bs, -1, self.num_basis)
        coarse_inp = misc.fps(initial_feat, self.num_query//2)
        coarse = torch.cat([coarse, coarse_inp], dim=1) # B 224+128 num_basis?
        
        #coarse_inp = misc.fps(xyz, self.num_query//2) # B 128 3
        #coarse = torch.cat([coarse, coarse_inp], dim=1) # B 224+128 3?

        mem = self.mem_link(x)
        
        # query selection
        query_ranking = self.query_ranking(coarse) # b n 1
        idx = torch.argsort(query_ranking, dim=1, descending=True) # b n 1
        coarse = torch.gather(coarse, 1, idx[:,:self.num_query].expand(-1, -1, coarse.size(-1)))

        if self.training:
            picked_points = misc.fps(xyz, self.denoise_point_num)
            noisy_basis = basis_points.clone()

            if 'basis' in self.noise_on:
                noisy_basis = misc.jitter_points(noisy_basis)
            if 'point' in self.noise_on:
                picked_points = misc.jitter_points(picked_points)

            dist_jitter_to_basis = torch.sqrt(torch.sum(eps +  (picked_points[:,:,None,:] - noisy_basis[:,None,:,:])**2, axis=-1))
            dist_jitter_to_basis = torch.trunc(dist_jitter_to_basis * 1e3) / 1e3
            
            coarse = torch.cat([coarse, dist_jitter_to_basis], dim=1) # B 256+64 3?
            denoise_length = self.denoise_point_num

            # produce query
            q = self.mlp_query(
            torch.cat([
                global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1),
                coarse], dim = -1)) # b n c

            # forward decoder
            q = self.decoder(q=q, v=mem, q_pos=coarse, v_pos=dist_to_basis_sampled, denoise_length=denoise_length)

            return q, coarse, denoise_length

        else:
            # produce query
            q = self.mlp_query(
            torch.cat([
                global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1),
                coarse], dim = -1)) # b n c
            
            # forward decoder
            q = self.decoder(q=q, v=mem, q_pos=coarse, v_pos=dist_to_basis_sampled)

            return q, coarse, 0

class SimpleRebuildFCLayer(nn.Module):
    def __init__(self, input_dims, step, out_dim=3, hidden_dim=512):
        super().__init__()
        self.input_dims = input_dims
        self.step = step
        self.out_dims = out_dim
        self.layer = Mlp(self.input_dims, hidden_dim, step * out_dim)

    def forward(self, rec_feature):
        '''
        Input BNC
        '''
        batch_size = rec_feature.size(0)
        g_feature = rec_feature.max(1)[0]
        token_feature = rec_feature
            
        patch_feature = torch.cat([
                g_feature.unsqueeze(1).expand(-1, token_feature.size(1), -1),
                token_feature
            ], dim = -1)
        rebuild_pc = self.layer(patch_feature).reshape(batch_size, -1, self.step , self.out_dims)
        assert rebuild_pc.size(1) == rec_feature.size(1)
        return rebuild_pc


class ESCAPE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.trans_dim = config.decoder_config.embed_dim
        self.num_query = config.num_query
        self.num_points = getattr(config, 'num_points', None)

        self.decoder_type = config.decoder_type

        self.fold_step = 8
        self.base_model = PCTransformer(config)
        self.num_basis = config.num_basis
        
        self.factor = self.num_points // self.num_query
        assert self.num_points % self.num_query == 0
        self.decode_head = SimpleRebuildFCLayer(self.trans_dim * 2, step=self.num_points // self.num_query, out_dim=config.num_basis)  # rebuild a cluster point

        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )
        self.reduce_map = nn.Linear(self.trans_dim + 1024 + self.num_basis, self.trans_dim)
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_func = CDNLoss()

    def get_loss(self, ret, gt, denoise_weight=0.5):
        pred_coarse, denoised_coarse, denoised_fine, pred_fine = ret
        
        assert pred_fine.size(1) == gt.size(1)

        # denoise loss
        idx = knn_point(self.factor, gt, denoised_coarse) # B n k 
        denoised_target = index_points(gt, idx) # B n k 3 
        denoised_target = denoised_target.reshape(gt.size(0), -1, self.num_basis)
        assert denoised_target.size(1) == denoised_fine.size(1)
        loss_denoised = self.loss_func.chamfer_loss(denoised_fine, denoised_target)
        loss_denoised = loss_denoised * denoise_weight

        # recon loss
        loss_coarse = self.loss_func.chamfer_loss(pred_coarse, gt)
        loss_fine = self.loss_func.chamfer_loss(pred_fine, gt)
        loss_recon = loss_coarse + loss_fine

        return loss_denoised, loss_recon

    def forward(self, xyz, basis_points):

        eps = 1e-6
        # Use distance to basis points as feature
        dist_to_basis = torch.sqrt(torch.sum(eps +  (xyz[:,:,None,:] - basis_points[:,None,:,:])**2, axis=-1))
        dist_to_basis = torch.trunc(dist_to_basis * 1e3) / 1e3

        #q, coarse_point_cloud, denoise_length = self.base_model(xyz, dist_to_basis, basis_points) # B M C and B M 3
        q, coarse_point_cloud, denoise_length = self.base_model(xyz, dist_to_basis, basis_points) # B M C and B M 3
    
        B,M,C = q.shape

        global_feature = self.increase_dim(q.transpose(1,2)).transpose(1,2) # B M 1024
        global_feature = torch.max(global_feature, dim=1)[0] # B 1024

        rebuild_feature = torch.cat([
            global_feature.unsqueeze(-2).expand(-1, M, -1),
            q,
            coarse_point_cloud], dim=-1)  # B M 1027 + C
        
        rebuild_feature = self.reduce_map(rebuild_feature) # B M C
        relative_xyz = self.decode_head(rebuild_feature)   # B M S 3
        rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-2))  # B M S 3

        if self.training:
            # split the reconstruction and denoise task
            pred_fine = rebuild_points[:, :-denoise_length].reshape(B, -1, self.num_basis).contiguous()
            pred_coarse = coarse_point_cloud[:, :-denoise_length].contiguous()

            denoised_fine = rebuild_points[:, -denoise_length:].reshape(B, -1, self.num_basis).contiguous()
            denoised_coarse = coarse_point_cloud[:, -denoise_length:].contiguous()

            assert pred_fine.size(1) == self.num_query * self.factor
            assert pred_coarse.size(1) == self.num_query

            ret = (pred_coarse, denoised_coarse, denoised_fine, pred_fine)
            return ret

        else:
            assert denoise_length == 0
            rebuild_points = rebuild_points.reshape(B, -1, self.num_basis).contiguous()  # B N 3

            assert rebuild_points.size(1) == self.num_query * self.factor
            assert coarse_point_cloud.size(1) == self.num_query

            ret = (coarse_point_cloud, rebuild_points)
            return ret
        