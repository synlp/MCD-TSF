import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from linear_attention_transformer import LinearAttentionTransformer
from SelfAttention_Family import FullAttention, TV_AttentionLayer, TV_AttentionEncoderLayer, AttentionLayer, AttentionEncoderLayer

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

def get_linear_trans(heads=8,layers=1,channels=64,localheads=0,localwindow=0):

  return LinearAttentionTransformer(
        dim = channels,
        depth = layers,
        heads = heads,
        max_seq_len = 256,
        n_local_attn_heads = 0, 
        local_attn_window_size = 0,
    )

def get_custom_tv_trans(heads=8, layers=1, channels=64, dropout=0.0, pre_norm=False):
    encoder_layer = TV_AttentionLayer(
        FullAttention(mask_flag=False),
        d_t=channels//4,
        d_v=channels,
        n_heads=heads
        )
    return TV_AttentionEncoderLayer(encoder_layer, d_t=channels//4, d_v=channels, d_ff=64, activation="gelu", dropout=dropout, pre_norm=pre_norm)

def get_cross_trans(heads=8, layers=1, channels=64, dropout=0.0, pre_norm=False):
    encoder_layer = AttentionLayer(
        FullAttention(mask_flag=False),
        d_model=channels,
        n_heads=heads
    )
    return AttentionEncoderLayer(encoder_layer, d_model=channels, d_ff=64, activation="gelu", dropout=dropout, pre_norm=pre_norm)

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step] # (B, diffusion_embedding_dim)
        x = self.projection1(x) # (B, diffusion_embedding_dim)
        x = F.silu(x)
        x = self.projection2(x) # (B, diffusion_embedding_dim)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, is_linear=False, with_text=False, context_dim=None, dropout=0., attn_drop=0., pre_norm=False, pred_len=-1):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.dropout = nn.Dropout(dropout)
        self.attn_drop = nn.Dropout(attn_drop)

        self.is_linear = is_linear
        self.with_text = with_text
        if is_linear:
            self.time_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)
            self.feature_layer = get_linear_trans(heads=nheads,layers=1,channels=channels)
        else:
            self.time_layer = get_custom_tv_trans(heads=nheads, layers=1, channels=channels, dropout=dropout, pre_norm=pre_norm)
            self.feature_layer = get_custom_tv_trans(heads=nheads, layers=1, channels=channels)
        if with_text:
            self.channel_proj = nn.Linear(context_dim, channels)
            # self.len_proj = nn.Linear(768, pred_len) ########################
            self.cross_modal_layer = get_cross_trans(heads=nheads, layers=1, channels=channels, dropout=dropout, pre_norm=pre_norm)

    # fusing information along L
    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)

        if self.is_linear:
            y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y


    # fusing information along K
    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        if self.is_linear:
            y = self.feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y
    
    def forward_time_TV(self, yt, yv, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return yt, yv
        yt = yt.reshape(B, -1, K, L).permute(0, 2, 1, 3).reshape(B * K, -1, L)
        yv = yv.reshape(B, -1, K, L).permute(0, 2, 1, 3).reshape(B * K, -1, L)

        yt, yv, attn = self.time_layer(yt, yv)

        yt = yt.reshape(B, K, -1, L).permute(0, 2, 1, 3).reshape(B, -1, K * L)
        yv = yv.reshape(B, K, -1, L).permute(0, 2, 1, 3).reshape(B, -1, K * L)
        return yt, yv, attn

    def forward_time_TV_nosep(self, yt, yv, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return yt, yv
        yt = yt.reshape(B, -1, K, L).permute(0, 2, 1, 3).reshape(B * K, -1, L)
        yv = yv.reshape(B, -1, K, L).permute(0, 2, 1, 3).reshape(B * K, -1, L)

        y = torch.cat([yt, yv], dim=1)

        if self.is_linear:
            y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb, timesteps_emb=None, context=None):
        """
        x: (B, channels, K, L)
        cond_info: (B, feature_emb+1, K, L)
        diffusion_emb: (B, diffusion_embedding_dim)
        context: (B, context_dim, context_L)
        """
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B, channel, 1)
        y = x + diffusion_emb # (B, channel, K*L)

        # 先融合时间戳再融合文本
        if self.is_linear:
            y = self.forward_time(y, base_shape) # (B, channel, K*L)
        elif timesteps_emb is not None:
            timesteps_emb, y, atten_series_timestamp = self.forward_time_TV(timesteps_emb, y, base_shape)
        else:
            y = self.forward_time(y, base_shape) # (B, channel, K*L)
        y = self.forward_feature(y, base_shape)  # (B, channel, K*L)
        
        if self.with_text and context is not None:
            context = self.channel_proj(context.permute(0,2,1)).permute(0,2,1) # (B, channel, context_L)
            y, attn_series_text = self.cross_modal_layer(y, context) # (B, channel, K*L)
            # y, attn_series_text = self.cross_modal_layer(context, y) # (B, channel, K*L)

        # 先融合文本再融合时间戳
        # if self.with_text and context is not None:
        #     context = self.channel_proj(context.permute(0,2,1)).permute(0,2,1) # (B, channel, context_L)
        #     y, _ = self.cross_modal_layer(y, context) # (B, channel, K*L)
        # if self.is_linear:
        #     y = self.forward_time(y, base_shape) # (B, channel, K*L)
        # elif timesteps_emb is not None:
        #     timesteps_emb, y = self.forward_time_TV(timesteps_emb, y, base_shape)
        # else:
        #     y = self.forward_time(y, base_shape) # (B, channel, K*L)
        # y = self.forward_feature(y, base_shape)  # (B, channel, K*L)

        y = self.attn_drop(y)

        y = self.mid_projection(y)  # (B, 2*channel, K*L)
        y = self.dropout(y)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info) # (B, 2*channel, K*L)
        y = y + cond_info # (B, 2*channel, K*L)

        gate, filter = torch.chunk(y, 2, dim=1) # (B, channels, K*L) * 2
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B, channel, K*L)
        y = self.output_projection(y) # (B, 2*channel, K*L)
        y = self.dropout(y)

        residual, skip = torch.chunk(y, 2, dim=1) # (B, channels, K*L) * 2
        x = x.reshape(base_shape) # (B, channels, K, L)
        residual = residual.reshape(base_shape) # (B, channels, K, L)
        skip = skip.reshape(base_shape) # (B, channels, K, L)
        return (x + residual) / math.sqrt(2.0), skip, timesteps_emb, atten_series_timestamp, attn_series_text # (B, channels, K, L), (B, channels, K, L), ...


class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim=2, mode_num=0):
        super().__init__()
        self.channels = config["channels"]
        self.mode_num = mode_num
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        self.lookback_len = config["lookback_len"]
        self.pred_len = config["pred_len"]
        self.with_timestep = config["with_timestep"]
        dropout = config["dropout"]
        attn_drop = config["attn_drop"]
        self.pre_norm = config["pre_norm"]
        self.time_weight= config["time_weight"]
        self.save_attn = config["save_attn"]

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        # if mode_num > 1:
        #     self.fusion_projection = Conv1d_with_init(self.channels * mode_num, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.dropout = nn.Dropout(dropout)

        if self.with_timestep:
            self.timestep_projection1 = Conv1d_with_init(self.channels//4, self.channels//4, 1)
            self.timestep_projection2 = Conv1d_with_init(self.channels//4, 1, 1)
            nn.init.zeros_(self.timestep_projection2.weight)
            self.weight_mlp = nn.Sequential(
                nn.Linear(self.lookback_len, 16),
                nn.GELU(),
                nn.Linear(16, 2),
                nn.Softmax(dim=-1)
            )   

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    is_linear=config["is_linear"],
                    with_text=config["with_texts"],
                    context_dim=config["context_dim"],
                    dropout=dropout,
                    attn_drop=attn_drop,
                    pre_norm=self.pre_norm,
                    pred_len=self.pred_len
                )
                for _ in range(config["layers"])
            ]
        )

    # def forward(self, x, cond_info, diffusion_step, cfg_mask, timestep_emb=None, size_emb=None, context=None):
    #     B, inputdim, K, L = x.shape
    #     obs = x[:, 0, :, :self.lookback_len]
    #     # cond_info: (B, side_info_dim, K, L)
    #     # timestep_emb: (B, diff_dim//4, K, L)
    #     # if cfg_mask is not None:
    #     #     cond_info_mask = cfg_mask[:, None, None, None].repeat(1, cond_info.shape[1], K, L)
    #         # cond_info = cond_info * cond_info_mask
    #     x = x.reshape(B, inputdim, K * L)
    #     x = self.input_projection(x) # (B, channels, K * L)
    #     x = self.dropout(x)
    #     x = F.relu(x)
    #     if timestep_emb is not None:
    #         timestep_emb = timestep_emb.reshape(B, -1, K*L)
    #         # if cfg_mask is not None:
    #         #     timestep_emb_mask = cfg_mask[:, None, None].repeat(1, timestep_emb.shape[1], timestep_emb.shape[2])
    #         #     timestep_emb = timestep_emb * timestep_emb_mask
    #         # x = torch.cat([x, timestep_emb], dim=1)
    #     if size_emb is not None:
    #         size_emb = size_emb.reshape(B, -1, K*L)
    #         x = torch.cat([x,  size_emb], dim=1)
    #     if context is not None and cfg_mask is not None:
    #         cfg_mask = cfg_mask[:, None, None].repeat(1, context.shape[1], context.shape[2])
    #         context = context * cfg_mask
    #     # if self.mode_num > 1:
    #     #     x = self.fusion_projection(x)
    #     x = x.reshape(B, self.channels, K, L) # (B, channels, K, L)

    #     diffusion_emb = self.diffusion_embedding(diffusion_step) # (B, diffusion_embedding_dim)

    #     skip = []
    #     timestep_emb_ls = []
    #     for layer in self.residual_layers: # 注意每一层残差连接层都要重复输入cond_info和diffusion_emb！
    #         x, skip_connection, timestep_emb = layer(x, cond_info, diffusion_emb, timestep_emb, context) # (B, channels, K, L), (B, channels, K, L)
    #         skip.append(skip_connection)
    #         timestep_emb_ls.append(timestep_emb)

    #     x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers)) # (B, channels, K, L)
    #     x = x.reshape(B, self.channels, K * L) # (B,channel,K*L)
    #     x = self.output_projection1(x) # (B, channel, K*L)
    #     x = self.dropout(x)
    #     x = F.relu(x)
    #     x = self.output_projection2(x) # (B, 1, K*L)
    #     x = x.reshape(B, K, L)

    #     if timestep_emb is not None:
    #         timestep_pred = torch.sum(torch.stack(timestep_emb_ls), dim=0) / math.sqrt(len(self.residual_layers)) # (B, channels//4, K, L)
    #         timestep_pred = timestep_pred.reshape(B, self.channels//4, K * L)
    #         timestep_pred = self.timestep_projection1(timestep_pred)
    #         timestep_pred = F.relu(timestep_pred)
    #         timestep_pred = self.timestep_projection2(timestep_pred)
    #         timestep_pred = timestep_pred.reshape(B, K, L)

    #         error = timestep_pred[:, :, :self.lookback_len] - obs
    #         conb_w = self.weight_mlp(error).unsqueeze(2) # (B, K, 1, 2)
    #         x = torch.stack([x, timestep_pred], dim=-1) # (B, K, L, 2)
    #         x = torch.sum(x * conb_w, dim=-1) # (B, K, L)

    #     return x # (B, K, L)

    def forward(self, x, cond_info, diffusion_step, cfg_mask, timestep_emb=None, size_emb=None, context=None):
        B, inputdim, K, L = x.shape
        obs = x[:, 0, :, :self.lookback_len]
        # cond_info: (B, side_info_dim, K, L)
        # timestep_emb: (B, diff_dim//4, K, L)
        # if cfg_mask is not None:
        #     cond_info_mask = cfg_mask[:, None, None, None].repeat(1, cond_info.shape[1], K, L)
            # cond_info = cond_info * cond_info_mask
        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x) # (B, channels, K * L)
        x = self.dropout(x)
        x = F.relu(x)
        if timestep_emb is not None:
            timestep_emb = timestep_emb.reshape(B, -1, K*L)
            # if cfg_mask is not None:
            #     timestep_emb_mask = cfg_mask[:, None, None].repeat(1, timestep_emb.shape[1], timestep_emb.shape[2])
            #     timestep_emb = timestep_emb * timestep_emb_mask
            # x = torch.cat([x, timestep_emb], dim=1)
        if size_emb is not None:
            size_emb = size_emb.reshape(B, -1, K*L)
            x = torch.cat([x,  size_emb], dim=1)
        if context is not None and cfg_mask is not None:
            cfg_mask = cfg_mask[:, None, None].repeat(1, context.shape[1], context.shape[2])
            context = context * cfg_mask
        # if self.mode_num > 1:
        #     x = self.fusion_projection(x)
        x = x.reshape(B, self.channels, K, L) # (B, channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step) # (B, diffusion_embedding_dim)

        skip = []
        timestep_emb_ls = []
        attn = []
        for layer in self.residual_layers: # 注意每一层残差连接层都要重复输入cond_info和diffusion_emb！
            x, skip_connection, timestep_emb, attn_time, attn_text = layer(x, cond_info, diffusion_emb, timestep_emb, context) # (B, channels, K, L), (B, channels, K, L)
            skip.append(skip_connection)
            timestep_emb_ls.append(timestep_emb)
            attn.append((attn_time, attn_text))

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers)) # (B, channels, K, L)
        x = x.reshape(B, self.channels, K * L) # (B,channel,K*L)
        x = self.output_projection1(x) # (B, channel, K*L)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.output_projection2(x) # (B, 1, K*L)
        x = x.reshape(B, K, L)

        if timestep_emb is not None:
            timestep_pred = torch.sum(torch.stack(timestep_emb_ls), dim=0) / math.sqrt(len(self.residual_layers)) # (B, channels//4, K, L)
            timestep_pred = timestep_pred.reshape(B, self.channels//4, K * L)
            timestep_pred = self.timestep_projection1(timestep_pred)
            timestep_pred = F.relu(timestep_pred)
            timestep_pred = self.timestep_projection2(timestep_pred)
            timestep_pred = timestep_pred.reshape(B, K, L)

            error = timestep_pred[:, :, :self.lookback_len] - obs
            conb_w = self.weight_mlp(error).unsqueeze(2) # (B, K, 1, 2)
            # x = timestep_pred * self.time_weight + x * (1 - self.time_weight)
            x = torch.stack([x, timestep_pred], dim=-1) # (B, K, L, 2)
            x = torch.sum(x * conb_w, dim=-1) # (B, K, L)
        if self.save_attn:
            return x, attn # (B, K, L), ...
        else:
            return x