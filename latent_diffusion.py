import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from models.audio_encoder import AudioEncoder
from models.face_encoder import FaceEncoder
from models.syncnet import SyncNet

class TimeEmbedding(nn.Module):
    """时间步嵌入模块，将时间步转换为高维特征"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.half_dim = dim // 2
        self.emb = math.log(10000) / (self.half_dim - 1)
        self.emb = torch.exp(torch.arange(self.half_dim) * -self.emb)

    def forward(self, x):
        x = x.unsqueeze(1)
        emb = x * self.emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        return emb

class CrossAttention(nn.Module):
    """交叉注意力模块，用于融合音频和视频特征"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim **-0.5
        
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, x, context):
        batch_size = x.shape[0]
        
        # 计算QKV
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # 重塑为多头
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # 计算注意力
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # 应用上下文
        context = rearrange(context, 'b n (h d) -> b h n d', h=self.num_heads)
        attn = torch.matmul(attn, context)
        
        # 输出投影
        attn = rearrange(attn, 'b h n d -> b n (h d)')
        return self.out_proj(attn)

class UNetResBlock(nn.Module):
    """UNet残差块，包含时间嵌入和交叉注意力"""
    def __init__(self, in_channels, out_channels, time_dim, num_heads, context_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        
        # 时间步嵌入
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels)
        )
        
        # 交叉注意力（如果提供上下文维度）
        self.attention = None
        if context_dim is not None:
            self.attention = CrossAttention(out_channels, num_heads)
            self.context_proj = nn.Linear(context_dim, out_channels)
        
        # 残差连接
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, t, context=None):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        
        # 加入时间嵌入
        time_emb = self.time_mlp(t)
        h = h + time_emb[:, :, None, None]
        
        # 应用注意力
        if self.attention is not None and context is not None:
            b, c, h_, w_ = h.shape
            h_flat = rearrange(h, 'b c h w -> b (h w) c')
            context_proj = self.context_proj(context)
            attn_out = self.attention(h_flat, context_proj)
            h = rearrange(attn_out, 'b (h w) c -> b c h w', h=h_, w=w_) + h
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        
        # 残差连接
        return h + self.residual_conv(x)

class DownBlock(nn.Module):
    """下采样模块"""
    def __init__(self, in_channels, out_channels, time_dim, num_heads, context_dim=None):
        super().__init__()
        self.block = UNetResBlock(in_channels, out_channels, time_dim, num_heads, context_dim)
        self.downsample = nn.Conv2d(out_channels, out_channels, 4, 2, 1)
        
    def forward(self, x, t, context=None):
        x = self.block(x, t, context)
        x_down = self.downsample(x)
        return x, x_down

class UpBlock(nn.Module):
    """上采样模块"""
    def __init__(self, in_channels, out_channels, time_dim, num_heads, context_dim=None):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1)
        self.block = UNetResBlock(in_channels + out_channels, out_channels, time_dim, num_heads, context_dim)
        
    def forward(self, x, skip_x, t, context=None):
        x = self.upsample(x)
        x = torch.cat([x, skip_x], dim=1)
        x = self.block(x, t, context)
        return x

class LatentDiffusionModel(nn.Module):
    """完整的潜在扩散模型，实现音频驱动的唇形生成"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config['model']['image_size']
        self.base_channels = config['model']['base_channels']
        self.diffusion_steps = config['model']['diffusion_steps']
        
        # 时间嵌入
        self.time_embedding = TimeEmbedding(config['model']['time_emb_dim'])
        self.time_mlp = nn.Sequential(
            nn.Linear(config['model']['time_emb_dim'], config['model']['time_emb_dim']),
            nn.SiLU(),
            nn.Linear(config['model']['time_emb_dim'], config['model']['time_emb_dim'])
        )
        
        # 条件编码器
        self.audio_encoder = AudioEncoder(
            input_dim=config['data']['n_mfcc'],
            hidden_dim=512,
            output_dim=config['model']['audio_dim']
        )
        self.face_encoder = FaceEncoder(
            input_dim=68*2,  # 68个面部特征点，每个有x,y坐标
            hidden_dim=512,
            output_dim=config['model']['face_dim']
        )
        
        # 条件投影
        self.audio_proj = nn.Linear(config['model']['audio_dim'], config['model']['time_emb_dim'])
        self.face_proj = nn.Linear(config['model']['face_dim'], config['model']['time_emb_dim'])
        
        # 初始卷积
        self.init_conv = nn.Conv2d(
            config['model']['in_channels'], 
            self.base_channels, 
            3, 
            padding=1
        )
        
        # 下采样路径
        self.down_blocks = nn.ModuleList([
            DownBlock(
                self.base_channels, 
                self.base_channels * 2, 
                config['model']['time_emb_dim'],
                config['model']['num_heads'],
                config['model']['audio_dim']
            ),
            DownBlock(
                self.base_channels * 2, 
                self.base_channels * 4, 
                config['model']['time_emb_dim'],
                config['model']['num_heads'],
                config['model']['audio_dim']
            ),
            DownBlock(
                self.base_channels * 4, 
                self.base_channels * 8, 
                config['model']['time_emb_dim'],
                config['model']['num_heads'],
                config['model']['audio_dim']
            ),
        ])
        
        # 中间块
        self.mid_block = UNetResBlock(
            self.base_channels * 8, 
            self.base_channels * 8, 
            config['model']['time_emb_dim'],
            config['model']['num_heads'],
            config['model']['audio_dim']
        )
        
        # 上采样路径
        self.up_blocks = nn.ModuleList([
            UpBlock(
                self.base_channels * 8, 
                self.base_channels * 4, 
                config['model']['time_emb_dim'],
                config['model']['num_heads'],
                config['model']['audio_dim']
            ),
            UpBlock(
                self.base_channels * 4, 
                self.base_channels * 2, 
                config['model']['time_emb_dim'],
                config['model']['num_heads'],
                config['model']['audio_dim']
            ),
            UpBlock(
                self.base_channels * 2, 
                self.base_channels, 
                config['model']['time_emb_dim'],
                config['model']['num_heads'],
                config['model']['audio_dim']
            ),
        ])
        
        # 最终输出
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, self.base_channels),
            nn.SiLU(),
            nn.Conv2d(self.base_channels, config['model']['out_channels'], 3, padding=1)
        )
        
        # 时间一致性损失模块 (TREPA)
        self.temporal_aligner = nn.Sequential(
            nn.Conv3d(self.base_channels, self.base_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.ReLU(),
            nn.Conv3d(self.base_channels, self.base_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        )
        
        # SyncNet监督头
        self.syncnet = SyncNet()
        
        # 注册噪声 scheduler
        self.register_buffer('betas', self._get_betas())
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))

    def _get_betas(self):
        """生成噪声调度的beta参数"""
        scale = 1000 / self.diffusion_steps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, self.diffusion_steps, dtype=torch.float32)

    def q_sample(self, x0, t, noise=None):
        """扩散过程前向步骤：从x0加噪到xt"""
        if noise is None:
            noise = torch.randn_like(x0)
            
        sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt().unsqueeze(1).unsqueeze(1).unsqueeze(1)
        one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t]).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        
        return sqrt_alphas_cumprod_t * x0 + torch.sqrt(one_minus_alphas_cumprod_t) * noise

    def predict_eps_from_x0(self, x0, xt, t):
        """从x0和xt预测噪声"""
        sqrt_alphas_cumprod_t = self.alphas_cumprod[t].sqrt().unsqueeze(1).unsqueeze(1).unsqueeze(1)
        one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t]).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        
        return (xt - sqrt_alphas_cumprod_t * x0) / torch.sqrt(one_minus_alphas_cumprod_t)

    def forward(self, x, t, audio_feat, face_feat):
        """前向传播：预测噪声"""
        # 计算时间嵌入
        time_emb = self.time_embedding(t)
        time_emb = self.time_mlp(time_emb)
        
        # 计算条件嵌入
        batch_size, seq_len = audio_feat.shape[0], audio_feat.shape[1]
        
        # 编码音频和面部特征
        audio_encoded = self.audio_encoder(audio_feat)  # (batch, seq_len, audio_dim)
        face_encoded = self.face_encoder(face_feat)    # (batch, seq_len, face_dim)
        
        # 对序列特征取平均
        audio_avg = torch.mean(audio_encoded, dim=1)  # (batch, audio_dim)
        face_avg = torch.mean(face_encoded, dim=1)    # (batch, face_dim)
        
        # 投影到时间嵌入维度
        audio_emb = self.audio_proj(audio_avg)
        face_emb = self.face_proj(face_avg)
        
        # 合并所有条件
        cond_emb = time_emb + audio_emb + face_emb
        
        # 初始卷积
        x = self.init_conv(x)
        
        # 下采样
        skips = []
        for down in self.down_blocks:
            h, x = down(x, cond_emb, audio_avg)
            skips.append(h)
        
        # 中间块
        x = self.mid_block(x, cond_emb, audio_avg)
        
        # 上采样
        for i, up in enumerate(self.up_blocks):
            x = up(x, skips[-(i+1)], cond_emb, audio_avg)
        
        # 最终输出（预测的噪声）
        return self.final_conv(x)
    
    def temporal_consistency_loss(self, frames):
        """计算时间一致性损失"""
        # frames shape: (batch, seq_len, channels, height, width)
        batch_size, seq_len, channels, height, width = frames.shape
        
        # 调整形状以适应3D卷积 (batch, channels, seq_len, height, width)
        frames_reshaped = frames.permute(0, 2, 1, 3, 4)
        
        # 应用时间卷积
        aligned = self.temporal_aligner(frames_reshaped)
        
        # 计算与原始帧的差异
        loss = F.l1_loss(aligned, frames_reshaped)
        return loss
    
    def syncnet_loss(self, generated_frames, audio_feat):
        """计算SyncNet损失，确保音频与视频同步"""
        return self.syncnet(generated_frames, audio_feat)
    
    @torch.no_grad()
    def p_sample(self, xt, t, audio_feat, face_feat):
        """扩散过程反向步骤：从xt去噪一步"""
        betas_t = self.betas[t]
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - self.alphas_cumprod[t])
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t])
        
        # 预测噪声
        noise_pred = self.forward(xt, t, audio_feat, face_feat)
        
        # 计算x_{t-1}
        x_prev = sqrt_recip_alphas_t * (xt - betas_t / sqrt_one_minus_alphas_cumprod_t * noise_pred)
        
        if t > 0:
            var = betas_t
            noise = torch.randn_like(xt)
            x_prev = x_prev + torch.sqrt(var) * noise
            
        return x_prev

    @torch.no_grad()
    def sample(self, audio_feat, face_feat, num_steps=None, guidance_scale=3.0):
        """完整采样过程：从噪声生成视频帧"""
        num_steps = num_steps or self.diffusion_steps
        batch_size, seq_len = audio_feat.shape[0], audio_feat.shape[1]
        device = audio_feat.device
        
        # 初始化噪声
        x = torch.randn(
            batch_size, 
            3, 
            self.image_size, 
            self.image_size, 
            device=device
        )
        
        # 时间步
        timesteps = torch.linspace(0, self.diffusion_steps-1, num_steps, device=device).long()
        
        # 生成进度条
        progress_bar = tqdm(enumerate(reversed(timesteps)), total=num_steps)
        
        for i, step in progress_bar:
            # 扩展到批次大小
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)
            
            # 分类器引导
            if guidance_scale > 0:
                # 双重采样用于引导
                x_input = torch.cat([x, x], dim=0)
                t_input = torch.cat([t, t], dim=0)
                audio_input = torch.cat([audio_feat, torch.zeros_like(audio_feat)], dim=0)
                face_input = torch.cat([face_feat, torch.zeros_like(face_feat)], dim=0)
                
                # 预测噪声
                noise_pred = self.forward(x_input, t_input, audio_input, face_input)
                
                # 拆分预测结果
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                
                # 去噪一步
                x = self.p_sample(x, t, audio_feat, face_feat)
            else:
                # 无引导采样
                x = self.p_sample(x, t, audio_feat, face_feat)
            
            # 更新进度
            progress_bar.set_description(f"采样步骤 {i+1}/{num_steps}")
        
        # 转换为0-1范围
        x = torch.sigmoid(x)
        return x
    