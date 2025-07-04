import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from einops import rearrange

class StructureConditionedDiT(nn.Module):
    """구조 조건화된 Diffusion Transformer"""
    
    def __init__(self,
                 base_dit_config: Dict,
                 structure_encoder: nn.Module,
                 conditioning_method: str = 'cross_attention'):
        super().__init__()
        
        self.structure_encoder = structure_encoder
        self.conditioning_method = conditioning_method
        
        # 기본 DiT 설정
        self.hidden_size = base_dit_config['hidden_size']
        self.num_heads = base_dit_config['num_heads']
        self.depth = base_dit_config['depth']
        self.mlp_ratio = base_dit_config.get('mlp_ratio', 4.0)
        
        # VAE 압축률 (Stable Audio 기준)
        self.compression_ratio = 512
        
        # 구조 조건화를 위한 추가 레이어
        if conditioning_method == 'cross_attention':
            self.structure_cross_attention_layers = nn.ModuleList([
                nn.MultiheadAttention(
                    self.hidden_size,
                    self.num_heads,
                    dropout=0.0,
                    batch_first=True
                ) for _ in range(self.depth // 3)  # 1/3 레이어마다
            ])
            
        elif conditioning_method == 'concat':
            # Input projection 수정
            self.structure_proj = nn.Linear(
                structure_encoder.embedding_dim,
                self.hidden_size
            )
            
        # Initialize as base DiT
        self._init_base_dit_layers()
        
    def _init_base_dit_layers(self):
        """기본 DiT 레이어 초기화"""
        # Patch embedding (for latent)
        self.latent_in = nn.Linear(8, self.hidden_size)  # 8 latent channels
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(256, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        
        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                self.hidden_size,
                self.num_heads,
                mlp_ratio=self.mlp_ratio
            ) for _ in range(self.depth)
        ])
        
        # Output layers
        self.norm_out = nn.LayerNorm(self.hidden_size)
        self.linear_out = nn.Linear(self.hidden_size, 8)  # 8 latent channels
        
    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                text_cond: Optional[torch.Tensor] = None,
                structure_sequences: Optional[List[List[Tuple[str, float, float]]]] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Noisy latents [B, 8, L]  (8 channels, L latent length)
            t: Timesteps [B]
            text_cond: Text embeddings from T5 [B, text_len, text_dim]
            structure_sequences: List of structure sequences
        """
        B, C, L = x.shape
        
        # Reshape latent for transformer: [B, L, C]
        x = rearrange(x, 'b c l -> b l c')
        
        # Project latent
        x = self.latent_in(x)
        
        # Timestep embedding
        t_emb = self.timestep_embedding(t, self.hidden_size)
        t_emb = self.time_embed(t_emb)
        
        # Structure conditioning
        structure_cond = None
        if structure_sequences is not None:
            # 전체 길이 계산 (latent space 기준)
            max_duration = max(seq[-1][2] for seq in structure_sequences if seq)
            
            if self.conditioning_method == 'cross_attention':
                # Encode structures for cross-attention
                structure_cond, structure_mask = self.structure_encoder(
                    structure_sequences, max_duration
                )
            else:
                # Encode at latent resolution
                structure_cond = self.structure_encoder.encode_at_latent_resolution(
                    structure_sequences, L, max_duration
                )
                
                if self.conditioning_method == 'concat':
                    # Project and add to input
                    structure_proj = self.structure_proj(structure_cond)
                    x = x + structure_proj
        
        # Process through DiT blocks
        cross_attn_counter = 0
        for i, block in enumerate(self.blocks):
            x = block(x, t_emb)
            
            # Apply structure cross-attention at specific layers
            if (self.conditioning_method == 'cross_attention' and 
                structure_cond is not None and
                i % (self.depth // len(self.structure_cross_attention_layers)) == 0 and
                cross_attn_counter < len(self.structure_cross_attention_layers)):
                
                x_struct, _ = self.structure_cross_attention_layers[cross_attn_counter](
                    x, structure_cond, structure_cond,
                    key_padding_mask=structure_mask
                )
                x = x + x_struct
                cross_attn_counter += 1
        
        # Output
        x = self.norm_out(x)
        x = self.linear_out(x)
        
        # Reshape back to latent format: [B, C, L]
        x = rearrange(x, 'b l c -> b c l')
        
        return x
    
    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Create sinusoidal timestep embeddings"""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class DiTBlock(nn.Module):
    """DiT block with adaptive layer norm"""
    
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )
        
        # Adaptive layer norm modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features [B, L, D]
            c: Conditioning (timestep embedding) [B, D]
        """
        # Adaptive layer norm
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Self-attention
        norm_x = self.norm1(x)
        norm_x = norm_x * (1 + scale_msa[:, None]) + shift_msa[:, None]
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + gate_msa[:, None] * attn_out
        
        # MLP
        norm_x = self.norm2(x)
        norm_x = norm_x * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        mlp_out = self.mlp(norm_x)
        x = x + gate_mlp[:, None] * mlp_out
        
        return x