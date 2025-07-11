# src/models/structure_dit.py - 수정된 버전
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from einops import rearrange
import math

class StructureConditionedDiT(nn.Module):
    """구조 조건화된 Diffusion Transformer - Stable Audio 호환"""
    
    def __init__(self,
                 base_dit_config: Dict,
                 structure_encoder: nn.Module,
                 conditioning_method: str = 'cross_attention'):
        super().__init__()
        
        self.structure_encoder = structure_encoder
        self.conditioning_method = conditioning_method
        
        # 기본 DiT 설정 - Stable Audio 구조에 맞게 수정
        self.config = type('Config', (), base_dit_config)()  # config 객체 생성
        self.hidden_size = base_dit_config['hidden_size']
        self.num_heads = base_dit_config['num_heads']
        self.depth = base_dit_config['depth']
        self.mlp_ratio = base_dit_config.get('mlp_ratio', 4.0)
        self.in_channels = base_dit_config.get('in_channels', 64)
        
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
                ) for _ in range(max(1, self.depth // 3))  # 최소 1개
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
        """기본 DiT 레이어 초기화 - Stable Audio 스타일"""
        # Patch embedding (for latent) - Stable Audio는 채널별로 처리
        self.latent_in = nn.Linear(self.in_channels, self.hidden_size)
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(256, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        
        # Global conditioning projection (for text + timing)
        self.global_cond_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
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
        self.linear_out = nn.Linear(self.hidden_size, self.in_channels)
    
    def forward_with_structure(self,
                              hidden_states: torch.Tensor,
                              timestep: torch.Tensor,
                              encoder_hidden_states: Optional[torch.Tensor] = None,
                              global_embeds: Optional[torch.Tensor] = None,
                              structure_sequences: Optional[List[List[Tuple[str, float, float]]]] = None,
                              attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """구조 조건화가 포함된 forward pass"""
        return self.forward(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            global_embeds=global_embeds,
            structure_sequences=structure_sequences,
            attention_mask=attention_mask
        )
    
    def forward(self,
                hidden_states: torch.Tensor,
                timestep: torch.Tensor,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                global_embeds: Optional[torch.Tensor] = None,
                structure_sequences: Optional[List[List[Tuple[str, float, float]]]] = None,
                attention_mask: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """
        Forward pass - Stable Audio 파이프라인 호환
        
        Args:
            hidden_states: Noisy latents [B, C, L]
            timestep: Timesteps [B]
            encoder_hidden_states: Text embeddings from T5 [B, text_len, text_dim]
            global_embeds: Global conditioning (text + timing) [B, hidden_size]
            structure_sequences: List of structure sequences
            attention_mask: Text attention mask [B, text_len]
        """
        B, C, L = hidden_states.shape
        
        # Reshape latent for transformer: [B, L, C]
        x = rearrange(hidden_states, 'b c l -> b l c')
        
        # Project latent
        x = self.latent_in(x)
        
        # Timestep embedding
        t_emb = self.timestep_embedding(timestep, 256)
        t_emb = self.time_embed(t_emb)
        
        # Global conditioning (text + timing)
        if global_embeds is not None:
            global_cond = self.global_cond_proj(global_embeds)
            t_emb = t_emb + global_cond
        
        # Structure conditioning
        structure_cond = None
        structure_mask = None
        
        if structure_sequences is not None:
            try:
                # 전체 길이 계산 (latent space 기준)
                max_duration = max(
                    seq[-1][2] if seq else 30.0 
                    for seq in structure_sequences
                )
                
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
                        
            except Exception as e:
                print(f"Warning: Structure conditioning failed: {e}")
                # Continue without structure conditioning
        
        # Process through DiT blocks
        cross_attn_counter = 0
        cross_attn_interval = max(1, self.depth // len(self.structure_cross_attention_layers)) if hasattr(self, 'structure_cross_attention_layers') else self.depth
        
        for i, block in enumerate(self.blocks):
            x = block(x, t_emb)
            
            # Apply structure cross-attention at specific layers
            if (self.conditioning_method == 'cross_attention' and 
                structure_cond is not None and
                hasattr(self, 'structure_cross_attention_layers') and
                i % cross_attn_interval == 0 and
                cross_attn_counter < len(self.structure_cross_attention_layers)):
                
                try:
                    x_struct, _ = self.structure_cross_attention_layers[cross_attn_counter](
                        x, structure_cond, structure_cond,
                        key_padding_mask=structure_mask
                    )
                    x = x + x_struct
                    cross_attn_counter += 1
                except Exception as e:
                    print(f"Warning: Structure cross-attention failed: {e}")
        
        # Output
        x = self.norm_out(x)
        x = self.linear_out(x)
        
        # Reshape back to latent format: [B, C, L]
        x = rearrange(x, 'b l c -> b c l')
        
        # Return as object with .sample attribute (Stable Audio 호환)
        return type('Output', (), {'sample': x})()
    
    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Create sinusoidal timestep embeddings"""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
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