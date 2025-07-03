import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math

class StructureEncoder(nn.Module):
    """음악 구조 정보를 인코딩하는 모듈"""
    
    STRUCTURE_TYPES = {
        'intro': 0,
        'verse': 1,
        'pre-chorus': 2,
        'chorus': 3,
        'bridge': 4,
        'instrumental': 5,
        'outro': 6,
        'break': 7,
        'unknown': 8,
        'pad': 9  # 패딩용
    }
    
    def __init__(self,
                 embedding_dim: int = 256,
                 hidden_dim: int = 512,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 max_structures: int = 50):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.max_structures = max_structures
        
        # 구조 타입 임베딩
        self.structure_embedding = nn.Embedding(
            len(self.STRUCTURE_TYPES),
            embedding_dim
        )
        
        # 시간 정보 인코딩 (시작, 끝, 길이)
        self.time_encoder = nn.Sequential(
            nn.Linear(3, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_structures, embedding_dim) * 0.02
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self,
                structure_sequences: List[List[Tuple[str, float, float]]],
                max_duration: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        구조 시퀀스를 인코딩
        
        Returns:
            encoded: [batch, max_structures, embedding_dim]
            mask: [batch, max_structures] - True for padded positions
        """
        batch_size = len(structure_sequences)
        device = next(self.parameters()).device
        
        # 최대 길이 계산
        if max_duration is None:
            max_duration = max(
                seq[-1][2] if seq else 1.0 
                for seq in structure_sequences
            )
        
        # 배치 준비
        structure_ids = torch.full(
            (batch_size, self.max_structures), 
            self.STRUCTURE_TYPES['pad'],
            dtype=torch.long,
            device=device
        )
        
        time_features = torch.zeros(
            batch_size, self.max_structures, 3,
            device=device
        )
        
        mask = torch.ones(
            batch_size, self.max_structures,
            dtype=torch.bool,
            device=device
        )
        
        # 각 시퀀스 처리
        for b, seq in enumerate(structure_sequences):
            seq_len = min(len(seq), self.max_structures)
            
            for i, (struct_type, start, end) in enumerate(seq[:seq_len]):
                # 구조 ID
                struct_id = self.STRUCTURE_TYPES.get(struct_type, self.STRUCTURE_TYPES['unknown'])
                structure_ids[b, i] = struct_id
                
                # 시간 특징 (정규화)
                time_features[b, i, 0] = start / max_duration
                time_features[b, i, 1] = end / max_duration
                time_features[b, i, 2] = (end - start) / max_duration
                
                mask[b, i] = False
        
        # 임베딩
        struct_emb = self.structure_embedding(structure_ids)
        time_emb = self.time_encoder(time_features)
        
        # 결합
        embeddings = struct_emb + time_emb + self.pos_encoding[:, :self.max_structures]
        
        # Transformer encoding
        encoded = self.transformer(embeddings, src_key_padding_mask=mask)
        
        # Output projection
        encoded = self.output_proj(encoded)
        
        return encoded, mask
    
    def encode_at_latent_resolution(self,
                                   structure_sequences: List[List[Tuple[str, float, float]]],
                                   latent_seq_len: int,
                                   max_duration: float) -> torch.Tensor:
        """
        구조 정보를 latent sequence 해상도로 인코딩
        각 latent frame이 어떤 구조에 속하는지 표시
        """
        batch_size = len(structure_sequences)
        device = next(self.parameters()).device
        
        # 각 latent position의 구조 정보
        structure_at_position = torch.zeros(
            batch_size, latent_seq_len, self.embedding_dim,
            device=device
        )
        
        for b, seq in enumerate(structure_sequences):
            for struct_type, start, end in seq:
                # Latent position 계산
                start_pos = int((start / max_duration) * latent_seq_len)
                end_pos = int((end / max_duration) * latent_seq_len)
                
                # 구조 임베딩
                struct_id = self.STRUCTURE_TYPES.get(struct_type, self.STRUCTURE_TYPES['unknown'])
                struct_emb = self.structure_embedding(
                    torch.tensor([struct_id], device=device)
                )
                
                # 시간 정보
                time_info = torch.tensor(
                    [[start/max_duration, end/max_duration, (end-start)/max_duration]],
                    device=device
                )
                time_emb = self.time_encoder(time_info)
                
                # 해당 위치에 할당
                combined_emb = struct_emb + time_emb
                structure_at_position[b, start_pos:end_pos] = combined_emb.squeeze(0)
        
        return structure_at_position