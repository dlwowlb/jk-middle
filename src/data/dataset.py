import torch
import torchaudio
import json
import pandas as pd
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import numpy as np

class StructuredAudioDataset(Dataset):
    """구조 정보가 포함된 오디오 데이터셋"""
    
    def __init__(self,
                 csv_path: str,
                 sample_rate: int = 44100,
                 segment_duration: float = 30.0,
                 hop_duration: float = 10.0,
                 return_full_song: bool = False):
        
        self.data = pd.read_csv(csv_path)
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.hop_duration = hop_duration
        self.return_full_song = return_full_song
        
        # 세그먼트 생성
        if not return_full_song:
            self.segments = self._create_segments()
        else:
            self.segments = [(i, 0, row['duration']) for i, row in self.data.iterrows()]
            
    def _create_segments(self) -> List[Tuple[int, float, float]]:
        """오디오를 세그먼트로 분할"""
        segments = []
        
        for idx, row in self.data.iterrows():
            duration = row['duration']
            
            # 슬라이딩 윈도우로 세그먼트 생성
            start = 0
            while start + self.segment_duration <= duration:
                segments.append((idx, start, start + self.segment_duration))
                start += self.hop_duration
                
        return segments
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx) -> Dict:
        song_idx, start_time, end_time = self.segments[idx]
        row = self.data.iloc[song_idx]
        
        # 오디오 로드
        audio_path = row['audio_path']
        
        try:
            # 전체 오디오 정보 가져오기
            info = torchaudio.info(audio_path)
            duration = info.num_frames / info.sample_rate
            
            # 세그먼트 로드
            frame_offset = int(start_time * self.sample_rate)
            num_frames = int((end_time - start_time) * self.sample_rate)
            
            waveform, sr = torchaudio.load(
                audio_path,
                frame_offset=frame_offset,
                num_frames=num_frames
            )
            
            # 리샘플링
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
                
            # 스테레오로 변환 (Stable Audio VAE는 스테레오 입력 필요)
            if waveform.shape[0] == 1:
                # 모노를 스테레오로 복사
                waveform = waveform.repeat(2, 1)
            elif waveform.shape[0] > 2:
                # 멀티채널을 스테레오로 축소
                waveform = waveform[:2]
                
            # 정규화 [-1, 1]
            waveform = waveform / (waveform.abs().max() + 1e-8)
                
            # 패딩 (필요한 경우)
            expected_length = int((end_time - start_time) * self.sample_rate)
            if waveform.shape[1] < expected_length:
                padding = expected_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
                
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            # 에러 시 무음 반환 (스테레오)
            waveform = torch.zeros(2, int((end_time - start_time) * self.sample_rate))
            
        # 구조 시퀀스 파싱 및 세그먼트에 맞게 조정
        full_structure = json.loads(row['structure_sequence'])
        segment_structure = self._adjust_structure_to_segment(
            full_structure, start_time, end_time
        )
        
        return {
            'audio': waveform,
            'structure_sequence': segment_structure,
            'start_time': start_time,
            'end_time': end_time,
            'SONG_ID': row['SONG_ID'],
            'audio_path': audio_path,
            'sample_rate': self.sample_rate
        }
    
    def _adjust_structure_to_segment(self,
                                    full_structure: List,
                                    segment_start: float,
                                    segment_end: float) -> List[Tuple[str, float, float]]:
        """전체 구조를 세그먼트에 맞게 조정"""
        adjusted = []
        
        for struct_type, start, end in full_structure:
            # 세그먼트와 겹치는 부분만 포함
            if end > segment_start and start < segment_end:
                adj_start = max(0, start - segment_start)
                adj_end = min(segment_end - segment_start, end - segment_start)
                adjusted.append((struct_type, adj_start, adj_end))
                
        return adjusted
    
    def get_collate_fn(self):
        """배치 처리를 위한 collate function"""
        def collate_fn(batch):
            # 오디오는 같은 길이로 패딩
            audios = torch.stack([item['audio'] for item in batch])
            
            # 나머지 정보는 리스트로
            structure_sequences = [item['structure_sequence'] for item in batch]
            
            return {
                'audio': audios,
                'structure_sequences': structure_sequences,
                'SONG_IDs': [item['SONG_ID'] for item in batch],
                'start_times': torch.tensor([item['start_time'] for item in batch]),
                'end_times': torch.tensor([item['end_time'] for item in batch])
            }
            
        return collate_fn