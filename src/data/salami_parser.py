import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re


import sys
sys.path.append('..')
sys.path.append('.')


class SALAMIParser:
    """SALAMI 데이터셋 파싱 및 전처리"""
    
    # 구조 매핑 정의
    STRUCTURE_MAPPING = {
        # Primary mappings
        'intro': ['intro', 'i', 'fade_in', 'opening', 'head'],
        'verse': ['verse', 'v', 'a', 'strophe'],
        'chorus': ['chorus', 'c', 'refrain', 'b', 'hook'],
        'bridge': ['bridge', 'br', 'middle_eight', 'd'],
        'outro': ['outro', 'o', 'fade_out', 'ending', 'coda', 'tail'],
        'instrumental': ['solo', 'instrumental', 'inst', 'interlude'],
        'pre-chorus': ['pre-chorus', 'prechorus', 'pre', 'build'],
        'break': ['break', 'breakdown', 'drop']
    }
    
    def __init__(self, salami_root: str, audio_root: str):
        self.salami_root = Path(salami_root)
        self.audio_root = Path(audio_root)
        self.annotations_dir = self.salami_root / 'annotations'
        self.metadata_path = self.salami_root / 'metadata.csv'
        
    def parse_annotation_file(self, file_path: Path) -> List[Dict]:
        """단일 SALAMI 어노테이션 파일 파싱"""
        annotations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # SALAMI format: "time\tlabel"
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        try:
                            time = float(parts[0])
                            label = parts[1].strip()
                            
                            # Clean label
                            label = re.sub(r'[0-9]+', '', label)  # Remove numbers
                            label = label.strip().lower()
                            
                            annotations.append({
                                'time': time,
                                'raw_label': parts[1].strip(),
                                'label': label,
                                'type': self._map_label_to_structure(label)
                            })
                        except ValueError:
                            continue
                            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            
        return annotations
    
    def _map_label_to_structure(self, label: str) -> str:
        """레이블을 표준 구조 타입으로 매핑"""
        label_lower = label.lower()
        
        for structure_type, patterns in self.STRUCTURE_MAPPING.items():
            for pattern in patterns:
                if pattern in label_lower:
                    return structure_type
                    
        return 'unknown'
    
    def create_structure_sequences(self, annotations: List[Dict]) -> List[Tuple[str, float, float]]:
        """어노테이션을 구조 시퀀스로 변환"""
        if len(annotations) < 2:
            return []
            
        sequences = []
        for i in range(len(annotations) - 1):
            structure_type = annotations[i]['type']
            start_time = annotations[i]['time']
            end_time = annotations[i + 1]['time']
            
            # 너무 짧은 세그먼트는 제외 (0.5초 미만)
            if end_time - start_time >= 0.5:
                sequences.append((structure_type, start_time, end_time))
                
        return sequences
    
    def find_audio_file(self, SONG_ID: str) -> Optional[Path]:
        """SALAMI ID에 해당하는 오디오 파일 찾기"""
        # 다양한 명명 패턴 시도
        patterns = [
            f"{SONG_ID}.*",
            f"SALAMI_{SONG_ID}.*",
            f"*_{SONG_ID}.*",
            f"{SONG_ID:0>4}.*"  # Zero-padded
        ]
        
        for pattern in patterns:
            matches = list(self.audio_root.glob(pattern))
            if matches:
                # 오디오 파일 확장자 확인
                for match in matches:
                    if match.suffix.lower() in ['.mp3', '.wav', '.flac', '.m4a']:
                        return match
                        
        return None
    
    def process_dataset(self) -> pd.DataFrame:
        """전체 데이터셋 처리 및 매칭"""
        # 메타데이터 로드
        metadata = pd.read_csv(self.metadata_path)
        
        processed_data = []
        
        for _, row in metadata.iterrows():
            SONG_ID = str(row['SONG_ID'])
            
            # 어노테이션 디렉토리 찾기
            ann_dir = self.annotations_dir / SONG_ID
            if not ann_dir.exists():
                continue
                
            # 어노테이션 파일 찾기 (textfile1.txt 우선)
            ann_file = ann_dir / 'textfile1.txt'
            if not ann_file.exists():
                ann_file = ann_dir / 'textfile2.txt'
                if not ann_file.exists():
                    continue
                    
            # 어노테이션 파싱
            annotations = self.parse_annotation_file(ann_file)
            if len(annotations) < 2:
                continue
                
            # 구조 시퀀스 생성
            structure_sequence = self.create_structure_sequences(annotations)
            if not structure_sequence:
                continue
                
            # 오디오 파일 찾기
            audio_path = self.find_audio_file(SONG_ID)
            if not audio_path:
                continue
                
            # 데이터 저장
            processed_data.append({
                'SONG_ID': SONG_ID,
                'audio_path': str(audio_path),
                'annotation_path': str(ann_file),
                'structure_sequence': json.dumps(structure_sequence),
                'duration': annotations[-1]['time'],
                'num_structures': len(structure_sequence),
                'artist': row.get('artist', 'Unknown'),
                'title': row.get('title', 'Unknown')
            })
            
        return pd.DataFrame(processed_data)
    
    def save_processed_data(self, output_path: str):
        """처리된 데이터를 저장"""
        df = self.process_dataset()
        df.to_csv(output_path, index=False)
        print(f"Processed {len(df)} songs and saved to {output_path}")
        
        # 통계 출력
        print("\nDataset Statistics:")
        print(f"Total songs: {len(df)}")
        print(f"Average duration: {df['duration'].mean():.1f} seconds")
        print(f"Average structures per song: {df['num_structures'].mean():.1f}")
        
        return df