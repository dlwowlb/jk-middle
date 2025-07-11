# src/data/salami_parser.py - 개선된 버전
import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
import torchaudio

class SALAMIParser:
    """SALAMI 데이터셋 파싱 및 전처리 - 동기화 문제 해결"""
    
    # 구조 매핑 정의
    STRUCTURE_MAPPING = {
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
                            label_clean = re.sub(r'[0-9]+', '', label).strip().lower()
                            
                            annotations.append({
                                'time': time,
                                'raw_label': label,
                                'label': label_clean,
                                'type': self._map_label_to_structure(label_clean)
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
    
    def find_audio_file(self, salami_id: str) -> Optional[Tuple[Path, float]]:
        """
        SALAMI ID에 해당하는 오디오 파일 찾기 및 실제 길이 반환
        Returns: (audio_path, duration) or None
        """
        # 다양한 명명 패턴 시도
        patterns = [
            f"{salami_id}.*",
            f"SALAMI_{salami_id}.*",
            f"*_{salami_id}.*",
            f"{salami_id:0>3}.*",  # 3자리 패딩
            f"{salami_id:0>4}.*"   # 4자리 패딩
        ]
        
        for pattern in patterns:
            matches = list(self.audio_root.glob(pattern))
            if matches:
                # 오디오 파일 확장자 확인
                for match in matches:
                    if match.suffix.lower() in ['.mp3', '.wav', '.flac', '.m4a']:
                        try:
                            # 실제 오디오 길이 확인
                            info = torchaudio.info(str(match))
                            duration = info.num_frames / info.sample_rate
                            return match, duration
                        except Exception as e:
                            print(f"Error loading audio {match}: {e}")
                            continue
                        
        return None
    
    def validate_annotation_audio_sync(self, annotations: List[Dict], audio_duration: float) -> bool:
        """어노테이션과 오디오 길이 동기화 확인"""
        if not annotations:
            return False
            
        # 어노테이션의 마지막 시간
        last_annotation_time = annotations[-1]['time']
        
        # 길이 차이가 10% 이내면 유효한 것으로 간주
        time_diff = abs(last_annotation_time - audio_duration)
        tolerance = max(audio_duration * 0.1, 5.0)  # 10% 또는 최소 5초
        
        if time_diff > tolerance:
            print(f"Warning: Annotation duration ({last_annotation_time:.1f}s) vs Audio duration ({audio_duration:.1f}s), diff: {time_diff:.1f}s")
            return False
        
        return True
    
    def process_dataset(self, min_duration: float = 10.0, max_duration: float = 600.0) -> pd.DataFrame:
        """전체 데이터셋 처리 및 매칭 - 동기화 확인 포함"""
        print("Processing SALAMI dataset with synchronization validation...")
        
        # 사용 가능한 오디오 파일 목록 먼저 생성
        available_audio = {}
        print(f"Scanning audio files in {self.audio_root}...")
        
        for audio_file in self.audio_root.glob("*"):
            if audio_file.suffix.lower() in ['.mp3', '.wav', '.flac', '.m4a']:
                # 파일명에서 SALAMI ID 추출 시도
                stem = audio_file.stem
                # 숫자만 추출 (가장 긴 연속 숫자)
                numbers = re.findall(r'\d+', stem)
                if numbers:
                    # 가장 긴 숫자 시퀀스를 ID로 사용
                    salami_id = max(numbers, key=len)
                    try:
                        info = torchaudio.info(str(audio_file))
                        duration = info.num_frames / info.sample_rate
                        available_audio[salami_id] = (audio_file, duration)
                    except:
                        continue
        
        print(f"Found {len(available_audio)} valid audio files")
        
        # 메타데이터 로드 (있는 경우)
        metadata = None
        if self.metadata_path.exists():
            metadata = pd.read_csv(self.metadata_path)
            print(f"Loaded metadata for {len(metadata)} songs")
        
        processed_data = []
        sync_valid = 0
        sync_invalid = 0
        
        # 오디오가 있는 항목만 처리
        for salami_id, (audio_path, audio_duration) in available_audio.items():
            # 어노테이션 디렉토리 찾기
            ann_dir = self.annotations_dir / salami_id
            if not ann_dir.exists():
                print(f"No annotation found for audio {salami_id}")
                continue
                
            # 어노테이션 파일 찾기 (textfile1.txt 우선)
            ann_file = ann_dir / 'textfile1.txt'
            if not ann_file.exists():
                ann_file = ann_dir / 'textfile2.txt'
                if not ann_file.exists():
                    print(f"No annotation file found in {ann_dir}")
                    continue
                    
            # 어노테이션 파싱
            annotations = self.parse_annotation_file(ann_file)
            if len(annotations) < 2:
                print(f"Insufficient annotations for {salami_id}")
                continue
            
            # 동기화 검증
            is_synced = self.validate_annotation_audio_sync(annotations, audio_duration)
            if not is_synced:
                sync_invalid += 1
                continue
            
            sync_valid += 1
                
            # 구조 시퀀스 생성
            structure_sequence = self.create_structure_sequences(annotations)
            if not structure_sequence:
                print(f"No valid structure sequence for {salami_id}")
                continue
            
            # 필터링 조건 확인
            if audio_duration < min_duration or audio_duration > max_duration:
                print(f"Duration out of range for {salami_id}: {audio_duration:.1f}s")
                continue
            
            if len(structure_sequence) < 3:  # 최소 3개 구조
                print(f"Too few structures for {salami_id}: {len(structure_sequence)}")
                continue
                
            # 메타데이터에서 정보 가져오기
            artist = "Unknown"
            title = "Unknown"
            if metadata is not None:
                matching_rows = metadata[metadata['SALAMI_id'].astype(str) == salami_id]
                if not matching_rows.empty:
                    row = matching_rows.iloc[0]
                    artist = row.get('artist', 'Unknown')
                    title = row.get('title', 'Unknown')
                
            # 데이터 저장
            processed_data.append({
                'SONG_ID': int(salami_id),
                'audio_path': str(audio_path),
                'annotation_path': str(ann_file),
                'structure_sequence': json.dumps(structure_sequence),
                'duration': audio_duration,  # 실제 오디오 길이 사용
                'num_structures': len(structure_sequence),
                'artist': artist,
                'title': title
            })
            
        print(f"\nProcessing Summary:")
        print(f"Synchronized pairs: {sync_valid}")
        print(f"Failed synchronization: {sync_invalid}")
        print(f"Final valid pairs: {len(processed_data)}")
            
        return pd.DataFrame(processed_data)
    
    def save_processed_data(self, output_path: str, **kwargs):
        """처리된 데이터를 저장"""
        df = self.process_dataset(**kwargs)
        
        if df.empty:
            print("No valid data to save!")
            return df
            
        df.to_csv(output_path, index=False)
        print(f"Processed {len(df)} songs and saved to {output_path}")
        
        # 통계 출력
        print("\nDataset Statistics:")
        print(f"Total songs: {len(df)}")
        print(f"Average duration: {df['duration'].mean():.1f} seconds")
        print(f"Duration range: {df['duration'].min():.1f} - {df['duration'].max():.1f} seconds")
        print(f"Average structures per song: {df['num_structures'].mean():.1f}")
        
        # 구조 타입 분포
        all_structures = []
        for _, row in df.iterrows():
            structures = json.loads(row['structure_sequence'])
            all_structures.extend([s[0] for s in structures])
        
        from collections import Counter
        structure_counts = Counter(all_structures)
        print("\nStructure type distribution:")
        for struct_type, count in structure_counts.most_common():
            print(f"  {struct_type}: {count}")
        
        return df