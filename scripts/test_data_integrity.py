# scripts/test_data_integrity.py
"""
다운로드된 데이터의 무결성을 확인하고 구조 라벨이 제대로 포함되어 있는지 테스트
"""

import sys
sys.path.append('..')

import os
from pathlib import Path
import pandas as pd
import json
from src.data.salami_parser import SALAMIParser
import torchaudio
from collections import Counter
import argparse


def test_salami_annotations(salami_root: Path):
    """SALAMI 어노테이션 테스트"""
    print("\n" + "="*50)
    print("Testing SALAMI Annotations")
    print("="*50)
    
    annotations_dir = salami_root / "annotations"
    
    if not annotations_dir.exists():
        print("❌ Annotations directory not found!")
        return False
    
    # 어노테이션 파일 수 확인
    ann_dirs = list(annotations_dir.iterdir())
    print(f"✓ Found {len(ann_dirs)} annotation directories")
    
    # 샘플 어노테이션 파싱 테스트
    parser = SALAMIParser(str(salami_root), str(salami_root.parent / "audio"))
    
    sample_count = min(10, len(ann_dirs))
    valid_annotations = 0
    structure_types = Counter()
    
    print(f"\nTesting {sample_count} sample annotations:")
    
    for i, ann_dir in enumerate(ann_dirs[:sample_count]):
        # textfile1.txt 찾기
        ann_file = ann_dir / "textfile1.txt"
        if not ann_file.exists():
            ann_file = ann_dir / "textfile2.txt"
            
        if ann_file.exists():
            try:
                # 어노테이션 파싱
                annotations = parser.parse_annotation_file(ann_file)
                
                if annotations:
                    # 구조 시퀀스 생성
                    structure_seq = parser.create_structure_sequences(annotations)
                    
                    if structure_seq:
                        valid_annotations += 1
                        
                        # 구조 타입 수집
                        for struct_type, _, _ in structure_seq:
                            structure_types[struct_type] += 1
                        
                        # 샘플 출력
                        if i < 3:  # 처음 3개만 상세 출력
                            print(f"\n  SALAMI ID: {ann_dir.name}")
                            print(f"  Annotations: {len(annotations)}")
                            print(f"  Structure sequence: {structure_seq[:3]}...")  # 처음 3개만
                            
            except Exception as e:
                print(f"  ❌ Error parsing {ann_file}: {e}")
    
    print(f"\n✓ Successfully parsed {valid_annotations}/{sample_count} annotations")
    
    # 구조 타입 통계
    print("\nStructure type distribution:")
    for struct_type, count in structure_types.most_common():
        print(f"  {struct_type}: {count}")
    
    return True


def test_audio_files(audio_root: Path):
    """오디오 파일 테스트"""
    print("\n" + "="*50)
    print("Testing Audio Files")
    print("="*50)
    
    if not audio_root.exists():
        print("❌ Audio directory not found!")
        return False
    
    audio_files = list(audio_root.glob("*.mp3")) + list(audio_root.glob("*.wav"))
    print(f"✓ Found {len(audio_files)} audio files")
    
    if not audio_files:
        print("❌ No audio files found!")
        return False
    
    # 샘플 오디오 파일 테스트
    sample_count = min(5, len(audio_files))
    valid_audio = 0
    
    print(f"\nTesting {sample_count} sample audio files:")
    
    for audio_file in audio_files[:sample_count]:
        try:
            # 오디오 정보 로드
            info = torchaudio.info(str(audio_file))
            duration = info.num_frames / info.sample_rate
            
            print(f"  ✓ {audio_file.name}: {duration:.1f}s, {info.sample_rate}Hz, {info.num_channels}ch")
            valid_audio += 1
            
        except Exception as e:
            print(f"  ❌ Error loading {audio_file.name}: {e}")
    
    print(f"\n✓ Successfully loaded {valid_audio}/{sample_count} audio files")
    
    return True


def test_data_matching(salami_root: Path, audio_root: Path):
    """어노테이션과 오디오 매칭 테스트"""
    print("\n" + "="*50)
    print("Testing Data Matching")
    print("="*50)
    
    # 메타데이터 로드
    metadata_path = salami_root / "metadata.csv"
    if not metadata_path.exists():
        print("❌ metadata.csv not found!")
        return False
    
    metadata = pd.read_csv(metadata_path)
    
    # 오디오 파일 목록
    audio_files = list(audio_root.glob("*.mp3")) + list(audio_root.glob("*.wav"))
    audio_ids = {f.stem for f in audio_files}
    
    # 매칭 확인
    matched = 0
    total = len(metadata)
    
    for _, row in metadata.iterrows():
        salami_id = str(row['SALAMI_id'])
        
        # 어노테이션 확인
        ann_dir = salami_root / "annotations" / salami_id
        has_annotation = ann_dir.exists()
        
        # 오디오 확인
        has_audio = salami_id in audio_ids
        
        if has_annotation and has_audio:
            matched += 1
    
    match_rate = (matched / total) * 100 if total > 0 else 0
    
    print(f"✓ Total songs in metadata: {total}")
    print(f"✓ Matched (annotation + audio): {matched}")
    print(f"✓ Match rate: {match_rate:.1f}%")
    
    # 샘플 매칭 데이터 출력
    print("\nSample matched data:")
    count = 0
    for _, row in metadata.iterrows():
        if count >= 5:
            break
            
        salami_id = str(row['SALAMI_id'])
        ann_dir = salami_root / "annotations" / salami_id
        
        if ann_dir.exists() and salami_id in audio_ids:
            print(f"  ID: {salami_id}, Artist: {row.get('artist', 'Unknown')}, Title: {row.get('title', 'Unknown')}")
            count += 1
    
    return True


def test_processed_data(processed_csv: Path):
    """처리된 데이터 테스트"""
    print("\n" + "="*50)
    print("Testing Processed Data")
    print("="*50)
    
    if not processed_csv.exists():
        print("❌ Processed data not found! Run prepare_salami_data.py first.")
        return False
    
    df = pd.read_csv(processed_csv)
    print(f"✓ Loaded processed data with {len(df)} entries")
    
    # 데이터 검증
    print("\nData validation:")
    
    # 필수 컬럼 확인
    required_cols = ['salami_id', 'audio_path', 'structure_sequence', 'duration']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
        return False
    else:
        print("✓ All required columns present")
    
    # 구조 시퀀스 파싱 테스트
    valid_sequences = 0
    for idx, row in df.head(5).iterrows():
        try:
            structure_seq = json.loads(row['structure_sequence'])
            if isinstance(structure_seq, list) and len(structure_seq) > 0:
                valid_sequences += 1
                print(f"  ✓ ID {row['salami_id']}: {len(structure_seq)} structures")
        except:
            print(f"  ❌ ID {row['salami_id']}: Invalid structure sequence")
    
    print(f"\n✓ Valid structure sequences: {valid_sequences}/5")
    
    # 통계
    print("\nDataset statistics:")
    print(f"  Average duration: {df['duration'].mean():.1f}s")
    print(f"  Average structures per song: {df['num_structures'].mean():.1f}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Test SALAMI dataset integrity")
    parser.add_argument('--base-dir', type=str, default='./datasets',
                      help='Base directory containing datasets')
    parser.add_argument('--processed-csv', type=str, default='./data/train.csv',
                      help='Path to processed CSV file')
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    salami_root = base_dir / "salami-data-public"
    audio_root = base_dir / "audio"
    
    print("SALAMI Dataset Integrity Test")
    print("="*50)
    
    # 테스트 실행
    tests_passed = 0
    total_tests = 4
    
    # 1. SALAMI 어노테이션 테스트
    if test_salami_annotations(salami_root):
        tests_passed += 1
    
    # 2. 오디오 파일 테스트
    if test_audio_files(audio_root):
        tests_passed += 1
    
    # 3. 데이터 매칭 테스트
    if test_data_matching(salami_root, audio_root):
        tests_passed += 1
    
    # 4. 처리된 데이터 테스트 (있는 경우)
    processed_csv = Path(args.processed_csv)
    if processed_csv.exists():
        if test_processed_data(processed_csv):
            tests_passed += 1
    else:
        print(f"\n⚠️  Processed data not found at {processed_csv}")
        print("   Run prepare_salami_data.py to process the data")
        total_tests -= 1
    
    # 최종 결과
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✅ All tests passed! Your dataset is ready for training.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    # 다음 단계 안내
    if tests_passed >= 3:  # 최소한 기본 테스트는 통과
        print("\nNext steps:")
        if not processed_csv.exists():
            print("1. Process the data:")
            print(f"   python scripts/prepare_salami_data.py --salami_root {salami_root} --audio_root {audio_root}")
        print("2. Start training:")
        print("   python scripts/train.py")


if __name__ == "__main__":
    main()


# scripts/visualize_structures.py
"""
구조 라벨 시각화 스크립트 (선택사항)
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import pandas as pd
import argparse


def visualize_structure_sequence(structure_sequence, title="Song Structure"):
    """구조 시퀀스를 시각화"""
    
    # 색상 매핑
    color_map = {
        'intro': '#FF6B6B',
        'verse': '#4ECDC4',
        'chorus': '#45B7D1',
        'bridge': '#96CEB4',
        'outro': '#FECA57',
        'instrumental': '#DDA0DD',
        'pre-chorus': '#98D8C8',
        'break': '#F7DC6F',
        'unknown': '#CCCCCC'
    }
    
    fig, ax = plt.subplots(figsize=(12, 3))
    
    for struct_type, start, end in structure_sequence:
        color = color_map.get(struct_type, '#CCCCCC')
        ax.barh(0, end - start, left=start, height=0.5, 
                color=color, edgecolor='black', label=struct_type)
        
        # 레이블 추가
        mid = (start + end) / 2
        ax.text(mid, 0, struct_type, ha='center', va='center', fontsize=8)
    
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('Time (seconds)')
    ax.set_title(title)
    ax.set_yticks([])
    
    # 범례 (중복 제거)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.1, 1))
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize SALAMI structure sequences")
    parser.add_argument('--csv', type=str, required=True,
                      help='Path to processed CSV file')
    parser.add_argument('--num-samples', type=int, default=5,
                      help='Number of samples to visualize')
    parser.add_argument('--output-dir', type=str, default='./visualizations',
                      help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 데이터 로드
    df = pd.read_csv(args.csv)
    
    # 샘플 시각화
    for idx, row in df.head(args.num_samples).iterrows():
        salami_id = row['salami_id']
        structure_seq = json.loads(row['structure_sequence'])
        
        # 시각화
        title = f"SALAMI {salami_id} - {row.get('artist', 'Unknown')} - {row.get('title', 'Unknown')}"
        fig = visualize_structure_sequence(structure_seq, title)
        
        # 저장
        output_path = output_dir / f"structure_{salami_id}.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved visualization to {output_path}")
    
    print(f"\nVisualized {args.num_samples} samples in {output_dir}")


if __name__ == "__main__":
    main()