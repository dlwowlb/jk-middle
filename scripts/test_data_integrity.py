import sys
sys.path.append('..')
sys.path.append('.')

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
    
    if len(ann_dirs) == 0:
        print("❌ No annotation directories found!")
        return False
    
    # 샘플 어노테이션 파싱 테스트
    try:
        parser = SALAMIParser(str(salami_root), str(salami_root.parent / "audio"))
    except Exception as e:
        print(f"❌ Failed to create SALAMI parser: {e}")
        return False
    
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
    if structure_types:
        print("\nStructure type distribution:")
        for struct_type, count in structure_types.most_common():
            print(f"  {struct_type}: {count}")
    
    return valid_annotations > 0


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
    
    # 메타데이터 로드 (선택사항)
    metadata_path = salami_root / "metadata.csv"
    if not metadata_path.exists():
        print("⚠️  metadata.csv not found - this is optional")
        print("   Proceeding with available audio and annotation files...")
        
        # 직접 매칭 확인
        annotations_dir = salami_root / "annotations"
        if not annotations_dir.exists():
            print("❌ Annotations directory not found!")
            return False
            
        # 오디오 파일 목록
        audio_files = list(audio_root.glob("*.mp3")) + list(audio_root.glob("*.wav"))
        audio_ids = set()
        
        # 오디오 파일에서 ID 추출
        import re
        for audio_file in audio_files:
            # 파일명에서 숫자 추출
            numbers = re.findall(r'\d+', audio_file.stem)
            if numbers:
                # 가장 긴 숫자를 ID로 사용
                audio_ids.add(max(numbers, key=len))
        
        # 어노테이션 디렉토리 목록
        ann_dirs = [d for d in annotations_dir.iterdir() if d.is_dir()]
        ann_ids = {d.name for d in ann_dirs}
        
        # 매칭 확인
        matched_ids = audio_ids & ann_ids
        
        print(f"✓ Audio files: {len(audio_files)}")
        print(f"✓ Audio IDs extracted: {len(audio_ids)}")
        print(f"✓ Annotation directories: {len(ann_dirs)}")
        print(f"✓ Matched IDs: {len(matched_ids)}")
        
        if len(matched_ids) > 0:
            match_rate = (len(matched_ids) / max(len(audio_ids), len(ann_ids))) * 100
            print(f"✓ Match rate: {match_rate:.1f}%")
            
            # 샘플 매칭 데이터 출력
            print("\nSample matched IDs:")
            for i, matched_id in enumerate(sorted(matched_ids)[:5]):
                print(f"  ID: {matched_id}")
        
        return len(matched_ids) > 0
    
    # metadata.csv가 있는 경우의 기존 로직
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
    
    # 컬럼명 확인 및 수정
    print(f"Available columns: {list(df.columns)}")
    
    # 필수 컬럼 확인 (실제 컬럼명에 맞게 수정)
    required_cols = ['SONG_ID', 'audio_path', 'structure_sequence', 'duration']  # salami_id -> SONG_ID
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return False
    else:
        print("✓ All required columns present")
    
    # 구조 시퀀스 파싱 테스트
    valid_sequences = 0
    sample_size = min(5, len(df))
    
    for idx, row in df.head(sample_size).iterrows():
        try:
            structure_seq = json.loads(row['structure_sequence'])
            if isinstance(structure_seq, list) and len(structure_seq) > 0:
                valid_sequences += 1
                print(f"  ✓ ID {row['SONG_ID']}: {len(structure_seq)} structures")
            else:
                print(f"  ❌ ID {row['SONG_ID']}: Empty structure sequence")
        except Exception as e:
            print(f"  ❌ ID {row['SONG_ID']}: Invalid structure sequence - {e}")
    
    print(f"\n✓ Valid structure sequences: {valid_sequences}/{sample_size}")
    
    # 통계
    print("\nDataset statistics:")
    print(f"  Average duration: {df['duration'].mean():.1f}s")
    if 'num_structures' in df.columns:
        print(f"  Average structures per song: {df['num_structures'].mean():.1f}")
    
    # 파일 존재 확인
    print("\nFile existence check:")
    existing_files = 0
    for idx, row in df.head(sample_size).iterrows():
        audio_path = Path(row['audio_path'])
        if audio_path.exists():
            existing_files += 1
            print(f"  ✓ {audio_path.name} exists")
        else:
            print(f"  ❌ {audio_path.name} not found")
    
    print(f"✓ Existing audio files: {existing_files}/{sample_size}")
    
    return valid_sequences > 0 and existing_files > 0


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
    elif tests_passed >= total_tests - 1:
        print("✅ Most tests passed! Dataset should work for training.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    # 다음 단계 안내
    if tests_passed >= 2:  # 최소한 기본 테스트는 통과
        print("\nNext steps:")
        if not processed_csv.exists():
            print("1. Process the data:")
            print(f"   python scripts/prepare_salami_data.py --salami_root {salami_root} --audio_root {audio_root}")
        print("2. Start training:")
        print("   python scripts/train.py --dry-run  # Test first")
        print("   python scripts/train.py           # Actual training")


if __name__ == "__main__":
    main()