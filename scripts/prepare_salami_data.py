# scripts/prepare_salami_data.py - 개선된 버전
import sys
sys.path.append('..')
sys.path.append('.')

from src.data.salami_parser import SALAMIParser
import argparse
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Prepare SALAMI dataset with synchronization validation')
    parser.add_argument('--salami_root', type=str, required=True,
                      help='Path to SALAMI dataset root')
    parser.add_argument('--audio_root', type=str, required=True,
                      help='Path to audio files')
    parser.add_argument('--output_dir', type=str, default='./data',
                      help='Output directory for processed data')
    parser.add_argument('--min_duration', type=float, default=10.0,
                      help='Minimum song duration in seconds')
    parser.add_argument('--max_duration', type=float, default=600.0,
                      help='Maximum song duration in seconds')
    parser.add_argument('--test_size', type=float, default=0.2,
                      help='Test set size (0.0-1.0)')
    parser.add_argument('--val_size', type=float, default=0.1,
                      help='Validation set size from training data (0.0-1.0)')
    parser.add_argument('--random_seed', type=int, default=42,
                      help='Random seed for splitting')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("SALAMI Dataset Preparation with Synchronization Validation")
    print("="*60)
    
    # SALAMI 파서 생성
    salami_parser = SALAMIParser(args.salami_root, args.audio_root)
    
    # 데이터 처리
    print("Processing SALAMI dataset...")
    df = salami_parser.save_processed_data(
        f"{args.output_dir}/salami_processed.csv",
        min_duration=args.min_duration,
        max_duration=args.max_duration
    )
    
    if df.empty:
        print("❌ No valid data found! Please check your audio and annotation paths.")
        print("\nTroubleshooting:")
        print("1. Verify audio files exist in", args.audio_root)
        print("2. Verify SALAMI annotations exist in", args.salami_root)
        print("3. Check if SALAMI IDs match between audio filenames and annotation directories")
        return
    
    if len(df) < 10:
        print(f"⚠️  Warning: Only {len(df)} valid samples found. This may not be enough for training.")
        print("Consider:")
        print("- Relaxing duration constraints")
        print("- Checking synchronization tolerance in SALAMIParser")
        print("- Verifying more audio files are properly named")
    
    # Train/Val/Test 분할
    print(f"\nSplitting dataset (test={args.test_size}, val={args.val_size})...")
    
    # 먼저 train+val과 test로 분할
    train_val_df, test_df = train_test_split(
        df, 
        test_size=args.test_size, 
        random_state=args.random_seed,
        stratify=None  # 구조 개수로 stratify하려면 구현 필요
    )
    
    # train과 val로 분할
    if args.val_size > 0:
        train_df, val_df = train_test_split(
            train_val_df, 
            test_size=args.val_size, 
            random_state=args.random_seed
        )
    else:
        train_df = train_val_df
        val_df = pd.DataFrame(columns=df.columns)  # 빈 DataFrame
    
    # 저장
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n✅ Dataset split completed:")
    print(f"   Train: {len(train_df)} songs → {train_path}")
    print(f"   Val:   {len(val_df)} songs → {val_path}")
    print(f"   Test:  {len(test_df)} songs → {test_path}")
    
    # 분할 통계
    print(f"\nSplit Statistics:")
    for name, subset_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        if len(subset_df) > 0:
            print(f"  {name}:")
            print(f"    Duration: {subset_df['duration'].mean():.1f}±{subset_df['duration'].std():.1f}s")
            print(f"    Structures: {subset_df['num_structures'].mean():.1f}±{subset_df['num_structures'].std():.1f}")
    
    # 데이터 품질 체크
    print(f"\n📊 Data Quality Check:")
    
    # 구조 타입 분포 확인
    import json
    from collections import Counter
    
    all_structures = []
    for _, row in df.iterrows():
        structures = json.loads(row['structure_sequence'])
        all_structures.extend([s[0] for s in structures])
    
    structure_counts = Counter(all_structures)
    print(f"  Structure types found: {len(structure_counts)}")
    print(f"  Most common structures:")
    for struct_type, count in structure_counts.most_common(5):
        percentage = (count / len(all_structures)) * 100
        print(f"    {struct_type}: {count} ({percentage:.1f}%)")
    
    # 길이 분포
    print(f"  Duration distribution:")
    print(f"    Min: {df['duration'].min():.1f}s")
    print(f"    Max: {df['duration'].max():.1f}s")
    print(f"    Mean: {df['duration'].mean():.1f}s")
    print(f"    Median: {df['duration'].median():.1f}s")
    
    # 다음 단계 안내
    print(f"\n🚀 Next Steps:")
    print("="*60)
    print("1. Verify the data:")
    print(f"   python scripts/test_data_integrity.py --processed-csv {train_path}")
    print("\n2. Start training:")
    print("   python scripts/train.py --config configs/train_config.yaml")
    print("\n3. Monitor training:")
    print("   tensorboard --logdir ./logs")
    
    # 경고 및 권장사항
    if len(train_df) < 100:
        print(f"\n⚠️  Training Recommendations:")
        print(f"   - With only {len(train_df)} training samples, consider:")
        print(f"   - Using smaller batch size (1-2)")
        print(f"   - More aggressive data augmentation")
        print(f"   - Shorter training segments")
        print(f"   - Pre-training on larger dataset first")
    
    print("="*60)


if __name__ == "__main__":
    main()