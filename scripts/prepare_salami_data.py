import sys
sys.path.append('..')

from src.data.salami_parser import SALAMIParser
import argparse
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--salami_root', type=str, required=True,
                      help='Path to SALAMI dataset root')
    parser.add_argument('--audio_root', type=str, required=True,
                      help='Path to audio files')
    parser.add_argument('--output_dir', type=str, default='./data',
                      help='Output directory for processed data')
    args = parser.parse_args()
    
    # SALAMI 파서 생성
    salami_parser = SALAMIParser(args.salami_root, args.audio_root)
    
    # 데이터 처리
    print("Processing SALAMI dataset...")
    df = salami_parser.save_processed_data(f"{args.output_dir}/salami_processed.csv")
    
    # Train/Val/Test 분할
    print("\nSplitting dataset...")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.125, random_state=42)
    
    # 저장
    train_df.to_csv(f"{args.output_dir}/train.csv", index=False)
    val_df.to_csv(f"{args.output_dir}/val.csv", index=False)
    test_df.to_csv(f"{args.output_dir}/test.csv", index=False)
    
    print(f"\nDataset split:")
    print(f"Train: {len(train_df)} songs")
    print(f"Val: {len(val_df)} songs")
    print(f"Test: {len(test_df)} songs")


if __name__ == "__main__":
    main()