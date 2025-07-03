# scripts/download_datasets.py
"""
SALAMI 데이터셋과 관련 오디오 파일 다운로드 스크립트
SALAMI_download.py 참고하여 개선된 버전
"""

import os
import sys
import requests
import zipfile
import csv
import urllib.request
import urllib.error
import subprocess
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import time
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SALAMIDownloader:
    """SALAMI 데이터셋 다운로더 - SALAMI_download.py 기반"""
    
    def __init__(self, base_dir: str = "./datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        self.salami_dir = self.base_dir / "salami-data-public"
        self.audio_dir = self.base_dir / "audio"
        self.audio_dir.mkdir(exist_ok=True)
        
        # Internet Archive 메타데이터 파일들
        self.ia_metadata_url = "https://raw.githubusercontent.com/DDMAL/salami-data-public/master/metadata/id_index_internetarchive.csv"
        self.metadata_url = "https://raw.githubusercontent.com/DDMAL/salami-data-public/master/metadata.csv"
        
    def download_file_urllib(self, url: str, local_path: Path, retries: int = 3) -> bool:
        """urllib을 사용한 파일 다운로드 (SALAMI_download.py 스타일)"""
        for attempt in range(retries):
            try:
                logger.info(f"Downloading: {url}")
                req = urllib.request.Request(url)
                req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
                
                with urllib.request.urlopen(req) as response:
                    total_size = int(response.headers.get('Content-Length', 0))
                    
                    with open(local_path, 'wb') as f:
                        downloaded = 0
                        block_size = 8192
                        
                        with tqdm(total=total_size, unit='B', unit_scale=True, desc=local_path.name) as pbar:
                            while True:
                                buffer = response.read(block_size)
                                if not buffer:
                                    break
                                    
                                downloaded += len(buffer)
                                f.write(buffer)
                                pbar.update(len(buffer))
                
                return True
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
        return False
    
    def download_salami_annotations(self):
        """SALAMI 어노테이션 데이터 다운로드"""
        logger.info("Downloading SALAMI annotations...")
        
        salami_url = "https://github.com/DDMAL/salami-data-public/archive/refs/heads/master.zip"
        zip_path = self.base_dir / "salami.zip"
        
        if not self.salami_dir.exists():
            if self.download_file_urllib(salami_url, zip_path):
                logger.info("Extracting SALAMI data...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.base_dir)
                
                # 디렉토리 이름 변경
                extracted_dir = self.base_dir / "salami-data-public-master"
                if extracted_dir.exists():
                    extracted_dir.rename(self.salami_dir)
                
                zip_path.unlink()
                logger.info(f"SALAMI data downloaded to {self.salami_dir}")
            else:
                logger.error("Failed to download SALAMI data")
                return False
        else:
            logger.info(f"SALAMI data already exists at {self.salami_dir}")
            
        return True
    
    def download_internet_archive_metadata(self):
        """Internet Archive 메타데이터 다운로드"""
        logger.info("Downloading Internet Archive metadata...")
        
        # id_index_internetarchive.csv 다운로드
        ia_csv_path = self.base_dir / "id_index_internetarchive.csv"
        if not ia_csv_path.exists():
            self.download_file_urllib(self.ia_metadata_url, ia_csv_path)
            
        # metadata.csv 다운로드 (이미 있을 수도 있음)
        metadata_path = self.base_dir / "metadata.csv"
        if not metadata_path.exists():
            self.download_file_urllib(self.metadata_url, metadata_path)
            
        return ia_csv_path.exists()
    
    def parse_internet_archive_csv(self):
        """Internet Archive CSV 파일 파싱 (SALAMI_download.py 방식)"""
        ia_csv_path = self.base_dir / "id_index_internetarchive.csv"
        
        if not ia_csv_path.exists():
            logger.error("Internet Archive metadata not found")
            return []
            
        download_list = []
        
        try:
            with open(ia_csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header if exists
                
                for row in reader:
                    if len(row) >= 5:
                        salami_id = row[0]
                        url = row[4]  # URL is in the 5th column
                        
                        if url and url.startswith('http'):
                            download_list.append({
                                'id': salami_id,
                                'url': url,
                                'filename': f"{salami_id}.mp3"
                            })
                            
        except Exception as e:
            logger.error(f"Error parsing CSV: {e}")
            
        return download_list
    
    def download_audio_from_internet_archive(self, download_list: list, max_files: int = None):
        """Internet Archive에서 오디오 다운로드"""
        logger.info(f"Downloading {len(download_list)} audio files from Internet Archive...")
        
        if max_files:
            download_list = download_list[:max_files]
            
        successful = 0
        failed = 0
        
        # 순차 다운로드 (Internet Archive는 동시 다운로드 제한이 있음)
        for item in tqdm(download_list, desc="Downloading audio"):
            output_path = self.audio_dir / item['filename']
            
            if output_path.exists():
                logger.debug(f"Already exists: {item['filename']}")
                successful += 1
                continue
                
            try:
                if self.download_file_urllib(item['url'], output_path):
                    successful += 1
                else:
                    failed += 1
                    logger.error(f"Failed to download: {item['id']}")
                    
                # Internet Archive rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                failed += 1
                logger.error(f"Error downloading {item['id']}: {e}")
                
        logger.info(f"Download complete: {successful} successful, {failed} failed")
        return successful, failed
    
    def download_youtube_matches(self):
        """YouTube 매칭 정보를 사용한 다운로드"""
        logger.info("Setting up YouTube downloads...")
        
        # matching-salami 데이터 다운로드
        matching_url = "https://raw.githubusercontent.com/jblsmith/matching-salami/master/salami_youtube_pairings.csv"
        matching_csv = self.base_dir / "salami_youtube_pairings.csv"
        
        if not matching_csv.exists():
            self.download_file_urllib(matching_url, matching_csv)
            
        # yt-dlp 설치 확인
        try:
            subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
        except:
            logger.info("Installing yt-dlp...")
            subprocess.run([sys.executable, "-m", "pip", "install", "yt-dlp"])
            
        return matching_csv
    
    def download_single_youtube(self, youtube_id: str, output_path: Path) -> bool:
        """단일 YouTube 비디오 다운로드"""
        try:
            cmd = [
                "yt-dlp",
                "-x",  # Extract audio
                "--audio-format", "mp3",
                "--audio-quality", "0",
                "-o", str(output_path),
                f"https://www.youtube.com/watch?v={youtube_id}"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Error downloading YouTube {youtube_id}: {e}")
            return False
    
    def create_audio_mapping(self):
        """오디오 파일 매핑 생성"""
        logger.info("Creating audio file mapping...")
        
        # 모든 가능한 소스에서 매핑 생성
        mapping = {}
        
        # 1. Internet Archive 매핑
        ia_csv = self.base_dir / "id_index_internetarchive.csv"
        if ia_csv.exists():
            df = pd.read_csv(ia_csv, header=None)
            for _, row in df.iterrows():
                if len(row) >= 5:
                    salami_id = str(row[0])
                    mapping[salami_id] = {
                        'source': 'internet_archive',
                        'url': row[4]
                    }
                    
        # 2. YouTube 매핑
        yt_csv = self.base_dir / "salami_youtube_pairings.csv"
        if yt_csv.exists():
            df = pd.read_csv(yt_csv)
            for _, row in df.iterrows():
                salami_id = str(row['salami_id'])
                if 'youtube_id' in row and pd.notna(row['youtube_id']):
                    mapping[salami_id] = {
                        'source': 'youtube',
                        'youtube_id': row['youtube_id']
                    }
                    
        # 3. 로컬 파일 확인
        for audio_file in self.audio_dir.glob("*"):
            if audio_file.suffix in ['.mp3', '.wav', '.flac']:
                salami_id = audio_file.stem
                if salami_id in mapping:
                    mapping[salami_id]['local_file'] = str(audio_file)
                    
        # 매핑 저장
        mapping_path = self.base_dir / "audio_mapping.json"
        with open(mapping_path, 'w') as f:
            json.dump(mapping, f, indent=2)
            
        logger.info(f"Created mapping for {len(mapping)} files")
        return mapping
    
    def verify_dataset_integrity(self):
        """데이터셋 무결성 확인"""
        logger.info("\nVerifying dataset integrity...")
        
        report = {
            'salami_annotations': False,
            'metadata': False,
            'audio_files': 0,
            'matched_files': 0,
            'total_annotations': 0
        }
        
        # SALAMI 어노테이션 확인
        if self.salami_dir.exists():
            ann_dir = self.salami_dir / "annotations"
            if ann_dir.exists():
                report['salami_annotations'] = True
                report['total_annotations'] = len(list(ann_dir.iterdir()))
                
        # 메타데이터 확인
        metadata_path = self.salami_dir / "metadata.csv"
        if metadata_path.exists():
            report['metadata'] = True
            
        # 오디오 파일 확인
        audio_files = list(self.audio_dir.glob("*.mp3")) + list(self.audio_dir.glob("*.wav"))
        report['audio_files'] = len(audio_files)
        
        # 매칭 확인
        if metadata_path.exists():
            metadata = pd.read_csv(metadata_path)
            for _, row in metadata.iterrows():
                salami_id = str(row['SALAMI_id'])
                if any(f.stem == salami_id for f in audio_files):
                    report['matched_files'] += 1
                    
        # 리포트 출력
        print("\n" + "="*50)
        print("DATASET VERIFICATION REPORT")
        print("="*50)
        print(f"✓ SALAMI Annotations: {'Found' if report['salami_annotations'] else 'Not Found'}")
        print(f"  - Total annotations: {report['total_annotations']}")
        print(f"✓ Metadata: {'Found' if report['metadata'] else 'Not Found'}")
        print(f"✓ Audio Files: {report['audio_files']}")
        print(f"✓ Matched Files: {report['matched_files']}/{report['total_annotations']}")
        print("="*50)
        
        return report
    
    def download_all(self, use_youtube: bool = False, max_files: int = None):
        """전체 다운로드 프로세스"""
        logger.info("Starting complete SALAMI download process...")
        
        # 1. SALAMI 어노테이션 다운로드
        if not self.download_salami_annotations():
            return
            
        # 2. Internet Archive 메타데이터 다운로드
        if not self.download_internet_archive_metadata():
            logger.warning("Could not download Internet Archive metadata")
            
        # 3. Internet Archive 오디오 다운로드
        download_list = self.parse_internet_archive_csv()
        if download_list:
            self.download_audio_from_internet_archive(download_list, max_files)
            
        # 4. YouTube 다운로드 (선택적)
        if use_youtube:
            self.download_youtube_matches()
            # YouTube 다운로드는 별도로 구현
            
        # 5. 매핑 생성
        self.create_audio_mapping()
        
        # 6. 데이터셋 검증
        self.verify_dataset_integrity()


def main():
    parser = argparse.ArgumentParser(
        description="Download SALAMI dataset and audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download only SALAMI annotations
  python download_datasets.py --annotations-only
  
  # Download annotations and first 100 audio files
  python download_datasets.py --max-files 100
  
  # Download everything including YouTube matches
  python download_datasets.py --use-youtube
        """
    )
    
    parser.add_argument('--base-dir', type=str, default='./datasets',
                        help='Base directory for datasets (default: ./datasets)')
    parser.add_argument('--annotations-only', action='store_true',
                        help='Download only SALAMI annotations without audio')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of audio files to download')
    parser.add_argument('--use-youtube', action='store_true',
                        help='Also download YouTube matches (requires yt-dlp)')
    
    args = parser.parse_args()
    
    # 다운로더 생성
    downloader = SALAMIDownloader(args.base_dir)
    
    if args.annotations_only:
        # 어노테이션만 다운로드
        downloader.download_salami_annotations()
        downloader.verify_dataset_integrity()
    else:
        # 전체 다운로드
        downloader.download_all(
            use_youtube=args.use_youtube,
            max_files=args.max_files
        )
    
    # 다음 단계 안내
    print("\n" + "="*50)
    print("NEXT STEPS:")
    print("="*50)
    print("1. Process the data:")
    print(f"   python scripts/prepare_salami_data.py \\")
    print(f"       --salami_root {downloader.salami_dir} \\")
    print(f"       --audio_root {downloader.audio_dir}")
    print("\n2. Start training:")
    print("   python scripts/train.py")
    print("="*50)


if __name__ == "__main__":
    main()