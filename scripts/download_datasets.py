# scripts/download_datasets.py
"""
SALAMI 데이터셋과 YouTube 매칭 오디오 다운로드 스크립트
matching-salami 프로젝트 기반
"""

import os
import sys
import requests
import zipfile
import csv
import subprocess
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import time
import logging
from typing import Dict, List, Tuple, Optional

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SALAMIYouTubeDownloader:
    """SALAMI 데이터셋과 YouTube 매칭 오디오 다운로더"""
    
    def __init__(self, base_dir: str = "./datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        self.salami_dir = self.base_dir / "salami-data-public"
        self.audio_dir = self.base_dir / "audio"
        self.audio_dir.mkdir(exist_ok=True)
        
        # matching-salami 파일 URL
        self.salami_youtube_pairings_url = "https://raw.githubusercontent.com/jblsmith/matching-salami/master/salami_youtube_pairings.csv"
        
    def download_file(self, url: str, dest_path: Path, desc: str = "Downloading") -> bool:
        """파일 다운로드 with progress bar"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(dest_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return False
    
    def download_salami_annotations(self):
        """SALAMI 어노테이션 데이터 다운로드"""
        logger.info("Downloading SALAMI annotations...")
        
        salami_url = "https://github.com/DDMAL/salami-data-public/archive/refs/heads/master.zip"
        zip_path = self.base_dir / "salami.zip"
        
        if not self.salami_dir.exists():
            if self.download_file(salami_url, zip_path, "SALAMI annotations"):
                logger.info("Extracting SALAMI data...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.base_dir)
                
                # 디렉토리 이름 변경
                extracted_dir = self.base_dir / "salami-data-public-master"
                if extracted_dir.exists():
                    extracted_dir.rename(self.salami_dir)
                
                zip_path.unlink()
                logger.info(f"✓ SALAMI data downloaded to {self.salami_dir}")
            else:
                logger.error("✗ Failed to download SALAMI data")
                return False
        else:
            logger.info(f"✓ SALAMI data already exists at {self.salami_dir}")
            
        return True
    
    def download_youtube_pairings(self) -> Path:
        """YouTube 매칭 정보 다운로드"""
        logger.info("Downloading YouTube pairings from matching-salami...")
        
        pairings_path = self.base_dir / "salami_youtube_pairings.csv"
        
        if not pairings_path.exists():
            if self.download_file(self.salami_youtube_pairings_url, pairings_path, "YouTube pairings"):
                logger.info("✓ YouTube pairings downloaded")
            else:
                logger.error("✗ Failed to download YouTube pairings")
                return None
        else:
            logger.info("✓ YouTube pairings already exist")
            
        return pairings_path
    
    def install_yt_dlp(self):
        """yt-dlp 설치 확인 및 설치"""
        try:
            subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
            logger.info("✓ yt-dlp is already installed")
        except:
            logger.info("Installing yt-dlp...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "yt-dlp"], check=True)
                logger.info("✓ yt-dlp installed successfully")
            except Exception as e:
                logger.error(f"✗ Failed to install yt-dlp: {e}")
                logger.info("Please install yt-dlp manually: pip install yt-dlp")
                return False
        return True
    
    def download_youtube_audio(self, youtube_id: str, output_path: Path, retries: int = 3) -> bool:
        """YouTube에서 오디오 다운로드"""
        # 이미 존재하면 스킵
        if output_path.exists():
            return True
            
        for attempt in range(retries):
            try:
                cmd = [
                    "yt-dlp",
                    "-x",  # 오디오만 추출
                    "--audio-format", "mp3",
                    "--audio-quality", "0",  # 최고 품질
                    "-o", str(output_path),
                    "--no-playlist",
                    "--quiet",
                    "--no-warnings",
                    f"https://www.youtube.com/watch?v={youtube_id}"
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0 and output_path.exists():
                    return True
                    
                if "Video unavailable" in result.stderr or "Private video" in result.stderr:
                    logger.warning(f"Video {youtube_id} is unavailable")
                    return False
                    
            except Exception as e:
                logger.error(f"Error downloading {youtube_id}: {e}")
                
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                
        return False
    
    def parse_youtube_pairings(self, pairings_path: Path) -> List[Dict]:
        """YouTube 페어링 CSV 파싱"""
        logger.info("Parsing YouTube pairings...")
        
        df = pd.read_csv(pairings_path)
        
        # 필요한 컬럼 확인
        required_cols = ['salami_id', 'youtube_id']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Missing required columns. Found: {df.columns.tolist()}")
            return []
        
        # YouTube ID가 있는 항목만 필터링
        df = df[df['youtube_id'].notna() & (df['youtube_id'] != '')]
        
        download_list = []
        for _, row in df.iterrows():
            salami_id = str(row['salami_id'])
            youtube_id = str(row['youtube_id'])
            
            download_list.append({
                'salami_id': salami_id,
                'youtube_id': youtube_id,
                'filename': f"{salami_id}.mp3"
            })
            
        logger.info(f"Found {len(download_list)} valid YouTube matches")
        return download_list
    
    def download_youtube_batch(self, download_list: List[Dict], max_workers: int = 4, max_files: Optional[int] = None):
        """YouTube 오디오 배치 다운로드"""
        if max_files:
            download_list = download_list[:max_files]
            
        logger.info(f"Starting download of {len(download_list)} audio files from YouTube...")
        
        successful = 0
        failed = 0
        unavailable = []
        
        # Progress bar
        with tqdm(total=len(download_list), desc="Downloading audio") as pbar:
            # ThreadPoolExecutor for parallel downloads
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_item = {
                    executor.submit(
                        self.download_youtube_audio,
                        item['youtube_id'],
                        self.audio_dir / item['filename']
                    ): item
                    for item in download_list
                }
                
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    
                    try:
                        if future.result():
                            successful += 1
                        else:
                            failed += 1
                            unavailable.append(item['salami_id'])
                    except Exception as e:
                        failed += 1
                        logger.error(f"Error with {item['salami_id']}: {e}")
                        
                    pbar.update(1)
                    pbar.set_postfix({
                        'success': successful,
                        'failed': failed
                    })
        
        # 결과 요약
        logger.info(f"\n{'='*50}")
        logger.info(f"Download Summary:")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")
        if unavailable:
            logger.info(f"  Unavailable videos: {len(unavailable)}")
            # 사용할 수 없는 비디오 ID 저장
            with open(self.base_dir / "unavailable_videos.txt", 'w') as f:
                f.write('\n'.join(unavailable))
        logger.info(f"{'='*50}\n")
        
        return successful, failed
    
    def align_audio_to_salami(self, salami_id: str):
        """오디오를 SALAMI 타이밍에 맞게 정렬 (align_audio.py 참고)"""
        # 이 부분은 sox를 사용하여 구현할 수 있습니다
        # 현재는 placeholder
        pass
    
    def create_dataset_info(self):
        """데이터셋 정보 생성"""
        logger.info("Creating dataset information...")
        
        info = {
            'salami_dir': str(self.salami_dir),
            'audio_dir': str(self.audio_dir),
            'statistics': {}
        }
        
        # SALAMI 통계
        if self.salami_dir.exists():
            ann_dir = self.salami_dir / "annotations"
            if ann_dir.exists():
                info['statistics']['total_annotations'] = len(list(ann_dir.iterdir()))
                
            metadata_path = self.salami_dir / "metadata.csv"
            if metadata_path.exists():
                metadata = pd.read_csv(metadata_path)
                info['statistics']['total_songs'] = len(metadata)
        
        # 오디오 파일 통계
        audio_files = list(self.audio_dir.glob("*.mp3"))
        info['statistics']['downloaded_audio'] = len(audio_files)
        
        # 매칭 통계
        if 'total_songs' in info['statistics']:
            matched = 0
            for audio_file in audio_files:
                salami_id = audio_file.stem
                if salami_id.isdigit():
                    matched += 1
            info['statistics']['matched_files'] = matched
            info['statistics']['match_rate'] = f"{(matched / info['statistics']['total_songs'] * 100):.1f}%"
        
        # 정보 저장
        info_path = self.base_dir / "dataset_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
            
        # 콘솔 출력
        logger.info(f"\n{'='*50}")
        logger.info("DATASET INFORMATION")
        logger.info(f"{'='*50}")
        for key, value in info['statistics'].items():
            logger.info(f"  {key}: {value}")
        logger.info(f"{'='*50}\n")
        
        return info
    
    def download_all(self, max_files: Optional[int] = None, max_workers: int = 4):
        """전체 다운로드 프로세스"""
        logger.info("Starting SALAMI YouTube download process...")
        
        # 1. SALAMI 어노테이션 다운로드
        if not self.download_salami_annotations():
            return
        
        # 2. yt-dlp 설치 확인
        if not self.install_yt_dlp():
            return
        
        # 3. YouTube pairings 다운로드
        pairings_path = self.download_youtube_pairings()
        if not pairings_path:
            return
        
        # 4. YouTube 오디오 다운로드
        download_list = self.parse_youtube_pairings(pairings_path)
        if download_list:
            self.download_youtube_batch(download_list, max_workers, max_files)
        
        # 5. 데이터셋 정보 생성
        self.create_dataset_info()
        
        # 6. 다음 단계 안내
        logger.info(f"\n{'='*50}")
        logger.info("NEXT STEPS:")
        logger.info(f"{'='*50}")
        logger.info("1. Process the downloaded data:")
        logger.info(f"   python scripts/prepare_salami_data.py \\")
        logger.info(f"       --salami_root {self.salami_dir} \\")
        logger.info(f"       --audio_root {self.audio_dir}")
        logger.info("")
        logger.info("2. Start training:")
        logger.info("   python scripts/train.py")
        logger.info(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download SALAMI dataset with YouTube audio matches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download first 100 audio files for testing
  python download_datasets.py --max-files 100
  
  # Download all available audio (may take hours)
  python download_datasets.py
  
  # Use more parallel workers for faster download
  python download_datasets.py --workers 8
        """
    )
    
    parser.add_argument('--base-dir', type=str, default='./datasets',
                        help='Base directory for datasets (default: ./datasets)')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of audio files to download (default: all)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel download workers (default: 4)')
    
    args = parser.parse_args()
    
    # 다운로더 생성 및 실행
    downloader = SALAMIYouTubeDownloader(args.base_dir)
    downloader.download_all(
        max_files=args.max_files,
        max_workers=args.workers
    )


if __name__ == "__main__":
    main()