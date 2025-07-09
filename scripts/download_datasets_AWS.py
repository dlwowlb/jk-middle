# scripts/download_datasets_aws.py - AWS 환경 최적화 버전
"""
AWS CodeEditor 환경을 위한 SALAMI 데이터셋 다운로드 스크립트
YouTube 다운로드 문제 해결 및 대안 제공
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
import random

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AWSOptimizedDownloader:
    """AWS 환경에 최적화된 다운로더"""
    
    def __init__(self, base_dir: str = "./datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        self.salami_dir = self.base_dir / "salami-data-public"
        self.audio_dir = self.base_dir / "audio"
        self.audio_dir.mkdir(exist_ok=True)
        
        # User agents 리스트 (차단 우회)
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        ]
        
        # 실패한 비디오 추적
        self.failed_videos = set()
        self.success_count = 0
        
    def download_file(self, url: str, dest_path: Path, desc: str = "Downloading") -> bool:
        """안전한 파일 다운로드"""
        try:
            headers = {'User-Agent': random.choice(self.user_agents)}
            response = requests.get(url, stream=True, headers=headers, timeout=30)
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
        """SALAMI 어노테이션 다운로드"""
        logger.info("Downloading SALAMI annotations...")
        
        salami_url = "https://github.com/DDMAL/salami-data-public/archive/refs/heads/master.zip"
        zip_path = self.base_dir / "salami.zip"
        
        if not self.salami_dir.exists():
            if self.download_file(salami_url, zip_path, "SALAMI annotations"):
                logger.info("Extracting SALAMI data...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.base_dir)
                
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
        logger.info("Downloading YouTube pairings...")
        
        pairings_url = "https://raw.githubusercontent.com/jblsmith/matching-salami/master/salami_youtube_pairings.csv"
        pairings_path = self.base_dir / "salami_youtube_pairings.csv"
        
        if not pairings_path.exists():
            if self.download_file(pairings_url, pairings_path, "YouTube pairings"):
                logger.info("✓ YouTube pairings downloaded")
            else:
                logger.error("✗ Failed to download YouTube pairings")
                return None
        else:
            logger.info("✓ YouTube pairings already exist")
            
        return pairings_path
    
    def install_yt_dlp(self):
        """yt-dlp 설치 및 업데이트"""
        try:
            # 최신 버전 설치
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"], 
                         check=True, capture_output=True)
            logger.info("✓ yt-dlp updated to latest version")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to install/update yt-dlp: {e}")
            return False
    
    def test_youtube_access(self, test_video_id: str = "dQw4w9WgXcQ") -> bool:
        """YouTube 접근 테스트"""
        try:
            cmd = [
                "yt-dlp",
                "--quiet",
                "--no-warnings",
                "--print", "title",
                f"https://www.youtube.com/watch?v={test_video_id}"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info("✓ YouTube access test passed")
                return True
            else:
                logger.warning(f"YouTube access test failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.warning(f"YouTube access test error: {e}")
            return False
    
    def download_youtube_audio_robust(self, youtube_id: str, output_path: Path, retries: int = 3) -> bool:
        """강화된 YouTube 오디오 다운로드"""
        if output_path.exists():
            return True
            
        if youtube_id in self.failed_videos:
            return False
            
        for attempt in range(retries):
            try:
                # 다양한 설정으로 시도
                configs = [
                    # 기본 설정
                    {
                        "format": "bestaudio[ext=m4a]/bestaudio/best",
                        "user_agent": random.choice(self.user_agents)
                    },
                    # 더 관대한 설정
                    {
                        "format": "worst[ext=m4a]/worst",
                        "user_agent": random.choice(self.user_agents),
                        "no_check_certificate": True
                    },
                    # 매우 관대한 설정
                    {
                        "format": "worst",
                        "user_agent": random.choice(self.user_agents),
                        "no_check_certificate": True,
                        "prefer_insecure": True
                    }
                ]
                
                config = configs[min(attempt, len(configs) - 1)]
                
                cmd = [
                    "yt-dlp",
                    "-x",
                    "--audio-format", "mp3",
                    "--audio-quality", "5",  # 중간 품질
                    "-o", str(output_path),
                    "--no-playlist",
                    "--quiet",
                    "--no-warnings",
                    "--user-agent", config["user_agent"],
                    "--format", config["format"]
                ]
                
                if config.get("no_check_certificate"):
                    cmd.append("--no-check-certificate")
                if config.get("prefer_insecure"):
                    cmd.append("--prefer-insecure")
                
                cmd.append(f"https://www.youtube.com/watch?v={youtube_id}")
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0 and output_path.exists():
                    self.success_count += 1
                    return True
                    
                # 특정 오류 메시지 확인
                error_msg = result.stderr.lower()
                if any(keyword in error_msg for keyword in ["unavailable", "private", "deleted", "blocked"]):
                    logger.warning(f"Video {youtube_id} permanently unavailable")
                    self.failed_videos.add(youtube_id)
                    return False
                    
            except subprocess.TimeoutExpired:
                logger.warning(f"Timeout downloading {youtube_id}, attempt {attempt + 1}")
            except Exception as e:
                logger.warning(f"Error downloading {youtube_id}: {e}")
                
            if attempt < retries - 1:
                wait_time = (2 ** attempt) + random.uniform(1, 3)
                time.sleep(wait_time)
                
        self.failed_videos.add(youtube_id)
        return False
    
    def parse_youtube_pairings(self, pairings_path: Path) -> List[Dict]:
        """YouTube 페어링 파싱"""
        logger.info("Parsing YouTube pairings...")
        
        df = pd.read_csv(pairings_path)
        
        required_cols = ['salami_id', 'youtube_id']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Missing required columns. Found: {df.columns.tolist()}")
            return []
        
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
    
    def download_youtube_batch_aws(self, download_list: List[Dict], max_files: Optional[int] = None):
        """AWS 환경에 최적화된 YouTube 배치 다운로드"""
        if max_files:
            download_list = download_list[:max_files]
            
        logger.info(f"Starting AWS-optimized download of {len(download_list)} audio files...")
        
        # YouTube 접근 테스트
        if not self.test_youtube_access():
            logger.warning("YouTube access test failed - downloads may fail")
            
        self.success_count = 0
        failed_count = 0
        
        # 순차 다운로드 (AWS 환경에서 더 안정적)
        with tqdm(total=len(download_list), desc="Downloading audio") as pbar:
            for i, item in enumerate(download_list):
                try:
                    success = self.download_youtube_audio_robust(
                        item['youtube_id'],
                        self.audio_dir / item['filename']
                    )
                    
                    if not success:
                        failed_count += 1
                        
                    # 진행상황 업데이트
                    pbar.set_postfix({
                        'success': self.success_count,
                        'failed': failed_count,
                        'rate': f"{(self.success_count / (i + 1) * 100):.1f}%"
                    })
                    
                    # AWS 환경에서 rate limiting
                    if i % 10 == 0:
                        time.sleep(2)
                        
                except KeyboardInterrupt:
                    logger.info("Download interrupted by user")
                    break
                except Exception as e:
                    logger.error(f"Unexpected error with {item['salami_id']}: {e}")
                    failed_count += 1
                    
                finally:
                    pbar.update(1)
        
        # 결과 요약
        total_attempted = self.success_count + failed_count
        success_rate = (self.success_count / total_attempted * 100) if total_attempted > 0 else 0
        
        logger.info(f"\n{'='*50}")
        logger.info(f"AWS Download Summary:")
        logger.info(f"  Successful: {self.success_count}")
        logger.info(f"  Failed: {failed_count}")
        logger.info(f"  Success rate: {success_rate:.1f}%")
        logger.info(f"{'='*50}\n")
        
        # 실패한 비디오 목록 저장
        if self.failed_videos:
            failed_path = self.base_dir / "failed_videos_aws.txt"
            with open(failed_path, 'w') as f:
                f.write('\n'.join(self.failed_videos))
            logger.info(f"Failed video IDs saved to: {failed_path}")
        
        return self.success_count, failed_count
    
    def create_alternative_dataset_info(self):
        """대안 데이터셋 정보 생성"""
        logger.info("Creating alternative dataset information...")
        
        info = {
            'download_method': 'aws_optimized',
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
        
        # 실제 다운로드된 오디오
        audio_files = list(self.audio_dir.glob("*.mp3"))
        info['statistics']['downloaded_audio'] = len(audio_files)
        info['statistics']['success_count'] = self.success_count
        info['statistics']['failed_count'] = len(self.failed_videos)
        
        # 정보 저장
        info_path = self.base_dir / "dataset_info_aws.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
            
        logger.info(f"\n{'='*50}")
        logger.info("AWS DATASET INFORMATION")
        logger.info(f"{'='*50}")
        for key, value in info['statistics'].items():
            logger.info(f"  {key}: {value}")
        logger.info(f"{'='*50}\n")
        
        return info
    
    def suggest_alternatives(self):
        """대안 제안"""
        logger.info("Suggesting alternatives for failed downloads...")
        
        alternatives = [
            "1. Manual download: Use a local machine to download and upload to AWS",
            "2. Use different AWS region: Some regions may have different access",
            "3. Use AWS EC2 with different settings: Configure proxy or VPN",
            "4. Download subset: Focus on successfully downloaded files only",
            "5. Use Internet Archive: Some tracks may be available there"
        ]
        
        logger.info("Alternative approaches:")
        for alt in alternatives:
            logger.info(f"  {alt}")
    
    def download_all_aws(self, max_files: Optional[int] = None):
        """AWS 환경용 전체 다운로드 프로세스"""
        logger.info("Starting AWS-optimized SALAMI download process...")
        
        # 1. SALAMI 어노테이션 다운로드
        if not self.download_salami_annotations():
            return
        
        # 2. yt-dlp 업데이트
        if not self.install_yt_dlp():
            logger.warning("Could not update yt-dlp, proceeding with existing version")
        
        # 3. YouTube pairings 다운로드
        pairings_path = self.download_youtube_pairings()
        if not pairings_path:
            return
        
        # 4. YouTube 오디오 다운로드 (AWS 최적화)
        download_list = self.parse_youtube_pairings(pairings_path)
        if download_list:
            self.download_youtube_batch_aws(download_list, max_files)
        
        # 5. 데이터셋 정보 생성
        self.create_alternative_dataset_info()
        
        # 6. 결과에 따른 가이드
        if self.success_count > 10:
            logger.info(f"\n🎉 Successfully downloaded {self.success_count} files!")
            logger.info("You can proceed with data processing:")
            logger.info(f"   python scripts/prepare_salami_data.py \\")
            logger.info(f"       --salami_root {self.salami_dir} \\")
            logger.info(f"       --audio_root {self.audio_dir}")
        elif self.success_count > 0:
            logger.info(f"\n⚠️ Partially successful: {self.success_count} files downloaded")
            logger.info("You can still proceed with limited data:")
            logger.info("   python scripts/prepare_salami_data.py --min_duration 5 --max_duration 300")
        else:
            logger.warning("\n❌ No files successfully downloaded")
            self.suggest_alternatives()


def main():
    parser = argparse.ArgumentParser(
        description="AWS-optimized SALAMI dataset downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
AWS Environment Examples:
  # Download small batch for testing
  python scripts/download_datasets_aws.py --max-files 20
  
  # Download with more retries
  python scripts/download_datasets_aws.py --max-files 50
  
  # Download everything (may take very long)
  python scripts/download_datasets_aws.py
        """
    )
    
    parser.add_argument('--base-dir', type=str, default='./datasets',
                        help='Base directory for datasets (default: ./datasets)')
    parser.add_argument('--max-files', type=int, default=100,
                        help='Maximum number of audio files to download (default: 100)')
    
    args = parser.parse_args()
    
    # AWS 환경 감지
    is_aws = any([
        os.getenv('AWS_EXECUTION_ENV'),
        os.getenv('AWS_LAMBDA_FUNCTION_NAME'),
        'amazonaws.com' in os.getenv('HOSTNAME', ''),
        os.path.exists('/opt/aws')
    ])
    
    if is_aws:
        logger.info("🌩️ AWS environment detected - using optimized settings")
    else:
        logger.info("💻 Local environment detected - using standard settings")
    
    # 다운로더 생성 및 실행
    downloader = AWSOptimizedDownloader(args.base_dir)
    downloader.download_all_aws(max_files=args.max_files)


if __name__ == "__main__":
    main()