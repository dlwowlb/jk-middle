# scripts/download_datasets_sagemaker.py
"""
AWS SageMaker 환경에 최적화된 SALAMI 데이터셋 다운로더
로컬 환경에서는 작동하지만 SageMaker에서 실패하는 문제 해결
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
import tempfile
import shutil

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SageMakerSALAMIDownloader:
    """AWS SageMaker 환경 최적화된 SALAMI 다운로더"""
    
    def __init__(self, base_dir: str = "./datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        self.salami_dir = self.base_dir / "salami-data-public"
        self.audio_dir = self.base_dir / "audio"
        self.audio_dir.mkdir(exist_ok=True)
        
        # SageMaker 환경 감지
        self.is_sagemaker = self._detect_sagemaker_environment()
        logger.info(f"Running in SageMaker: {self.is_sagemaker}")
        
        # SageMaker 전용 설정
        if self.is_sagemaker:
            self._setup_sagemaker_environment()
    
    def _detect_sagemaker_environment(self) -> bool:
        """SageMaker 환경 감지"""
        sagemaker_indicators = [
            'SM_FRAMEWORK_MODULE',
            'SM_TRAINING_ENV',
            'SM_MODEL_DIR',
            'SAGEMAKER_CONTAINER_LOG_LEVEL'
        ]
        
        # 환경변수로 감지
        for indicator in sagemaker_indicators:
            if indicator in os.environ:
                return True
                
        # 파일시스템으로 감지
        sagemaker_paths = [
            '/opt/ml',
            '/tmp/.sagemaker-internal'
        ]
        
        for path in sagemaker_paths:
            if os.path.exists(path):
                return True
                
        return False
    
    def _setup_sagemaker_environment(self):
        """SageMaker 환경 설정"""
        logger.info("Setting up SageMaker environment...")
        
        # 1. yt-dlp용 임시 디렉토리 설정
        self.temp_dir = Path("/tmp/ytdl_cache")
        self.temp_dir.mkdir(exist_ok=True)
        
        # 2. 환경 변수 설정
        os.environ['TMPDIR'] = str(self.temp_dir)
        os.environ['XDG_CACHE_HOME'] = str(self.temp_dir)
        
        # 3. 더 관대한 요청 헤더 설정
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        # 4. 네트워크 타임아웃 설정
        self.timeout = 60
        self.max_retries = 5
    
    def check_network_connectivity(self) -> bool:
        """네트워크 연결 확인"""
        test_urls = [
            'https://www.google.com',
            'https://www.youtube.com',
            'https://raw.githubusercontent.com'
        ]
        
        for url in test_urls:
            try:
                response = requests.get(url, timeout=10, headers=getattr(self, 'headers', {}))
                if response.status_code == 200:
                    logger.info(f"✓ Network connectivity confirmed: {url}")
                    return True
            except Exception as e:
                logger.warning(f"Network test failed for {url}: {e}")
                
        logger.error("❌ Network connectivity issues detected")
        return False
    
    def install_yt_dlp_sagemaker(self) -> bool:
        """SageMaker 환경에서 yt-dlp 설치"""
        logger.info("Installing yt-dlp for SageMaker...")
        
        try:
            # 1. 기존 설치 확인
            result = subprocess.run(["yt-dlp", "--version"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info(f"✓ yt-dlp already installed: {result.stdout.strip()}")
                return True
        except:
            pass
        
        try:
            # 2. pip 업그레이드
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                         check=True, timeout=60)
            
            # 3. yt-dlp 설치 (여러 시도)
            install_commands = [
                [sys.executable, "-m", "pip", "install", "yt-dlp"],
                [sys.executable, "-m", "pip", "install", "yt-dlp", "--no-cache-dir"],
                [sys.executable, "-m", "pip", "install", "yt-dlp", "--force-reinstall"],
            ]
            
            for cmd in install_commands:
                try:
                    logger.info(f"Trying: {' '.join(cmd)}")
                    subprocess.run(cmd, check=True, timeout=120)
                    
                    # 설치 확인
                    result = subprocess.run(["yt-dlp", "--version"], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        logger.info(f"✓ yt-dlp installed successfully: {result.stdout.strip()}")
                        return True
                        
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Install attempt failed: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"❌ Failed to install yt-dlp: {e}")
            
        return False
    
    def download_file_with_retries(self, url: str, dest_path: Path, desc: str = "Downloading") -> bool:
        """재시도 로직이 포함된 파일 다운로드"""
        for attempt in range(self.max_retries if hasattr(self, 'max_retries') else 3):
            try:
                headers = getattr(self, 'headers', {})
                timeout = getattr(self, 'timeout', 30)
                
                logger.info(f"Download attempt {attempt + 1}: {url}")
                
                response = requests.get(url, stream=True, headers=headers, timeout=timeout)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                with open(dest_path, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                
                logger.info(f"✓ Download successful: {dest_path}")
                return True
                
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if dest_path.exists():
                    dest_path.unlink()
                    
                if attempt < (self.max_retries if hasattr(self, 'max_retries') else 3) - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    
        logger.error(f"❌ Failed to download after all attempts: {url}")
        return False
    
    def download_youtube_audio_sagemaker(self, youtube_id: str, output_path: Path, retries: int = 3) -> bool:
        """SageMaker 환경에 최적화된 YouTube 오디오 다운로드"""
        if output_path.exists():
            return True
            
        for attempt in range(retries):
            try:
                # SageMaker 최적화된 yt-dlp 명령
                cmd = [
                    "yt-dlp",
                    "-x",  # 오디오만 추출
                    "--audio-format", "mp3",
                    "--audio-quality", "0",  # 최고 품질
                    "-o", str(output_path),
                    "--no-playlist",
                    "--quiet",
                    "--no-warnings",
                    "--no-check-certificate",  # SageMaker에서 SSL 문제 우회
                    "--socket-timeout", "30",
                    "--retries", "3",
                    "--fragment-retries", "3",
                    "--extractor-retries", "3",
                    f"https://www.youtube.com/watch?v={youtube_id}"
                ]
                
                # SageMaker 환경에서 추가 옵션
                if self.is_sagemaker:
                    cmd.extend([
                        "--no-cache-dir",
                        "--prefer-ffmpeg",
                        "--embed-subs", "false",
                        "--write-thumbnail", "false",
                        "--write-info-json", "false"
                    ])
                    
                    # 임시 디렉토리 설정
                    if hasattr(self, 'temp_dir'):
                        cmd.extend(["--cache-dir", str(self.temp_dir)])
                
                logger.debug(f"Running: {' '.join(cmd)}")
                
                # 더 긴 타임아웃 설정
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=300,  # 5분
                    cwd=str(self.temp_dir) if hasattr(self, 'temp_dir') else None
                )
                
                if result.returncode == 0 and output_path.exists():
                    logger.debug(f"✓ Downloaded: {youtube_id}")
                    return True
                    
                # 에러 분석
                error_msg = result.stderr.lower()
                if any(phrase in error_msg for phrase in [
                    "video unavailable", "private video", "deleted", "removed",
                    "not available", "copyright", "blocked"
                ]):
                    logger.warning(f"Video {youtube_id} is unavailable: {result.stderr}")
                    return False
                    
                logger.warning(f"yt-dlp failed for {youtube_id}: {result.stderr}")
                
            except subprocess.TimeoutExpired:
                logger.warning(f"Timeout downloading {youtube_id}")
            except Exception as e:
                logger.warning(f"Error downloading {youtube_id}: {e}")
                
            if attempt < retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Retrying {youtube_id} in {wait_time}s...")
                time.sleep(wait_time)
                
        return False
    
    def download_salami_annotations(self):
        """SALAMI 어노테이션 다운로드"""
        logger.info("Downloading SALAMI annotations...")
        
        salami_url = "https://github.com/DDMAL/salami-data-public/archive/refs/heads/master.zip"
        zip_path = self.base_dir / "salami.zip"
        
        if not self.salami_dir.exists():
            if self.download_file_with_retries(salami_url, zip_path, "SALAMI annotations"):
                logger.info("Extracting SALAMI data...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.base_dir)
                
                # 디렉토리 이름 변경
                extracted_dir = self.base_dir / "salami-data-public-master"
                if extracted_dir.exists():
                    extracted_dir.rename(self.salami_dir)
                
                zip_path.unlink()
                logger.info(f"✓ SALAMI data downloaded to {self.salami_dir}")
                return True
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
            if self.download_file_with_retries(pairings_url, pairings_path, "YouTube pairings"):
                logger.info("✓ YouTube pairings downloaded")
                return pairings_path
            else:
                logger.error("✗ Failed to download YouTube pairings")
                return None
        else:
            logger.info("✓ YouTube pairings already exist")
            return pairings_path
    
    def test_yt_dlp_functionality(self) -> bool:
        """yt-dlp 기능 테스트"""
        logger.info("Testing yt-dlp functionality...")
        
        # 테스트용 짧은 동영상 (Creative Commons)
        test_ids = [
            "jNQXAC9IVRw",  # "Me at the zoo" - 첫 YouTube 동영상
            "BaW_jenozKc",  # "YouTube" 공식 채널 동영상
        ]
        
        test_dir = self.base_dir / "test"
        test_dir.mkdir(exist_ok=True)
        
        for test_id in test_ids:
            test_output = test_dir / f"test_{test_id}.mp3"
            
            try:
                if self.download_youtube_audio_sagemaker(test_id, test_output, retries=1):
                    logger.info(f"✓ yt-dlp test successful with {test_id}")
                    if test_output.exists():
                        test_output.unlink()
                    return True
            except Exception as e:
                logger.warning(f"Test failed for {test_id}: {e}")
                
        logger.error("❌ yt-dlp functionality test failed")
        return False
    
    def download_youtube_batch_sagemaker(self, download_list: List[Dict], 
                                       max_workers: int = 2, 
                                       max_files: Optional[int] = None):
        """SageMaker 최적화된 배치 다운로드"""
        if max_files:
            download_list = download_list[:max_files]
            
        logger.info(f"Starting SageMaker-optimized download of {len(download_list)} files...")
        
        # SageMaker에서는 더 적은 worker 사용
        actual_workers = min(max_workers, 2) if self.is_sagemaker else max_workers
        
        successful = 0
        failed = 0
        unavailable = []
        
        # Progress bar
        with tqdm(total=len(download_list), desc="Downloading audio") as pbar:
            if actual_workers == 1:
                # 순차 다운로드 (SageMaker에서 더 안정적)
                for item in download_list:
                    try:
                        if self.download_youtube_audio_sagemaker(
                            item['youtube_id'],
                            self.audio_dir / item['filename']
                        ):
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
                    
                    # SageMaker에서 rate limiting
                    if self.is_sagemaker:
                        time.sleep(1)
            else:
                # 병렬 다운로드
                with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                    future_to_item = {
                        executor.submit(
                            self.download_youtube_audio_sagemaker,
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
        logger.info(f"SageMaker Download Summary:")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Success rate: {(successful / len(download_list) * 100):.1f}%")
        
        if unavailable:
            logger.info(f"  Unavailable videos: {len(unavailable)}")
            with open(self.base_dir / "unavailable_videos.txt", 'w') as f:
                f.write('\n'.join(unavailable))
        
        logger.info(f"{'='*50}\n")
        
        return successful, failed
    
    def download_all_sagemaker(self, max_files: Optional[int] = None, max_workers: int = 2):
        """SageMaker 최적화된 전체 다운로드 프로세스"""
        logger.info("Starting SageMaker-optimized SALAMI download...")
        
        # 1. 네트워크 연결 확인
        if not self.check_network_connectivity():
            logger.error("❌ Network connectivity issues. Aborting.")
            return
        
        # 2. SALAMI 어노테이션 다운로드
        if not self.download_salami_annotations():
            logger.error("❌ Failed to download SALAMI annotations")
            return
        
        # 3. yt-dlp 설치
        if not self.install_yt_dlp_sagemaker():
            logger.error("❌ Failed to install yt-dlp")
            return
        
        # 4. yt-dlp 기능 테스트
        if not self.test_yt_dlp_functionality():
            logger.warning("⚠️ yt-dlp test failed, but continuing...")
        
        # 5. YouTube pairings 다운로드
        pairings_path = self.download_youtube_pairings()
        if not pairings_path:
            logger.error("❌ Failed to download YouTube pairings")
            return
        
        # 6. YouTube 오디오 다운로드
        from scripts.download_datasets import SALAMIYouTubeDownloader
        downloader = SALAMIYouTubeDownloader(str(self.base_dir))
        download_list = downloader.parse_youtube_pairings(pairings_path)
        
        if download_list:
            self.download_youtube_batch_sagemaker(download_list, max_workers, max_files)
        
        # 7. 최종 정리
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                logger.info("✓ Cleaned up temporary files")
            except:
                pass
        
        logger.info("SageMaker download process completed!")


def main():
    parser = argparse.ArgumentParser(
        description="Download SALAMI dataset optimized for AWS SageMaker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
SageMaker-optimized examples:
  # Test with small dataset
  python download_datasets_sagemaker.py --max-files 10 --workers 1
  
  # Full download with conservative settings
  python download_datasets_sagemaker.py --workers 1
        """
    )
    
    parser.add_argument('--base-dir', type=str, default='./datasets',
                        help='Base directory for datasets')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of audio files to download')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers (recommended: 1 for SageMaker)')
    parser.add_argument('--test-only', action='store_true',
                        help='Only test the setup without downloading')
    
    args = parser.parse_args()
    
    # 다운로더 생성
    downloader = SageMakerSALAMIDownloader(args.base_dir)
    
    if args.test_only:
        logger.info("Running test mode...")
        
        # 기본 테스트들
        tests = [
            ("Network connectivity", downloader.check_network_connectivity),
            ("SALAMI annotations", downloader.download_salami_annotations),
            ("yt-dlp installation", downloader.install_yt_dlp_sagemaker),
            ("yt-dlp functionality", downloader.test_yt_dlp_functionality)
        ]
        
        results = {}
        for test_name, test_func in tests:
            logger.info(f"\n{'='*50}")
            logger.info(f"Testing: {test_name}")
            logger.info(f"{'='*50}")
            
            try:
                results[test_name] = test_func()
                status = "✓ PASS" if results[test_name] else "✗ FAIL"
                logger.info(f"{test_name}: {status}")
            except Exception as e:
                results[test_name] = False
                logger.error(f"{test_name}: ✗ FAIL - {e}")
        
        # 결과 요약
        logger.info(f"\n{'='*50}")
        logger.info("TEST SUMMARY")
        logger.info(f"{'='*50}")
        for test_name, result in results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            logger.info(f"{test_name}: {status}")
        
        all_passed = all(results.values())
        logger.info(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
        
        if all_passed:
            logger.info("\n🚀 System is ready for download!")
            logger.info("Run without --test-only to start downloading")
        else:
            logger.info("\n🔧 Please fix the failing tests before downloading")
            
    else:
        # 실제 다운로드
        downloader.download_all_sagemaker(
            max_files=args.max_files,
            max_workers=args.workers
        )


if __name__ == "__main__":
    main()