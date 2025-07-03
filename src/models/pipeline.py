
## 5. 학습 파이프라인 (src/models/pipeline.py)


import torch
import torch.nn as nn
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import T5EncoderModel, T5Tokenizer
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

class StructureAwareStableAudioPipeline(nn.Module):
    """구조 인식 Stable Audio 파이프라인"""
    
    def __init__(self,
                 vae_model_name: str = "stabilityai/stable-audio-open-1.0",
                 text_encoder_name: str = "google/t5-v1_1-base",
                 structure_dit: nn.Module = None,
                 device: str = "cuda"):
        super().__init__()
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # VAE 로드 (frozen)
        print("Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(
            vae_model_name,
            subfolder="vae",
            low_cpu_mem_usage=False
        ).to(self.device)
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
            
        # Text encoder 로드 (frozen)
        print("Loading text encoder...")
        self.text_encoder = T5EncoderModel.from_pretrained(text_encoder_name).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(text_encoder_name)
        self.text_encoder.eval()
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        # Structure-conditioned DiT
        self.dit = structure_dit
        
        # Noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="scaled_linear",
            prediction_type="epsilon"
        )
        
        # 설정
        self.sample_rate = 44100
        self.latent_sr = self.sample_rate // 512  # VAE compression
        
    @torch.no_grad()
    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """오디오를 VAE latent로 인코딩"""
        # audio: [B, 1, T]
        latent_dist = self.vae.encode(audio)
        latents = latent_dist.latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        return latents
    
    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Latent를 오디오로 디코딩"""
        latents = latents / self.vae.config.scaling_factor
        audio = self.vae.decode(latents).sample
        return audio
    
    @torch.no_grad()
    def encode_text(self, text_prompts: List[str]) -> torch.Tensor:
        """텍스트를 T5로 인코딩"""
        inputs = self.tokenizer(
            text_prompts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        text_embeddings = self.text_encoder(**inputs).last_hidden_state
        return text_embeddings
    
    def training_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """학습 스텝"""
        audio = batch['audio'].to(self.device)  # [B, 1, T]
        structure_sequences = batch['structure_sequences']
        
        # 1. Audio를 latent로 인코딩
        latents = self.encode_audio(audio)
        
        # 2. 텍스트 프롬프트 생성 (구조 정보 기반)
        text_prompts = self._generate_text_prompts(structure_sequences)
        text_embeddings = self.encode_text(text_prompts)
        
        # 3. Diffusion 학습
        # 노이즈 추가
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps,
            (latents.shape[0],), device=self.device
        ).long()
        
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # 4. 노이즈 예측 (구조 조건화)
        noise_pred = self.dit(
            noisy_latents,
            timesteps,
            text_cond=text_embeddings,
            structure_sequences=structure_sequences
        )
        
        # 5. 손실 계산
        loss = F.mse_loss(noise_pred, noise)
        
        return {"loss": loss}
    
    def _generate_text_prompts(self, structure_sequences: List[List[Tuple[str, float, float]]]) -> List[str]:
        """구조 시퀀스를 텍스트 프롬프트로 변환"""
        prompts = []
        
        for seq in structure_sequences:
            # 구조 요약 생성
            structure_names = [s[0] for s in seq]
            unique_structures = list(dict.fromkeys(structure_names))  # 순서 유지하며 중복 제거
            
            prompt = f"A song with {', '.join(unique_structures)}"
            prompts.append(prompt)
            
        return prompts
    
    @torch.no_grad()
    def generate(self,
                 text_prompt: str,
                 structure_sequence: List[Tuple[str, float, float]],
                 duration: float,
                 num_inference_steps: int = 50,
                 guidance_scale: float = 7.0) -> torch.Tensor:
        """구조 조건화 음악 생성"""
        self.dit.eval()
        
        # 텍스트 인코딩
        text_embeddings = self.encode_text([text_prompt])
        
        # Latent 크기 계산
        latent_length = int(duration * self.latent_sr)
        shape = (1, 8, latent_length)  # 8 channels for VAE latent
        
        # 초기 노이즈
        latents = torch.randn(shape, device=self.device)
        
        # Denoising loop
        self.noise_scheduler.set_timesteps(num_inference_steps)
        
        for t in self.noise_scheduler.timesteps:
            # 노이즈 예측
            noise_pred = self.dit(
                latents,
                t.unsqueeze(0).to(self.device),
                text_cond=text_embeddings,
                structure_sequences=[structure_sequence]
            )
            
            # Classifier-free guidance
            if guidance_scale > 1.0:
                noise_pred_uncond = self.dit(
                    latents,
                    t.unsqueeze(0).to(self.device),
                    text_cond=None,
                    structure_sequences=None
                )
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
            
            # 디노이징 스텝
            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample
        
        # 오디오로 디코딩
        audio = self.decode_latents(latents)
        
        return audio


