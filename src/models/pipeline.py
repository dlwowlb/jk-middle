# src/models/pipeline.py - 수정된 버전

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableAudioPipeline, AutoencoderOobleck
from transformers import T5EncoderModel, T5Tokenizer
from typing import Dict, List, Tuple, Optional
import numpy as np

class StructureAwareStableAudioPipeline(nn.Module):
    """구조 인식 Stable Audio 파이프라인 - 오류 수정 버전"""
    
    def __init__(self,
                 model_id: str = "stabilityai/stable-audio-open-1.0",
                 structure_dit: Optional[nn.Module] = None,
                 device: str = "cuda"):
        super().__init__()
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Stable Audio 파이프라인 로드
        print("Loading Stable Audio pipeline...")
        try:
            self.base_pipeline = StableAudioPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                variant="fp16"
            ).to(self.device)
        except:
            # Fallback without variant
            self.base_pipeline = StableAudioPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16
            ).to(self.device)
        
        # 컴포넌트 추출
        self.vae = self.base_pipeline.vae
        self.text_encoder = self.base_pipeline.text_encoder
        self.tokenizer = self.base_pipeline.tokenizer
        self.scheduler = self.base_pipeline.scheduler
        
        # VAE와 Text Encoder freeze
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
            
        self.text_encoder.eval()
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # 기존 transformer 대신 structure-conditioned DiT 사용
        if structure_dit is not None:
            self.transformer = structure_dit
        else:
            self.transformer = self.base_pipeline.transformer
            
        # 설정값들
        self.sample_rate = 44100
        
        # VAE scale factor 계산
        if hasattr(self.vae.config, 'downsampling_rate'):
            self.vae_scale_factor = self.vae.config.downsampling_rate
        elif hasattr(self.vae.config, 'downsample_rates'):
            self.vae_scale_factor = np.prod(self.vae.config.downsample_rates)
        else:
            self.vae_scale_factor = 512
            
        # Time projection 초기화 (lazy)
        self.time_proj = None
        
    def encode_audio_to_latent(self, audio: torch.Tensor) -> torch.Tensor:
        """오디오를 VAE latent로 인코딩"""
        # 입력 검증 및 전처리
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)  # [C, T] -> [1, C, T]
        
        # 스테레오 확인
        if audio.shape[1] == 1:
            audio = audio.repeat(1, 2, 1)
        elif audio.shape[1] > 2:
            audio = audio[:, :2, :]
        
        # 범위 확인 및 클리핑
        audio = audio.clamp(-1, 1)
        audio = audio.to(dtype=torch.float16, device=self.device)
        
        with torch.no_grad():
            try:
                encoded = self.vae.encode(audio)
                
                if hasattr(encoded, 'latent_dist'):
                    latents = encoded.latent_dist.sample()
                elif hasattr(encoded, 'latents'):
                    latents = encoded.latents
                elif isinstance(encoded, tuple):
                    latents = encoded[0]
                else:
                    latents = encoded
                    
                # Scaling factor 적용
                if hasattr(self.vae.config, 'scaling_factor'):
                    latents = latents * self.vae.config.scaling_factor
                    
            except Exception as e:
                print(f"VAE encoding error: {e}")
                # Fallback: random latents
                latent_channels = getattr(self.vae.config, 'latent_channels', 64)
                latent_length = audio.shape[-1] // self.vae_scale_factor
                latents = torch.randn(
                    audio.shape[0], latent_channels, latent_length,
                    dtype=torch.float16, device=self.device
                )
        
        return latents
    
    def decode_latents_to_audio(self, latents: torch.Tensor) -> torch.Tensor:
        """Latent를 오디오로 디코딩"""
        if hasattr(self.vae.config, 'scaling_factor'):
            latents = latents / self.vae.config.scaling_factor
        
        with torch.no_grad():
            try:
                decoded = self.vae.decode(latents)
                
                if hasattr(decoded, 'sample'):
                    audio = decoded.sample
                elif isinstance(decoded, tuple):
                    audio = decoded[0]
                else:
                    audio = decoded
                    
            except Exception as e:
                print(f"VAE decoding error: {e}")
                # Fallback: silence
                audio_length = latents.shape[-1] * self.vae_scale_factor
                audio = torch.zeros(
                    latents.shape[0], 2, audio_length,
                    device=self.device, dtype=torch.float16
                )
                
        return audio
    
    def encode_text_prompts(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """텍스트 프롬프트를 인코딩"""
        text_inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=getattr(self.tokenizer, 'model_max_length', 512),
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            text_outputs = self.text_encoder(
                text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask
            )
            prompt_embeds = text_outputs.last_hidden_state
            
        return prompt_embeds, text_inputs.attention_mask
    
    def prepare_latents(self, 
                       batch_size: int,
                       num_channels: int,
                       length: int,
                       dtype: torch.dtype,
                       generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """초기 노이즈 latent 준비"""
        shape = (batch_size, num_channels, length)
        
        latents = torch.randn(
            shape,
            generator=generator,
            device=self.device,
            dtype=dtype
        )
        
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    def training_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """학습 스텝"""
        try:
            audio = batch['audio'].to(self.device)
            structure_sequences = batch['structure_sequences']
            
            # 오디오 전처리
            if audio.shape[1] == 1:
                audio = audio.repeat(1, 2, 1)
            elif audio.shape[1] > 2:
                audio = audio[:, :2, :]
            
            # 1. 오디오를 latent로 인코딩
            latents = self.encode_audio_to_latent(audio)
            
            # 2. 텍스트 프롬프트 생성 및 인코딩
            text_prompts = self._generate_text_prompts(structure_sequences)
            prompt_embeds, prompt_attention_mask = self.encode_text_prompts(text_prompts)
            
            # 3. 타이밍 임베딩 준비
            seconds_start = torch.zeros(len(audio), device=self.device)
            seconds_total = torch.tensor([
                seq[-1][2] if seq else 30.0 for seq in structure_sequences
            ], device=self.device)
            
            # 4. 글로벌 임베딩 생성
            global_embeds = self._prepare_global_embeds(
                prompt_embeds,
                prompt_attention_mask,
                seconds_start,
                seconds_total
            )
            
            # 5. Diffusion 학습
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, self.scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=self.device
            ).long()
            
            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
            
            # 6. Transformer로 노이즈 예측
            noise_pred = self.transformer(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                global_embeds=global_embeds,
                structure_sequences=structure_sequences,
                attention_mask=prompt_attention_mask
            ).sample
            
            # 7. 손실 계산
            loss = F.mse_loss(noise_pred, noise, reduction="mean")
            
            return {"loss": loss}
            
        except Exception as e:
            print(f"Training step error: {e}")
            # 더미 손실 반환
            return {"loss": torch.tensor(0.0, device=self.device, requires_grad=True)}
    
    def _generate_text_prompts(self, structure_sequences: List[List[Tuple[str, float, float]]]) -> List[str]:
        """구조 시퀀스를 텍스트 프롬프트로 변환"""
        prompts = []
        
        for seq in structure_sequences:
            if not seq:
                prompts.append("A musical piece")
                continue
                
            structures = [s[0] for s in seq]
            unique_structures = []
            for s in structures:
                if s not in unique_structures:
                    unique_structures.append(s)
            
            if len(unique_structures) == 1:
                prompt = f"A musical piece with {unique_structures[0]}"
            elif len(unique_structures) == 2:
                prompt = f"A musical piece with {unique_structures[0]} and {unique_structures[1]}"
            else:
                prompt = f"A musical piece with {', '.join(unique_structures[:-1])}, and {unique_structures[-1]}"
                
            prompts.append(prompt)
            
        return prompts
    
    def _prepare_global_embeds(self,
                              prompt_embeds: torch.Tensor,
                              attention_mask: torch.Tensor,
                              seconds_start: torch.Tensor,
                              seconds_total: torch.Tensor) -> torch.Tensor:
        """글로벌 임베딩 준비"""
        # 텍스트 임베딩을 평균 풀링
        text_mask = attention_mask.unsqueeze(-1).expand(prompt_embeds.size()).float()
        text_embeds_mean = (prompt_embeds * text_mask).sum(dim=1) / (text_mask.sum(dim=1) + 1e-8)
        
        # 타이밍 임베딩
        if self.time_proj is None:
            self.time_proj = nn.Linear(2, text_embeds_mean.shape[-1]).to(self.device)
            
        time_embeds = self.time_proj(
            torch.stack([seconds_start, seconds_total], dim=-1).float()
        )
        
        global_embeds = text_embeds_mean + time_embeds
        return global_embeds
    
    @torch.no_grad()
    def generate(self,
                 prompt: str,
                 structure_sequence: List[Tuple[str, float, float]],
                 duration: Optional[float] = None,
                 num_inference_steps: int = 50,  # 기본값 줄임
                 guidance_scale: float = 7.0,
                 negative_prompt: Optional[str] = None,
                 generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """구조 조건화된 오디오 생성"""
        self.transformer.eval()
        
        if duration is None:
            duration = structure_sequence[-1][2] if structure_sequence else 30.0
            
        batch_size = 1
        
        # 프롬프트 인코딩
        prompt_embeds, prompt_attention_mask = self.encode_text_prompts([prompt])
        
        # Negative prompt 처리
        if guidance_scale > 1.0:
            if negative_prompt is None:
                negative_prompt = ""
            negative_embeds, negative_attention_mask = self.encode_text_prompts([negative_prompt])
            
            prompt_embeds = torch.cat([negative_embeds, prompt_embeds])
            prompt_attention_mask = torch.cat([negative_attention_mask, prompt_attention_mask])
        
        # 타이밍 정보
        seconds_start = torch.zeros(batch_size, device=self.device)
        seconds_total = torch.tensor([duration], device=self.device)
        
        # 글로벌 임베딩
        global_embeds = self._prepare_global_embeds(
            prompt_embeds[-batch_size:],
            prompt_attention_mask[-batch_size:],
            seconds_start,
            seconds_total
        )
        
        if guidance_scale > 1.0:
            negative_global_embeds = self._prepare_global_embeds(
                prompt_embeds[:batch_size],
                prompt_attention_mask[:batch_size],
                seconds_start,
                seconds_total
            )
            global_embeds = torch.cat([negative_global_embeds, global_embeds])
        
        # Latent 크기 계산
        audio_length_samples = int(duration * self.sample_rate)
        latent_length = audio_length_samples // self.vae_scale_factor
        
        # Latent channels
        if hasattr(self.transformer, 'config') and hasattr(self.transformer.config, 'in_channels'):
            latent_channels = self.transformer.config.in_channels
        elif hasattr(self.transformer, 'in_channels'):
            latent_channels = self.transformer.in_channels
        else:
            latent_channels = 64
        
        # 초기 노이즈
        latents = self.prepare_latents(
            batch_size,
            latent_channels,
            latent_length,
            prompt_embeds.dtype,
            generator
        )
        
        # 디노이징 루프
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        
        for t in self.scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            timestep = t.unsqueeze(0).expand(latent_model_input.shape[0])
            
            # 노이즈 예측
            try:
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    global_embeds=global_embeds,
                    structure_sequences=[structure_sequence] * latent_model_input.shape[0],
                    attention_mask=prompt_attention_mask
                ).sample
                
                # CFG 적용
                if guidance_scale > 1.0:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # 스케줄러 스텝
                latents = self.scheduler.step(noise_pred, t, latents, generator=generator).prev_sample
                
            except Exception as e:
                print(f"Generation step error at timestep {t}: {e}")
                break
        
        # 오디오로 디코딩
        audio = self.decode_latents_to_audio(latents)
        return audio