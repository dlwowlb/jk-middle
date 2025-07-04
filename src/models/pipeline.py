# src/models/pipeline.py - Stable Audio 공식 구현 방식 사용

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableAudioPipeline, AutoencoderOobleck
from transformers import T5EncoderModel, T5Tokenizer
from typing import Dict, List, Tuple, Optional
import numpy as np

class StructureAwareStableAudioPipeline(nn.Module):
    """
    구조 인식 Stable Audio 파이프라인
    HuggingFace의 StableAudioPipeline 구조를 기반으로 구현
    """
    
    def __init__(self,
                 model_id: str = "stabilityai/stable-audio-open-1.0",
                 structure_dit: Optional[nn.Module] = None,
                 device: str = "cuda"):
        super().__init__()
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Stable Audio 파이프라인 로드
        print("Loading Stable Audio pipeline...")
        self.base_pipeline = StableAudioPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16
        ).to(self.device)
        
        # 컴포넌트 추출
        self.vae = self.base_pipeline.vae  # AutoencoderOobleck - 오디오용 VAE
        self.text_encoder = self.base_pipeline.text_encoder  # T5
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
            # 기본 transformer 사용 (구조 조건화 없음)
            self.transformer = self.base_pipeline.transformer
            
        # 설정값들
        self.sample_rate = 44100
        
        # VAE scale factor 계산
        # Stable Audio는 다운샘플링 레이트를 직접 사용
        if hasattr(self.vae.config, 'downsampling_rate'):
            self.vae_scale_factor = self.vae.config.downsampling_rate
        elif hasattr(self.vae.config, 'downsample_rates'):
            # 각 다운샘플링 레이트의 곱
            self.vae_scale_factor = np.prod(self.vae.config.downsample_rates)
        else:
            # 기본값 (Stable Audio Open 1.0 기준)
            self.vae_scale_factor = 512
        
    def encode_audio_to_latent(self, audio: torch.Tensor) -> torch.Tensor:
        """
        오디오를 VAE latent로 인코딩
        
        Args:
            audio: [batch, channels, samples] 형태의 오디오
        Returns:
            latents: [batch, latent_channels, latent_length]
        """
        # VAE는 [-1, 1] 범위의 입력을 기대
        audio = audio.clamp(-1, 1)
        
        # float16으로 변환 (메모리 효율)
        audio = audio.to(dtype=torch.float16, device=self.device)
        
        with torch.no_grad():
            # Oobleck VAE encode
            encoded = self.vae.encode(audio)
            
            # encoded가 tuple이나 dict일 수 있음
            if hasattr(encoded, 'latent_dist'):
                latents = encoded.latent_dist.sample()
            elif hasattr(encoded, 'latents'):
                latents = encoded.latents
            elif isinstance(encoded, tuple):
                latents = encoded[0]
            else:
                latents = encoded
                
            # Scaling factor 적용 (있는 경우)
            if hasattr(self.vae.config, 'scaling_factor'):
                latents = latents * self.vae.config.scaling_factor
            
        return latents
    
    def decode_latents_to_audio(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Latent를 오디오로 디코딩
        
        Args:
            latents: [batch, latent_channels, latent_length]
        Returns:
            audio: [batch, channels, samples]
        """
        # Scaling factor 적용 (있는 경우)
        if hasattr(self.vae.config, 'scaling_factor'):
            latents = latents / self.vae.config.scaling_factor
        
        with torch.no_grad():
            decoded = self.vae.decode(latents)
            
            # decoded가 여러 형태일 수 있음
            if hasattr(decoded, 'sample'):
                audio = decoded.sample
            elif isinstance(decoded, tuple):
                audio = decoded[0]
            else:
                audio = decoded
                
        return audio
    
    def encode_text_prompts(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        텍스트 프롬프트를 인코딩
        
        Returns:
            prompt_embeds: 텍스트 임베딩
            attention_mask: 어텐션 마스크
        """
        text_inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
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
        
        # 스케줄러 초기 스케일링
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    def training_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """학습 스텝"""
        audio = batch['audio'].to(self.device)  # [B, C, T]
        structure_sequences = batch['structure_sequences']
        
        # 오디오 채널 확인 및 조정
        if audio.shape[1] == 1:
            # 모노를 스테레오로 변환
            audio = audio.repeat(1, 2, 1)
        elif audio.shape[1] > 2:
            # 멀티채널을 스테레오로
            audio = audio[:, :2, :]
        
        # 1. 오디오를 latent로 인코딩
        latents = self.encode_audio_to_latent(audio)
        
        # 2. 텍스트 프롬프트 생성 및 인코딩
        text_prompts = self._generate_text_prompts(structure_sequences)
        prompt_embeds, prompt_attention_mask = self.encode_text_prompts(text_prompts)
        
        # 3. 타이밍 임베딩 준비 (Stable Audio는 시작/끝 시간 조건화 사용)
        seconds_start = torch.zeros(len(audio), device=self.device)
        seconds_total = torch.tensor(
            [seq[-1][2] if seq else 30.0 for seq in structure_sequences],
            device=self.device
        )
        
        # 4. Projection 모델로 조건 임베딩 생성
        # (실제 Stable Audio는 projection_model을 사용하지만, 간단히 구현)
        global_embeds = self._prepare_global_embeds(
            prompt_embeds,
            prompt_attention_mask,
            seconds_start,
            seconds_total
        )
        
        # 5. Diffusion 학습
        # 노이즈 추가
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=self.device
        ).long()
        
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        # 6. Transformer로 노이즈 예측 (구조 조건화 포함)
        if hasattr(self.transformer, 'forward_with_structure'):
            # 우리의 structure-conditioned transformer
            noise_pred = self.transformer.forward_with_structure(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                global_embeds=global_embeds,
                structure_sequences=structure_sequences,
                attention_mask=prompt_attention_mask
            )
        else:
            # 기본 transformer (구조 조건화 없음)
            noise_pred = self.transformer(
                hidden_states=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                global_embeds=global_embeds,
                attention_mask=prompt_attention_mask
            ).sample
        
        # 7. 손실 계산
        loss = F.mse_loss(noise_pred, noise, reduction="mean")
        
        return {"loss": loss}
    
    def _generate_text_prompts(self, structure_sequences: List[List[Tuple[str, float, float]]]) -> List[str]:
        """구조 시퀀스를 텍스트 프롬프트로 변환"""
        prompts = []
        
        for seq in structure_sequences:
            if not seq:
                prompts.append("A musical piece")
                continue
                
            # 구조 정보를 자연어로 변환
            structures = [s[0] for s in seq]
            unique_structures = []
            for s in structures:
                if s not in unique_structures:
                    unique_structures.append(s)
            
            # 프롬프트 생성
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
        """
        글로벌 임베딩 준비 (텍스트 + 타이밍 정보)
        실제 구현은 StableAudioProjectionModel을 사용하지만, 여기서는 간단히 구현
        """
        # 텍스트 임베딩을 평균 풀링
        text_mask = attention_mask.unsqueeze(-1).expand(prompt_embeds.size())
        text_embeds_mean = (prompt_embeds * text_mask).sum(dim=1) / text_mask.sum(dim=1)
        
        # 타이밍 임베딩 (간단한 선형 프로젝션)
        if not hasattr(self, 'time_proj'):
            self.time_proj = nn.Linear(2, text_embeds_mean.shape[-1]).to(self.device)
            
        time_embeds = self.time_proj(
            torch.stack([seconds_start, seconds_total], dim=-1).float()
        )
        
        # 결합
        global_embeds = text_embeds_mean + time_embeds
        
        return global_embeds
    
    @torch.no_grad()
    def generate(self,
                 prompt: str,
                 structure_sequence: List[Tuple[str, float, float]],
                 duration: Optional[float] = None,
                 num_inference_steps: int = 100,
                 guidance_scale: float = 7.0,
                 negative_prompt: Optional[str] = None,
                 generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        구조 조건화된 오디오 생성
        
        Returns:
            audio: [1, channels, samples] 형태의 생성된 오디오
        """
        self.transformer.eval()
        
        # Duration 설정
        if duration is None:
            duration = structure_sequence[-1][2] if structure_sequence else 30.0
            
        # 배치 크기 1로 설정
        batch_size = 1
        
        # 프롬프트 인코딩
        prompt_embeds, prompt_attention_mask = self.encode_text_prompts([prompt])
        
        # Negative prompt 처리
        if guidance_scale > 1.0:
            if negative_prompt is None:
                negative_prompt = ""
            negative_embeds, negative_attention_mask = self.encode_text_prompts([negative_prompt])
            
            # Concat for classifier-free guidance
            prompt_embeds = torch.cat([negative_embeds, prompt_embeds])
            prompt_attention_mask = torch.cat([negative_attention_mask, prompt_attention_mask])
        
        # 타이밍 정보
        seconds_start = torch.zeros(batch_size, device=self.device)
        seconds_total = torch.tensor([duration], device=self.device)
        
        # 글로벌 임베딩
        global_embeds = self._prepare_global_embeds(
            prompt_embeds[-batch_size:],  # positive prompt만
            prompt_attention_mask[-batch_size:],
            seconds_start,
            seconds_total
        )
        
        if guidance_scale > 1.0:
            # Negative를 위한 글로벌 임베딩도 준비
            negative_global_embeds = self._prepare_global_embeds(
                prompt_embeds[:batch_size],  # negative prompt
                prompt_attention_mask[:batch_size],
                seconds_start,
                seconds_total
            )
            global_embeds = torch.cat([negative_global_embeds, global_embeds])
        
        # Latent 크기 계산
        audio_length_in_s = duration
        audio_length_samples = int(audio_length_in_s * self.sample_rate)
        
        # VAE 압축을 고려한 latent 길이
        latent_length = audio_length_samples // self.vae_scale_factor
        
        # Transformer의 latent channels 가져오기
        if hasattr(self.transformer, 'config') and hasattr(self.transformer.config, 'in_channels'):
            latent_channels = self.transformer.config.in_channels
        elif hasattr(self.base_pipeline.transformer, 'config'):
            latent_channels = self.base_pipeline.transformer.config.in_channels
        else:
            # 기본값 (Stable Audio Open 1.0)
            latent_channels = 64
        
        # 초기 노이즈 생성
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
            # Classifier-free guidance를 위한 latent 확장
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # 타임스텝
            timestep = t.unsqueeze(0).expand(latent_model_input.shape[0])
            
            # 노이즈 예측
            if hasattr(self.transformer, 'forward_with_structure'):
                noise_pred = self.transformer.forward_with_structure(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    global_embeds=global_embeds,
                    structure_sequences=[structure_sequence] * latent_model_input.shape[0],
                    attention_mask=prompt_attention_mask
                )
            else:
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    global_embeds=global_embeds,
                    attention_mask=prompt_attention_mask
                ).sample
            
            # Classifier-free guidance 적용
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # 스케줄러 스텝
            latents = self.scheduler.step(noise_pred, t, latents, generator=generator).prev_sample
        
        # Latent를 오디오로 디코딩
        audio = self.decode_latents_to_audio(latents)
        
        return audio
    
    def save_pretrained(self, save_directory: str):
        """모델 저장"""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Transformer만 저장 (VAE와 text encoder는 frozen)
        transformer_path = os.path.join(save_directory, "transformer.pt")
        torch.save(self.transformer.state_dict(), transformer_path)
        
        # 설정 저장
        config_path = os.path.join(save_directory, "config.json")
        config = {
            "model_id": "stabilityai/stable-audio-open-1.0",
            "sample_rate": self.sample_rate,
            "vae_scale_factor": self.vae_scale_factor
        }
        
        import json
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
    @classmethod
    def from_pretrained(cls, load_directory: str, structure_dit: nn.Module):
        """저장된 모델 로드"""
        import os
        import json
        
        # 설정 로드
        config_path = os.path.join(load_directory, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # 파이프라인 생성
        pipeline = cls(
            model_id=config["model_id"],
            structure_dit=structure_dit
        )
        
        # Transformer 가중치 로드
        transformer_path = os.path.join(load_directory, "transformer.pt")
        pipeline.transformer.load_state_dict(torch.load(transformer_path))
        
        return pipeline