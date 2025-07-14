# scripts/generate.py
import sys
sys.path.append('..')
sys.path.append('.')

import torch
import torchaudio
from src.models.structure_encoder import StructureEncoder
from src.models.structure_dit import StructureConditionedDiT
from src.models.pipeline import StructureAwareStableAudioPipeline
import argparse
import yaml

def parse_structure_string(structure_str: str):
    """구조 문자열을 파싱"""
    structure_sequence = []
    for part in structure_str.split(','):
        name, start, end = part.split(':')
        structure_sequence.append((name.strip(), float(start), float(end)))
    return structure_sequence

def main():
    parser = argparse.ArgumentParser(description='Generate music with structure conditioning')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--text', type=str, default="A pop song with guitar and drums",
                        help='Text prompt for generation')
    parser.add_argument('--structure', type=str, 
                        default="intro:0:4,verse:4:8,outro:8:12",
                        help='Structure sequence (format: name:start:end,name:start:end,...)')
    parser.add_argument('--output', type=str, default="generated.wav",
                        help='Output audio file path')
    parser.add_argument('--steps', type=int, default=20,
                        help='Number of denoising steps')
    parser.add_argument('--guidance', type=float, default=1.0,
                        help='Classifier-free guidance scale')
    parser.add_argument('--negative_prompt', type=str, default="low quality, noise",
                        help='Negative prompt')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for generation')
    args = parser.parse_args()
    
    # 구조 파싱
    structure_sequence = parse_structure_string(args.structure)
    duration = structure_sequence[-1][2] if structure_sequence else 30.0
    
    print(f"Generating {duration}s audio with structure: {[s[0] for s in structure_sequence]}")
    
    # 체크포인트 로드
    print("Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # 설정 추출
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})
    
    # 모델 재생성
    structure_encoder = StructureEncoder(
        **model_config.get('structure_encoder', {
            'embedding_dim': 768,
            'hidden_dim': 512,
            'num_layers': 4,
            'num_heads': 8,
            'dropout': 0.1,
            'max_structures': 50
        })
    )
    
    # DiT 설정
    base_dit_config = {
        'hidden_size': 768,
        'num_heads': 16,
        'depth': 24,
        'mlp_ratio': 4.0,
        'in_channels': 64,
    }
    base_dit_config.update(model_config.get('dit', {}))
    
    structure_dit = StructureConditionedDiT(
        base_dit_config=base_dit_config,
        structure_encoder=structure_encoder,
        conditioning_method=model_config.get('conditioning_method', 'cross_attention')
    )
    
    # 파이프라인 생성
    model = StructureAwareStableAudioPipeline(
        model_id=model_config.get('model_id', "stabilityai/stable-audio-open-1.0"),
        structure_dit=structure_dit
    )
    
    # 체크포인트에서 가중치 로드
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    # 시드 설정
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=model.device).manual_seed(args.seed)
    
    # 생성
    print(f"Generating with prompt: '{args.text}'")
    with torch.no_grad():
        audio = model.generate(
            prompt=args.text,
            structure_sequence=structure_sequence,
            duration=duration,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            negative_prompt=args.negative_prompt,
            generator=generator
        )
    
    # 저장
    audio = audio.squeeze(0).cpu()
    
    # 정규화 ([-1, 1] 범위로)
    audio = audio / audio.abs().max() * 0.95
    
    torchaudio.save(
        args.output,
        audio,
        model.sample_rate,
        bits_per_sample=16
    )
    
    print(f"Generated audio saved to: {args.output}")
    print(f"Duration: {audio.shape[-1] / model.sample_rate:.1f}s")
    print(f"Sample rate: {model.sample_rate}Hz")


if __name__ == "__main__":
    main()