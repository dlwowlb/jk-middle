## 8. 생성 스크립트 (scripts/generate.py)


import sys
sys.path.append('..')
sys.path.append('.')

import torch
import torchaudio
from src.models.structure_encoder import StructureEncoder
from src.models.structure_dit import StructureConditionedDiT
from src.models.pipeline import StructureAwareStableAudioPipeline
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--text', type=str, default="A pop song with guitar")
    parser.add_argument('--structure', type=str, default="intro:0:8,verse:8:24,chorus:24:40,outro:40:48")
    parser.add_argument('--output', type=str, default="generated.wav")
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--guidance', type=float, default=7.0)
    args = parser.parse_args()
    
    # 구조 파싱
    structure_sequence = []
    for part in args.structure.split(','):
        name, start, end = part.split(':')
        structure_sequence.append((name, float(start), float(end)))
    
    duration = structure_sequence[-1][2]
    
    # 모델 로드
    print("Loading model...")
    checkpoint = torch.load(args.checkpoint)
    config = checkpoint['config']
    
    # 모델 재생성
    structure_encoder = StructureEncoder(**config['model']['structure_encoder'])
    dit = StructureConditionedDiT(
        config['model']['dit'],
        structure_encoder,
        config['model']['conditioning_method']
    )
    
    model = StructureAwareStableAudioPipeline(
        structure_dit=dit
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 생성
    print(f"Generating {duration}s audio...")
    with torch.no_grad():
        audio = model.generate(
            text_prompt=args.text,
            structure_sequence=structure_sequence,
            duration=duration,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance
        )
    
    # 저장
    audio = audio.squeeze(0).cpu()
    torchaudio.save(args.output, audio, model.sample_rate)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()


