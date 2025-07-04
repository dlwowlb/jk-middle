# scripts/evaluate.py
import sys
sys.path.append('..')
sys.path.append('.')

import torch
import torchaudio
import numpy as np
from src.models.structure_encoder import StructureEncoder
from src.models.structure_dit import StructureConditionedDiT
from src.models.pipeline import StructureAwareStableAudioPipeline
from src.data.dataset import StructuredAudioDataset
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import json

def evaluate_structure_accuracy(model, dataloader, device='cuda'):
    """구조 예측 정확도 평가"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            audio = batch['audio'].to(device)
            structure_sequences = batch['structure_sequences']
            
            # 구조 예측 (실제 구현은 모델에 따라 다름)
            # 여기서는 간단한 예시
            
            # TODO: 실제 구조 예측 로직 구현
            
    return correct / total if total > 0 else 0

def evaluate_generation_quality(model, test_prompts, output_dir):
    """생성 품질 평가"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    results = []
    
    for i, prompt_data in enumerate(test_prompts):
        prompt = prompt_data['prompt']
        structure = prompt_data['structure']
        
        print(f"Generating sample {i+1}/{len(test_prompts)}: {prompt}")
        
        with torch.no_grad():
            audio = model.generate(
                prompt=prompt,
                structure_sequence=structure,
                num_inference_steps=100,
                guidance_scale=7.0
            )
        
        # 저장
        output_path = os.path.join(output_dir, f"sample_{i:03d}.wav")
        torchaudio.save(
            output_path,
            audio.squeeze(0).cpu(),
            model.sample_rate
        )
        
        results.append({
            'prompt': prompt,
            'structure': structure,
            'output_path': output_path
        })
    
    # 결과 저장
    with open(os.path.join(output_dir, 'generation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--test_csv', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./evaluation_results')
    parser.add_argument('--num_samples', type=int, default=10)
    args = parser.parse_args()
    
    # 모델 로드 (generate.py와 동일한 방식)
    # ... (모델 로드 코드)
    
    # 테스트 데이터셋
    test_dataset = StructuredAudioDataset(
        csv_path=args.test_csv,
        sample_rate=44100,
        segment_duration=30.0,
        return_full_song=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=test_dataset.get_collate_fn()
    )
    
    # 평가
    print("Evaluating model...")
    
    # 1. 구조 정확도 평가
    # structure_accuracy = evaluate_structure_accuracy(model, test_loader)
    # print(f"Structure accuracy: {structure_accuracy:.2%}")
    
    # 2. 생성 품질 평가
    test_prompts = [
        {
            'prompt': 'Upbeat pop song with guitar',
            'structure': [('intro', 0, 8), ('verse', 8, 24), ('chorus', 24, 40)]
        },
        {
            'prompt': 'Electronic dance music with heavy bass',
            'structure': [('intro', 0, 16), ('drop', 16, 32), ('outro', 32, 40)]
        },
        # 더 많은 테스트 프롬프트...
    ]
    
    results = evaluate_generation_quality(
        model,
        test_prompts[:args.num_samples],
        args.output_dir
    )
    
    print(f"Generated {len(results)} samples in {args.output_dir}")


if __name__ == "__main__":
    main()