import os
import torch
from torch.utils.data import DataLoader
from diffusers import StableAudioDiTModel, AutoencoderOobleck
from transformers import T5EncoderModel, AutoTokenizer
from datasets import load_dataset


def main(model_dir: str, dataset_name: str, output_dir: str, num_epochs: int = 1, batch_size: int = 4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained components
    vae = AutoencoderOobleck.from_pretrained(model_dir, subfolder="vae").to(device)
    text_encoder = T5EncoderModel.from_pretrained(model_dir, subfolder="text_encoder").to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, subfolder="tokenizer")
    transformer = StableAudioDiTModel.from_pretrained(model_dir, subfolder="transformer")

    # Freeze the VAE parameters
    vae.requires_grad_(False)
    transformer.train()

    # Load dataset of dicts with keys 'audio' and 'prompt'
    dataset = load_dataset(dataset_name, split="train")

    def preprocess(example):
        # Tokenize prompt text
        tokens = tokenizer(
            example["prompt"],
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        example["input_ids"] = tokens.input_ids.squeeze(0)
        example["attention_mask"] = tokens.attention_mask.squeeze(0)

        # Convert audio to torch tensor if provided as path
        audio = example["audio"]
        if isinstance(audio, str):
            # If dataset stores path to the waveform, load it
            import soundfile as sf
            waveform, _ = sf.read(audio)
            audio = torch.tensor(waveform.T)
        example["waveform"] = audio
        return example

    dataset = dataset.map(preprocess)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(transformer.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        for batch in dataloader:
            waveform = batch["waveform"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.no_grad():
                latents = vae.encode(waveform).latent_dist.sample() * vae.config.scaling_factor
                text_embeds = text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

            outputs = transformer(latents, encoder_hidden_states=text_embeds)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"Epoch {epoch} Loss: {loss.item():.4f}")

    transformer.save_pretrained(output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Stable Audio transformer with pretrained VAE")
    parser.add_argument("--model_dir", type=str, default=".", help="Directory with pretrained model components")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name or path")
    parser.add_argument("--output_dir", type=str, default="finetuned_transformer", help="Directory to save the fine-tuned transformer")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")

    args = parser.parse_args()
    main(args.model_dir, args.dataset_name, args.output_dir, args.epochs, args.batch_size)