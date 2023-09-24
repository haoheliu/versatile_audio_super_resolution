import os
import torch
import logging
import argparse
from audiosr import super_resolution, build_model, save_wave

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.set_float32_matmul_precision("high")


def main(args):
    audiosr = build_model(model_name=args.model_name, device="auto")

    waveform = super_resolution(
        audiosr,
        args.input_path,
        seed=42,
        guidance_scale=3.5,
        ddim_steps=50,
        latent_t_per_second=12.8
    )
    
    save_wave(waveform, args.save_path, name="output", samplerate=48000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform super-resolution on audio files using audiosr package.')

    parser.add_argument('-i', '--input_path', required=True, help='Path to the input waveform file.')
    parser.add_argument('-s', '--save_path', required=True, help='Path to save the output waveform file.')
    parser.add_argument('--model_name', choices=['basic', 'speech'], default='speech', help='Name of the model to be used.')
    parser.add_argument('-d', '--device', default="auto", help='The device for computation. If not specified, the script will automatically choose the device based on your environment.')
    parser.add_argument('--ddim_steps', type=int, default=50, help='The sampling step for DDIM.')
    parser.add_argument('-gs', '--guidance_scale', type=float, default=3.5, help='Guidance scale (Large => better quality and relavancy to text; Small => better diversity).')
    parser.add_argument('--seed', type=int, default=42, help='Change this value (any integer number) will lead to a different generation result.')
    parser.add_argument('-il', '--input_file_list', help='A file that contains all audio files that need to perform audio super resolution.')

    args = parser.parse_args()
    main(args)
