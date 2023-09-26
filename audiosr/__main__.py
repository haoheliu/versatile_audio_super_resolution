import os
import torch
import logging
import argparse
from audiosr import super_resolution, build_model, save_wave, read_list

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.set_float32_matmul_precision("high")

def main(args):
    audiosr = build_model(model_name=args.model_name, device=args.device)

    if args.input_file_list:
        input_files = read_list(args.input_file_list)
    else:
        input_files = [args.input_path]

    for input_file in input_files:
        name = os.path.splitext(os.path.basename(input_file))[0]
        waveform = super_resolution(
            audiosr,
            input_file,
            seed=args.seed,
            guidance_scale=args.guidance_scale,
            ddim_steps=args.ddim_steps,
            latent_t_per_second=args.latent_t_per_second
        )
        save_wave(waveform, args.save_path, name=name, samplerate=args.samplerate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform super-resolution on audio files using audiosr package.')

    parser.add_argument('-i', '--input_path', required=True, help='Path to the input waveform file.')
    parser.add_argument('-s', '--save_path', required=True, help='Path to save the output waveform file.')
    parser.add_argument('--model_name', choices=['basic', 'speech'], default='speech', help='Name of the model to be used.')
    parser.add_argument('-d', '--device', default="auto", help='The device for computation. If not specified, the script will automatically choose the device based on your environment.')
    parser.add_argument('--ddim_steps', type=int, default=50, help='The sampling step for DDIM.')
    parser.add_argument('-gs', '--guidance_scale', type=float, default=3.5, help='Guidance scale (Large => better quality and relevancy to text; Small => better diversity).')
    parser.add_argument('--seed', type=int, default=42, help='Change this value (any integer number) will lead to a different generation result.')
    parser.add_argument('--input_file_list', '-il', help='A file that contains a list of audio files to perform audio super-resolution on.')
    parser.add_argument('--latent_t_per_second', type=float, default=12.8, help='Latent sampling rate per second.')
    parser.add_argument('--samplerate', type=int, default=48000, help='Samplerate for the output waveform.')

    args = parser.parse_args()
    main(args)
