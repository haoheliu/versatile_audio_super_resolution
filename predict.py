import os
import random

import numpy as np
import soundfile as sf
import torch
from cog import BasePredictor, Input, Path

from audiosr import build_model, super_resolution

os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.set_float32_matmul_precision("high")

class Predictor(BasePredictor):
    def setup(self, model_name="basic", device="auto"):
        self.model_name = model_name
        self.device = device
        self.sr = 48000
        self.audiosr = build_model(model_name=self.model_name, device=self.device)

    def predict(self,
        input_file: Path = Input(description="Audio to upsample"),
        ddim_steps: int = Input(description="Number of inference steps", default=50, ge=10, le=500),
        guidance_scale: float = Input(description="Scale for classifier free guidance", default=3.5, ge=1.0, le=20.0),
        seed: int = Input(description="Random seed. Leave blank to randomize the seed", default=None)
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
            print(f"Setting seed to: {seed}")

        waveform = super_resolution(
            self.audiosr,
            input_file,
            seed=seed,
            guidance_scale=guidance_scale,
            ddim_steps=ddim_steps,
            latent_t_per_second=12.8
        )
        out_wav = (waveform[0] * 32767).astype(np.int16).T
        sf.write("out.wav", data=out_wav, samplerate=48000)
        return Path("out.wav")


if __name__ == "__main__":
    p = Predictor()
    p.setup()
    out = p.predict(
        "example/music.wav",
        ddim_steps=50,
        guidance_scale=3.5,
        seed=42
    )