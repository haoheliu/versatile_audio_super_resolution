import os
import re

import yaml
import torch
import torchaudio

import audiosr.latent_diffusion.modules.phoneme_encoder.text as text
from audiosr.latent_diffusion.models.ddpm import LatentDiffusion
from audiosr.latent_diffusion.util import get_vits_phoneme_ids_no_padding
from audiosr.utils import (
    default_audioldm_config,
    download_checkpoint,
    read_audio_file,
    lowpass_filtering_prepare_inference,
    wav_feature_extraction,
)
import os


def seed_everything(seed):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def text2phoneme(data):
    return text._clean_text(re.sub(r"<.*?>", "", data), ["english_cleaners2"])


def text_to_filename(text):
    return text.replace(" ", "_").replace("'", "_").replace('"', "_")


def extract_kaldi_fbank_feature(waveform, sampling_rate, log_mel_spec):
    norm_mean = -4.2677393
    norm_std = 4.5689974

    if sampling_rate != 16000:
        waveform_16k = torchaudio.functional.resample(
            waveform, orig_freq=sampling_rate, new_freq=16000
        )
    else:
        waveform_16k = waveform

    waveform_16k = waveform_16k - waveform_16k.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform_16k,
        htk_compat=True,
        sample_frequency=16000,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=128,
        dither=0.0,
        frame_shift=10,
    )

    TARGET_LEN = log_mel_spec.size(0)

    # cut and pad
    n_frames = fbank.shape[0]
    p = TARGET_LEN - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[:TARGET_LEN, :]

    fbank = (fbank - norm_mean) / (norm_std * 2)

    return {"ta_kaldi_fbank": fbank}  # [1024, 128]


def make_batch_for_super_resolution(input_file, waveform=None, fbank=None):
    log_mel_spec, stft, waveform, duration, target_frame = read_audio_file(input_file)

    batch = {
        "waveform": torch.FloatTensor(waveform),
        "stft": torch.FloatTensor(stft),
        "log_mel_spec": torch.FloatTensor(log_mel_spec),
        "sampling_rate": 48000,
    }

    # print(batch["waveform"].size(), batch["stft"].size(), batch["log_mel_spec"].size())

    batch.update(lowpass_filtering_prepare_inference(batch))

    assert "waveform_lowpass" in batch.keys()
    lowpass_mel, lowpass_stft = wav_feature_extraction(
        batch["waveform_lowpass"], target_frame
    )
    batch["lowpass_mel"] = lowpass_mel

    for k in batch.keys():
        if type(batch[k]) == torch.Tensor:
            batch[k] = torch.FloatTensor(batch[k]).unsqueeze(0)

    return batch, duration


def round_up_duration(duration):
    return int(round(duration / 2.5) + 1) * 2.5


def build_model(ckpt_path=None, config=None, device=None, model_name="basic"):
    if device is None or device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    print("Loading AudioSR: %s" % model_name)
    print("Loading model on %s" % device)

    ckpt_path = download_checkpoint(model_name)

    if config is not None:
        assert type(config) is str
        config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
    else:
        config = default_audioldm_config(model_name)

    # # Use text as condition instead of using waveform during training
    config["model"]["params"]["device"] = device
    # config["model"]["params"]["cond_stage_key"] = "text"

    # No normalization here
    latent_diffusion = LatentDiffusion(**config["model"]["params"])

    resume_from_checkpoint = ckpt_path

    checkpoint = torch.load(resume_from_checkpoint, map_location=device)

    latent_diffusion.load_state_dict(checkpoint["state_dict"])

    latent_diffusion.eval()
    latent_diffusion = latent_diffusion.to(device)

    return latent_diffusion


def super_resolution(
    latent_diffusion,
    input_file,
    seed=42,
    ddim_steps=200,
    guidance_scale=3.5,
    latent_t_per_second=12.8,
    config=None,
):
    seed_everything(int(seed))
    waveform = None

    batch, duration = make_batch_for_super_resolution(input_file, waveform=waveform)

    with torch.no_grad():
        waveform = latent_diffusion.generate_batch(
            batch,
            unconditional_guidance_scale=guidance_scale,
            ddim_steps=ddim_steps,
            duration=duration,
        )

    return waveform
