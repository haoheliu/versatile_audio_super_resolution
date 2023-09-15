import contextlib
import importlib
from huggingface_hub import hf_hub_download
import numpy as np
import torch

from inspect import isfunction
import os
import soundfile as sf
import time
import wave
import torchaudio
import progressbar
from librosa.filters import mel as librosa_mel_fn
from audiosr.lowpass import lowpass

hann_window = {}
mel_basis = {}


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


def _locate_cutoff_freq(stft, percentile=0.97):
    def _find_cutoff(x, percentile=0.95):
        percentile = x[-1] * percentile
        for i in range(1, x.shape[0]):
            if x[-i] < percentile:
                return x.shape[0] - i
        return 0

    magnitude = torch.abs(stft)
    energy = torch.cumsum(torch.sum(magnitude, dim=0), dim=0)
    return _find_cutoff(energy, percentile)


def pad_wav(waveform, target_length):
    waveform_length = waveform.shape[-1]
    assert waveform_length > 100, "Waveform is too short, %s" % waveform_length

    if waveform_length == target_length:
        return waveform

    # Pad
    temp_wav = np.zeros((1, target_length), dtype=np.float32)
    rand_start = 0

    temp_wav[:, rand_start : rand_start + waveform_length] = waveform
    return temp_wav


def lowpass_filtering_prepare_inference(dl_output):
    waveform = dl_output["waveform"]  # [1, samples]
    sampling_rate = dl_output["sampling_rate"]

    cutoff_freq = (
        _locate_cutoff_freq(dl_output["stft"], percentile=0.985) / 1024
    ) * 24000

    order = 8
    ftype = np.random.choice(["butter", "cheby1", "ellip", "bessel"])
    filtered_audio = lowpass(
        waveform.numpy().squeeze(),
        highcut=cutoff_freq,
        fs=sampling_rate,
        order=order,
        _type=ftype,
    )

    filtered_audio = torch.FloatTensor(filtered_audio.copy()).unsqueeze(0)

    if waveform.size(-1) <= filtered_audio.size(-1):
        filtered_audio = filtered_audio[..., : waveform.size(-1)]
    else:
        filtered_audio = torch.functional.pad(
            filtered_audio, (0, waveform.size(-1) - filtered_audio.size(-1))
        )

    return {"waveform_lowpass": filtered_audio}


def mel_spectrogram_train(y):
    global mel_basis, hann_window

    sampling_rate = 48000
    filter_length = 2048
    hop_length = 480
    win_length = 2048
    n_mel = 256
    mel_fmin = 20
    mel_fmax = 24000

    if 24000 not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, filter_length, n_mel, mel_fmin, mel_fmax)
        mel_basis[str(mel_fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(win_length).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((filter_length - hop_length) / 2), int((filter_length - hop_length) / 2)),
        mode="reflect",
    )

    y = y.squeeze(1)

    stft_spec = torch.stft(
        y,
        filter_length,
        hop_length=hop_length,
        win_length=win_length,
        window=hann_window[str(y.device)],
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    stft_spec = torch.abs(stft_spec)

    mel = spectral_normalize_torch(
        torch.matmul(mel_basis[str(mel_fmax) + "_" + str(y.device)], stft_spec)
    )

    return mel[0], stft_spec[0]


def pad_spec(log_mel_spec, target_frame):
    n_frames = log_mel_spec.shape[0]
    p = target_frame - n_frames
    # cut and pad
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        log_mel_spec = m(log_mel_spec)
    elif p < 0:
        log_mel_spec = log_mel_spec[0:target_frame, :]

    if log_mel_spec.size(-1) % 2 != 0:
        log_mel_spec = log_mel_spec[..., :-1]

    return log_mel_spec


def wav_feature_extraction(waveform, target_frame):
    waveform = waveform[0, ...]
    waveform = torch.FloatTensor(waveform)

    log_mel_spec, stft = mel_spectrogram_train(waveform.unsqueeze(0))

    log_mel_spec = torch.FloatTensor(log_mel_spec.T)
    stft = torch.FloatTensor(stft.T)

    log_mel_spec, stft = pad_spec(log_mel_spec, target_frame), pad_spec(
        stft, target_frame
    )
    return log_mel_spec, stft


def normalize_wav(waveform):
    waveform = waveform - np.mean(waveform)
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    return waveform * 0.5


def read_wav_file(filename):
    waveform, sr = torchaudio.load(filename)
    duration = waveform.size(-1) / sr
    pad_duration = duration + (2.56 - duration % 2.56)
    target_frame = int(pad_duration * 100)

    waveform = torchaudio.functional.resample(waveform, sr, 48000)

    waveform = waveform.numpy()[0, ...]

    waveform = normalize_wav(
        waveform
    )  # TODO rescaling the waveform will cause low LSD score

    waveform = waveform[None, ...]
    waveform = pad_wav(waveform, target_length=int(48000 * pad_duration))
    return waveform, target_frame, pad_duration


def read_audio_file(filename):
    waveform, target_frame, duration = read_wav_file(filename)
    log_mel_spec, stft = wav_feature_extraction(waveform, target_frame)
    return log_mel_spec, stft, waveform, duration, target_frame


def read_list(fname):
    result = []
    with open(fname, "r", encoding="utf-8") as f:
        for each in f.readlines():
            each = each.strip("\n")
            result.append(each)
    return result


def get_duration(fname):
    with contextlib.closing(wave.open(fname, "r")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / float(rate)


def get_bit_depth(fname):
    with contextlib.closing(wave.open(fname, "r")) as f:
        bit_depth = f.getsampwidth() * 8
        return bit_depth


def get_time():
    t = time.localtime()
    return time.strftime("%d_%m_%Y_%H_%M_%S", t)


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


def save_wave(waveform, savepath, name="outwav", samplerate=16000):
    if type(name) is not list:
        name = [name] * waveform.shape[0]

    for i in range(waveform.shape[0]):
        if waveform.shape[0] > 1:
            fname = "%s_%s.wav" % (
                os.path.basename(name[i])
                if (not ".wav" in name[i])
                else os.path.basename(name[i]).split(".")[0],
                i,
            )
        else:
            fname = (
                "%s.wav" % os.path.basename(name[i])
                if (not ".wav" in name[i])
                else os.path.basename(name[i]).split(".")[0]
            )
            # Avoid the file name too long to be saved
            if len(fname) > 255:
                fname = f"{hex(hash(fname))}.wav"

        path = os.path.join(savepath, fname)
        print("Save audio to %s" % path)
        sf.write(path, waveform[i, 0], samplerate=samplerate)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    try:
        return get_obj_from_str(config["target"])(**config.get("params", dict()))
    except:
        import ipdb

        ipdb.set_trace()


def default_audioldm_config(model_name="basic"):
    basic_config = get_basic_config()
    return basic_config


class MyProgressBar:
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


def download_checkpoint(checkpoint_name="basic"):
    model_id = "haoheliu/wellsolve_audio_super_resolution_48k"

    checkpoint_path = hf_hub_download(
        repo_id=model_id, filename=checkpoint_name + ".pth"
    )
    return checkpoint_path


def get_basic_config():
    return {
        "preprocessing": {
            "audio": {
                "sampling_rate": 48000,
                "max_wav_value": 32768,
                "duration": 10.24,
            },
            "stft": {"filter_length": 2048, "hop_length": 480, "win_length": 2048},
            "mel": {"n_mel_channels": 256, "mel_fmin": 20, "mel_fmax": 24000},
        },
        "augmentation": {"mixup": 0.5},
        "model": {
            "target": "audiosr.latent_diffusion.models.ddpm.LatentDiffusion",
            "params": {
                "first_stage_config": {
                    "base_learning_rate": 0.000008,
                    "target": "audiosr.latent_encoder.autoencoder.AutoencoderKL",
                    "params": {
                        "reload_from_ckpt": "/mnt/bn/lqhaoheliu/project/audio_generation_diffusion/log/vae/vae_48k_256/ds_8_kl_1/checkpoints/ckpt-checkpoint-484999.ckpt",
                        "sampling_rate": 48000,
                        "batchsize": 4,
                        "monitor": "val/rec_loss",
                        "image_key": "fbank",
                        "subband": 1,
                        "embed_dim": 16,
                        "time_shuffle": 1,
                        "ddconfig": {
                            "double_z": True,
                            "mel_bins": 256,
                            "z_channels": 16,
                            "resolution": 256,
                            "downsample_time": False,
                            "in_channels": 1,
                            "out_ch": 1,
                            "ch": 128,
                            "ch_mult": [1, 2, 4, 8],
                            "num_res_blocks": 2,
                            "attn_resolutions": [],
                            "dropout": 0.1,
                        },
                    },
                },
                "base_learning_rate": 0.0001,
                "warmup_steps": 5000,
                "optimize_ddpm_parameter": True,
                "sampling_rate": 48000,
                "batchsize": 16,
                "beta_schedule": "cosine",
                "linear_start": 0.0015,
                "linear_end": 0.0195,
                "num_timesteps_cond": 1,
                "log_every_t": 200,
                "timesteps": 1000,
                "unconditional_prob_cfg": 0.1,
                "parameterization": "v",
                "first_stage_key": "fbank",
                "latent_t_size": 128,
                "latent_f_size": 32,
                "channels": 16,
                "monitor": "val/loss_simple_ema",
                "scale_by_std": True,
                "unet_config": {
                    "target": "audiosr.latent_diffusion.modules.diffusionmodules.openaimodel.UNetModel",
                    "params": {
                        "image_size": 64,
                        "in_channels": 32,
                        "out_channels": 16,
                        "model_channels": 128,
                        "attention_resolutions": [8, 4, 2],
                        "num_res_blocks": 2,
                        "channel_mult": [1, 2, 3, 5],
                        "num_head_channels": 32,
                        "extra_sa_layer": True,
                        "use_spatial_transformer": True,
                        "transformer_depth": 1,
                    },
                },
                "evaluation_params": {
                    "unconditional_guidance_scale": 3.5,
                    "ddim_sampling_steps": 200,
                    "n_candidates_per_samples": 1,
                },
                "cond_stage_config": {
                    "concat_lowpass_cond": {
                        "cond_stage_key": "lowpass_mel",
                        "conditioning_key": "concat",
                        "target": "audiosr.latent_diffusion.modules.encoders.modules.VAEFeatureExtract",
                        "params": {
                            "first_stage_config": {
                                "base_learning_rate": 0.000008,
                                "target": "audiosr.latent_encoder.autoencoder.AutoencoderKL",
                                "params": {
                                    "sampling_rate": 48000,
                                    "batchsize": 4,
                                    "monitor": "val/rec_loss",
                                    "image_key": "fbank",
                                    "subband": 1,
                                    "embed_dim": 16,
                                    "time_shuffle": 1,
                                    "ddconfig": {
                                        "double_z": True,
                                        "mel_bins": 256,
                                        "z_channels": 16,
                                        "resolution": 256,
                                        "downsample_time": False,
                                        "in_channels": 1,
                                        "out_ch": 1,
                                        "ch": 128,
                                        "ch_mult": [1, 2, 4, 8],
                                        "num_res_blocks": 2,
                                        "attn_resolutions": [],
                                        "dropout": 0.1,
                                    },
                                },
                            }
                        },
                    }
                },
            },
        },
    }
