import gradio as gr
import numpy as np
import librosa
import torch
from audiosr import build_model
from audiosr.pipeline import super_resolution_from_waveform, super_resolution_batch
import os

def calculate_amplitude_stats(audio):
    """Calculate amplitude statistics for audio normalization"""
    rms = np.sqrt(np.mean(np.square(audio)))
    peak = np.max(np.abs(audio))
    return rms, peak

def normalize_chunk_amplitude(processed_chunk, original_chunk):
    """Normalize processed chunk to match original chunk's amplitude characteristics"""
    orig_rms, orig_peak = calculate_amplitude_stats(original_chunk)
    proc_rms, proc_peak = calculate_amplitude_stats(processed_chunk)
    
    # Avoid division by zero
    if proc_rms < 1e-8:
        return processed_chunk
    
    # Calculate scaling factor based on RMS ratio
    scale_factor = orig_rms / proc_rms
    
    # Apply scaling while ensuring we don't exceed the original peak ratio
    peak_ratio = orig_peak / proc_peak if proc_peak > 0 else 1
    scale_factor = min(scale_factor, peak_ratio)
    
    return processed_chunk * scale_factor


def process_chunks_batch(audiosr, chunks, original_lengths, sr, guidance_scale, ddim_steps):
    """
    Process multiple chunks in a single GPU batch.
    
    Args:
        audiosr: The model
        chunks: List of numpy arrays (audio chunks)
        original_lengths: List of original lengths
        sr: Sample rate
        guidance_scale: Guidance scale
        ddim_steps: DDIM steps
        
    Returns:
        List of processed chunks as numpy arrays
    """
    adjusted_ddim_steps = min(ddim_steps - 2, 998)
    
    # Use batch processing for true GPU parallelism
    processed_list = super_resolution_batch(
            audiosr,
        chunks,
            guidance_scale=guidance_scale,
            ddim_steps=adjusted_ddim_steps
        )
        
    # Normalize amplitude for each result
    results = []
    for i, (processed, chunk, orig_len) in enumerate(zip(processed_list, chunks, original_lengths)):
        result = normalize_chunk_amplitude(processed, chunk)
        result = np.squeeze(result)
        if len(result) > orig_len:
            result = result[:orig_len]
        results.append(result)
    
    return results


def process_chunk_direct(audiosr, chunk, original_length, sr, guidance_scale, ddim_steps):
    """
    Process a single chunk directly without saving to file.
    """
    adjusted_ddim_steps = min(ddim_steps - 2, 998)
            
    # Process chunk directly (no file I/O)
    processed = super_resolution_from_waveform(
        audiosr,
        chunk,
        guidance_scale=guidance_scale,
        ddim_steps=adjusted_ddim_steps
    )
    
    # Normalize amplitude
    result = normalize_chunk_amplitude(processed, chunk)
    
    # Trim to original length
    result = np.squeeze(result)
    if len(result) > original_length:
        result = result[:original_length]
        
        return result
    

def process_audio_channel(audiosr, audio_channel, sr, guidance_scale, ddim_steps, batch_size=1):
    """Process a single audio channel with optional batch processing"""
    # Calculate chunk parameters
    chunk_duration = 5.1  # seconds
    chunk_size = int(chunk_duration * sr)
    overlap_duration = 0.5  # 500ms overlap
    overlap_size = int(overlap_duration * sr)
    
    # Calculate number of chunks
    total_samples = len(audio_channel)
    num_chunks = int(np.ceil(total_samples / (chunk_size - overlap_size)))
    
    print(f"Total chunks to process: {num_chunks}, batch_size: {batch_size}")
    
    # Collect all chunks and their metadata
    all_chunks = []
    all_orig_lens = []
    chunk_positions = []  # (start, end) for each chunk
    
    for i in range(num_chunks):
        start = i * (chunk_size - overlap_size)
        end = min(start + chunk_size, total_samples)
        chunk = audio_channel[start:end]
        
        all_chunks.append(chunk)
        all_orig_lens.append(len(chunk))
        chunk_positions.append((start, end))
    
    # Process chunks in batches
    processed_results = []
    
    if batch_size > 1:
        # True GPU batch processing
        num_batches = int(np.ceil(num_chunks / batch_size))
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, num_chunks)
            
            batch_chunks = all_chunks[batch_start:batch_end]
            batch_orig_lens = all_orig_lens[batch_start:batch_end]
            
            print(f"\nBatch {batch_idx + 1}/{num_batches}: Processing chunks {batch_start + 1}-{batch_end}/{num_chunks}")
            for j, (start, end) in enumerate(chunk_positions[batch_start:batch_end]):
                print(f"  Chunk {batch_start + j + 1}: {start/sr:.2f}s - {end/sr:.2f}s")
            
            # Process batch
            batch_results = process_chunks_batch(
                audiosr, batch_chunks, batch_orig_lens, sr, guidance_scale, ddim_steps
            )
            processed_results.extend(batch_results)
            
            # Clean up GPU memory between batches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
        # Sequential processing (fallback)
        for i, (chunk, orig_len, (start, end)) in enumerate(zip(all_chunks, all_orig_lens, chunk_positions)):
            print(f"\nChunk {i+1}/{num_chunks}: {start/sr:.2f}s - {end/sr:.2f}s ({(end-start)/sr:.2f}s)")
            
            result = process_chunk_direct(
                audiosr, chunk, orig_len, sr, guidance_scale, ddim_steps
            )
            processed_results.append(result)
        
    # Apply crossfade and concatenate
    processed_chunks = []
    for i, result in enumerate(processed_results):
        # Ensure processed chunk is 2D
        processed_chunk = np.squeeze(result)
        if len(processed_chunk.shape) == 1:
            processed_chunk = processed_chunk.reshape(1, -1)
        
        # Apply crossfade for overlapping regions (except for first chunk)
        if i > 0:
            actual_overlap_size = overlap_size
            actual_overlap_size = min(actual_overlap_size, processed_chunk.shape[1], processed_chunks[-1].shape[1])
            
            # Create fade curves
            fade_in = np.linspace(0, 1, actual_overlap_size).reshape(1, -1)
            fade_out = np.linspace(1, 0, actual_overlap_size).reshape(1, -1)
            
            # Get overlapping regions
            current_overlap = processed_chunk[:, :actual_overlap_size]
            previous_overlap = processed_chunks[-1][:, -actual_overlap_size:]
            
            # Adjust fade curves based on RMS ratio
            current_rms = np.sqrt(np.mean(np.square(current_overlap)))
            previous_rms = np.sqrt(np.mean(np.square(previous_overlap)))
            
            if current_rms > 0 and previous_rms > 0:
                rms_ratio = np.sqrt(previous_rms / current_rms)
                fade_in = fade_in * rms_ratio
            
            # Apply crossfade
            processed_chunk[:, :actual_overlap_size] *= fade_in
            processed_chunks[-1][:, -actual_overlap_size:] *= fade_out
            processed_chunks[-1][:, -actual_overlap_size:] += processed_chunk[:, :actual_overlap_size]
            processed_chunk = processed_chunk[:, actual_overlap_size:]
        
        processed_chunks.append(processed_chunk)
    
    print(f"\nConcatenating {len(processed_chunks)} processed chunks...")
    return np.concatenate(processed_chunks, axis=1)

def normalize_audio(audio):
    """Normalize audio to be within [-1, 1] range"""
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio / max_val
    return audio

def convert_audio_for_gradio(audio):
    """Convert audio to the format expected by Gradio"""
    # Ensure audio is in float32 format
    audio = audio.astype(np.float32)
    # Normalize to [-1, 1] range
    audio = normalize_audio(audio)
    # Transpose to (samples, channels) format if needed
    if audio.shape[0] == 2:  # If first dimension is channels
        audio = audio.T
    return audio

def warmup_model(audiosr, ddim_steps):
    """
    Warmup the model with a single dummy chunk to trigger CUDA kernel compilation.
    This prevents memory spikes during actual batch processing.
    
    Optimizations:
    - Use short 2-second audio (reduces memory usage during warmup)
    - Use only 10 DDIM steps (enough to compile all kernels, faster warmup)
    """
    print("\n" + "="*50)
    print("Warming up model (triggering CUDA kernel compilation)...")
    print("="*50)
    
    # Create a short 2-second dummy waveform to minimize memory usage
    # (NOT exactly 5.12s multiple to ensure padding logic triggers correctly)
    dummy_duration = 2.0
    dummy_sr = 48000
    dummy_waveform = np.zeros(int(dummy_duration * dummy_sr), dtype=np.float32)
    
    # Add small noise to create realistic audio signal (avoids numerical edge cases)
    dummy_waveform += np.random.randn(len(dummy_waveform)).astype(np.float32) * 0.01
    
    # Use minimal DDIM steps - only need to trigger kernel compilation, not quality output
    warmup_ddim_steps = 10  # Much faster than full ddim_steps, still compiles all kernels
    
    _ = super_resolution_from_waveform(
        audiosr,
        dummy_waveform,
        guidance_scale=2.5,
        ddim_steps=warmup_ddim_steps
    )
    
    # Clear GPU cache after warmup to free temporary memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("Warmup complete!")
    
    print("Warmup complete! Model is ready for batch processing.\n")


def inference(audio_file, model_name, guidance_scale, ddim_steps, batch_size):
    # Initialize the model
    audiosr = build_model(model_name=model_name)
    
    # Warmup model to trigger CUDA kernel compilation (prevents memory spikes during batch processing)
    warmup_model(audiosr, ddim_steps)
    
    # Load the audio file with original number of channels
    audio, sr = librosa.load(audio_file, sr=48000, mono=False)
    
    # Convert to stereo if mono
    if len(audio.shape) == 1:
        audio = np.stack([audio, audio])
    
    print(f"\nProcessing audio file of length: {audio.shape[1]/sr:.2f} seconds")
    print(f"Number of channels: {audio.shape[0]}")
    print(f"Batch size: {batch_size}")
    
    # Process each channel separately
    processed_channels = []
    for channel_idx in range(audio.shape[0]):
        print(f"\n{'='*50}")
        print(f"Processing channel {channel_idx + 1}/{audio.shape[0]}")
        print(f"{'='*50}")
        channel_audio = audio[channel_idx]
        processed_channel = process_audio_channel(
            audiosr, channel_audio, sr, guidance_scale, ddim_steps, batch_size
        )
        # Ensure the channel is 1D
        processed_channel = processed_channel.squeeze()
        processed_channels.append(processed_channel)
    
    # Stack channels for stereo output (shape will be [2, samples])
    if len(processed_channels[0].shape) > 1:
        # If channels are 2D, take the first row
        processed_channels = [channel[0] if len(channel.shape) > 1 else channel for channel in processed_channels]
    
    final_audio = np.stack(processed_channels)
    
    # Convert audio to the format expected by Gradio
    final_audio = convert_audio_for_gradio(final_audio)
    
    print(f"\nFinal audio shape: {final_audio.shape}")
    print(f"Final audio length: {final_audio.shape[0]/sr:.2f} seconds")
    print(f"Audio value range: [{final_audio.min():.3f}, {final_audio.max():.3f}]")
    print(f"Audio dtype: {final_audio.dtype}")
    
    return (48000, final_audio)

iface = gr.Interface(
    fn=inference, 
    inputs=[
        gr.Audio(type="filepath", label="Input Audio"),
        gr.Dropdown(["basic", "speech"], value="basic", label="Model"),
        gr.Slider(1, 10, value=2.6, step=0.1, label="Guidance Scale"),  
        gr.Slider(1, 100, value=100, step=1, label="DDIM Steps"),
        gr.Slider(1, 8, value=2, step=1, label="Batch Size (并行处理chunk数)")
    ],
    outputs=gr.Audio(type="numpy", label="Output Audio"),
    title="AudioSR - Audio Super Resolution",
    description="音频超分辨率处理。支持GPU批量并行处理多个音频chunk，加速长音频处理。Batch Size > 1 启用真正的GPU并行。"
)

iface.launch()
