import gradio as gr
import numpy as np
import librosa
from audiosr import super_resolution, build_model
import tempfile
import soundfile as sf
import os

def detect_audio_end(audio, sr, window_size=2048, hop_length=512, threshold_db=-50):
    """Detect the end of actual audio content using RMS energy"""
    # Calculate RMS energy
    rms = librosa.feature.rms(y=audio, frame_length=window_size, hop_length=hop_length)[0]
    
    # Convert to dB
    db = librosa.amplitude_to_db(rms, ref=np.max)
    
    # Find the last frame above threshold
    valid_frames = np.where(db > threshold_db)[0]
    
    if len(valid_frames) > 0:
        last_frame = valid_frames[-1]
        # Convert frame index to sample index
        last_sample = (last_frame + 1) * hop_length
        return last_sample
    return len(audio)

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

def process_chunk(audiosr, chunk, sr, guidance_scale, ddim_steps, is_last_chunk=False, target_length=None):
    # Create a temporary directory in the current working directory
    temp_dir = os.path.join(os.getcwd(), "temp_audio")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create a unique temporary file path
    temp_path = os.path.join(temp_dir, f"chunk_{np.random.randint(0, 1000000)}.wav")
    
    try:
        # Save chunk to temporary file
        sf.write(temp_path, chunk, sr)
        
        # For the last chunk, adjust ddim_steps based on length
        if is_last_chunk:
            chunk_duration = len(chunk) / sr
            # Scale ddim_steps proportionally for shorter chunks, ensuring it stays within valid bounds
            # Subtract 2 to ensure we're well within the valid range (0 to ddim_steps-1)
            max_steps = min(ddim_steps - 2, 998)  # Ensure we never exceed the valid range
            adjusted_ddim_steps = max(10, min(max_steps, int(ddim_steps * (chunk_duration / 5.1))))
            print(f"Adjusted ddim_steps for last chunk: {adjusted_ddim_steps}")
        else:
            adjusted_ddim_steps = min(ddim_steps - 2, 998)  # Also bound regular chunks for safety
        
        # Process the chunk
        processed_chunk = super_resolution(
            audiosr,
            temp_path,
            guidance_scale=guidance_scale,
            ddim_steps=adjusted_ddim_steps
        )
        
        result = processed_chunk  # Keep the result as is, no channel selection
        
        # Normalize the processed chunk's amplitude relative to input chunk
        result = normalize_chunk_amplitude(result, chunk)
        
        # For the last chunk, ensure the output length matches the input length
        if is_last_chunk and target_length is not None:
            # Calculate the scale factor between input and output
            scale_factor = len(result) / len(chunk)
            target_output_length = int(target_length * scale_factor)
            
            # Find the actual end of audio content
            audio_end = detect_audio_end(result, sr)
            
            # Use the minimum of detected end and target length
            end_point = min(audio_end, target_output_length)
            result = result[:end_point]
            
            print(f"Adjusted last chunk length from {len(result)} to {end_point} samples")
        
        return result
    
    finally:
        # Clean up: remove the temporary file
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            print(f"Warning: Could not remove temporary file {temp_path}: {e}")

def process_audio_channel(audiosr, audio_channel, sr, guidance_scale, ddim_steps):
    """Process a single audio channel"""
    # Calculate chunk parameters
    chunk_duration = 5.1  # seconds
    chunk_size = int(chunk_duration * sr)
    overlap_duration = 0.5  # 500ms overlap
    overlap_size = int(overlap_duration * sr)
    
    # Process audio in chunks
    processed_chunks = []
    
    # Calculate number of chunks
    total_samples = len(audio_channel)
    num_chunks = int(np.ceil(total_samples / (chunk_size - overlap_size)))
    
    print(f"Total chunks to process: {num_chunks}")
    
    for i in range(num_chunks):
        # Calculate chunk boundaries
        start = i * (chunk_size - overlap_size)
        end = min(start + chunk_size, total_samples)
        
        print(f"\nProcessing chunk {i+1}/{num_chunks} with Audio Super Resolution")
        print(f"Chunk time range: {start/sr:.2f}s to {end/sr:.2f}s of total {total_samples/sr:.2f}s")
        print(f"Chunk size: {(end-start)/sr:.2f} seconds")
        
        # Extract chunk
        chunk = audio_channel[start:end]
        
        # Check if this is the last chunk
        is_last_chunk = (i == num_chunks - 1)
        
        # Process chunk
        # For last chunk, pass the actual remaining length
        if is_last_chunk:
            remaining_samples = total_samples - start
            processed_chunk = process_chunk(audiosr, chunk, sr, guidance_scale, ddim_steps, 
                                         is_last_chunk=True, target_length=remaining_samples)
        else:
            processed_chunk = process_chunk(audiosr, chunk, sr, guidance_scale, ddim_steps, 
                                         is_last_chunk=False)
        
        # Ensure processed chunk is 2D by removing any singleton dimensions
        processed_chunk = np.squeeze(processed_chunk)
        if len(processed_chunk.shape) == 1:
            processed_chunk = processed_chunk.reshape(1, -1)
        
        # Apply crossfade for overlapping regions (except for first chunk)
        if i > 0:
            print(f"Applying crossfade with previous chunk (overlap: {overlap_duration}s)")
            
            # Calculate the actual overlap size based on the processed chunk size
            scale_factor = processed_chunk.shape[1] / len(chunk)
            actual_overlap_size = int(overlap_size * scale_factor)
            
            # Create fade curves with the correct size
            fade_in = np.linspace(0, 1, actual_overlap_size)
            fade_out = np.linspace(1, 0, actual_overlap_size)
            
            # Reshape fade curves to match the processed chunk dimensions
            fade_in = fade_in.reshape(1, -1)
            fade_out = fade_out.reshape(1, -1)
            
            # Get the overlapping regions
            current_overlap = processed_chunk[:, :actual_overlap_size]
            previous_overlap = processed_chunks[-1][:, -actual_overlap_size:]
            
            # Calculate average RMS of the overlapping regions
            current_rms = np.sqrt(np.mean(np.square(current_overlap)))
            previous_rms = np.sqrt(np.mean(np.square(previous_overlap)))
            
            # Adjust fade curves based on RMS ratio to maintain energy consistency
            if current_rms > 0 and previous_rms > 0:
                rms_ratio = np.sqrt(previous_rms / current_rms)
                fade_in = fade_in * rms_ratio
            
            # Apply crossfade
            processed_chunk[:, :actual_overlap_size] *= fade_in
            processed_chunks[-1][:, -actual_overlap_size:] *= fade_out
            
            # Add overlapping regions
            processed_chunks[-1][:, -actual_overlap_size:] += processed_chunk[:, :actual_overlap_size]
            processed_chunk = processed_chunk[:, actual_overlap_size:]
        
        processed_chunks.append(processed_chunk)
        print(f"Chunk {i+1} processed successfully")
        print(f"Processed chunk shape: {processed_chunk.shape}")
    
    # Concatenate all processed chunks along the time axis (axis=1)
    print("\nConcatenating processed chunks...")
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

def inference(audio_file, model_name, guidance_scale, ddim_steps):
    # Initialize the model
    audiosr = build_model(model_name=model_name)
    
    # Load the audio file with original number of channels
    audio, sr = librosa.load(audio_file, sr=48000, mono=False)
    
    # Convert to stereo if mono
    if len(audio.shape) == 1:
        audio = np.stack([audio, audio])
    
    print(f"\nProcessing audio file of length: {audio.shape[1]/sr:.2f} seconds")
    print(f"Number of channels: {audio.shape[0]}")
    
    # Process each channel separately
    processed_channels = []
    for channel_idx in range(audio.shape[0]):
        print(f"\nProcessing channel {channel_idx + 1}")
        channel_audio = audio[channel_idx]
        processed_channel = process_audio_channel(audiosr, channel_audio, sr, guidance_scale, ddim_steps)
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
    
    print(f"Final audio shape: {final_audio.shape}")
    print(f"Final audio length: {final_audio.shape[0]/sr:.2f} seconds")
    print(f"Audio value range: [{final_audio.min():.3f}, {final_audio.max():.3f}]")
    print(f"Audio dtype: {final_audio.dtype}")
    
    # Clean up temporary directory
    temp_dir = os.path.join(os.getcwd(), "temp_audio")
    try:
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)
    except Exception as e:
        print(f"Warning: Could not clean up temporary directory: {e}")
    
    return (48000, final_audio)

iface = gr.Interface(
    fn=inference, 
    inputs=[
        gr.Audio(type="filepath", label="Input Audio"),
        gr.Dropdown(["basic", "speech"], value="basic", label="Model"),
        gr.Slider(1, 10, value=2.6, step=0.1, label="Guidance Scale"),  
        gr.Slider(1, 100, value=100, step=1, label="DDIM Steps")
    ],
    outputs=gr.Audio(type="numpy", label="Output Audio"),
    title="AudioSR",
    description="Audio Super Resolution with AudioSR"
)

# Create temp directory on startup
temp_dir = os.path.join(os.getcwd(), "temp_audio")
os.makedirs(temp_dir, exist_ok=True)

iface.launch()