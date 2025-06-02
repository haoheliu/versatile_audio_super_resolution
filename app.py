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
            # Scale ddim_steps proportionally for shorter chunks
            adjusted_ddim_steps = max(10, int(ddim_steps * (chunk_duration / 5.1)))
            print(f"Adjusted ddim_steps for last chunk: {adjusted_ddim_steps}")
        else:
            adjusted_ddim_steps = ddim_steps
        
        # Process the chunk
        processed_chunk = super_resolution(
            audiosr,
            temp_path,
            guidance_scale=guidance_scale,
            ddim_steps=adjusted_ddim_steps
        )
        
        result = processed_chunk[0]  # Get first channel if stereo
        
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

def inference(audio_file, model_name, guidance_scale, ddim_steps):
    # Initialize the model
    audiosr = build_model(model_name=model_name)
    
    # Load the audio file
    audio, sr = librosa.load(audio_file, sr=48000, mono=True)
    
    print(f"\nProcessing audio file of length: {len(audio)/sr:.2f} seconds")
    
    # Calculate chunk parameters
    chunk_duration = 5.1  # seconds
    chunk_size = int(chunk_duration * sr)
    overlap_duration = 0.1  # 100ms overlap
    overlap_size = int(overlap_duration * sr)
    
    print(f"Chunk duration: {chunk_duration} seconds")
    print(f"Overlap duration: {overlap_duration} seconds")
    
    # Create temporary directory if it doesn't exist
    temp_dir = os.path.join(os.getcwd(), "temp_audio")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Process audio in chunks
    processed_chunks = []
    
    # Calculate number of chunks
    total_samples = len(audio)
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
        chunk = audio[start:end]
        
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
        
        # Ensure processed chunk is 1D
        if len(processed_chunk.shape) > 1:
            processed_chunk = processed_chunk.flatten()
        
        # Apply crossfade for overlapping regions (except for first chunk)
        if i > 0:
            print(f"Applying crossfade with previous chunk (overlap: {overlap_duration}s)")
            
            # Calculate the actual overlap size based on the processed chunk size
            scale_factor = len(processed_chunk) / len(chunk)
            actual_overlap_size = int(overlap_size * scale_factor)
            
            # Create fade curves with the correct size
            fade_in = np.linspace(0, 1, actual_overlap_size)
            fade_out = np.linspace(1, 0, actual_overlap_size)
            
            # Get the overlapping regions
            current_overlap = processed_chunk[:actual_overlap_size]
            previous_overlap = processed_chunks[-1][-actual_overlap_size:]
            
            # Calculate average RMS of the overlapping regions
            current_rms = np.sqrt(np.mean(np.square(current_overlap)))
            previous_rms = np.sqrt(np.mean(np.square(previous_overlap)))
            
            # Adjust fade curves based on RMS ratio to maintain energy consistency
            if current_rms > 0 and previous_rms > 0:
                rms_ratio = np.sqrt(previous_rms / current_rms)
                fade_in = fade_in * rms_ratio
            
            # Apply crossfade
            processed_chunk[:actual_overlap_size] *= fade_in
            processed_chunks[-1][-actual_overlap_size:] *= fade_out
            
            # Add overlapping regions
            processed_chunks[-1][-actual_overlap_size:] += processed_chunk[:actual_overlap_size]
            processed_chunk = processed_chunk[actual_overlap_size:]
        
        processed_chunks.append(processed_chunk)
        print(f"Chunk {i+1} processed successfully")
        print(f"Processed chunk shape: {processed_chunk.shape}")
    
    # Concatenate all processed chunks
    print("\nConcatenating processed chunks...")
    final_audio = np.concatenate(processed_chunks)
    
    print(f"Final audio length: {len(final_audio)/sr:.2f} seconds")
    
    # Clean up temporary directory
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
        gr.Slider(1, 10, value=3.5, step=0.1, label="Guidance Scale"),  
        gr.Slider(1, 100, value=50, step=1, label="DDIM Steps")
    ],
    outputs=gr.Audio(type="numpy", label="Output Audio"),
    title="AudioSR",
    description="Audio Super Resolution with AudioSR"
)

# Create temp directory on startup
temp_dir = os.path.join(os.getcwd(), "temp_audio")
os.makedirs(temp_dir, exist_ok=True)

iface.launch()