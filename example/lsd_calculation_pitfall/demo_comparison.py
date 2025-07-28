#!/usr/bin/env python3
"""
Demo script to show the effect of energy scaling on LSD calculation.
Compares LSD values with and without energy unification.
"""

import numpy as np
import librosa
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Constants
EPS = 1e-8
SR = 22050
HOP_LENGTH = 512
N_FFT = 2048
LOWER_PERCENTILE = 10


def unify_lower_frequency_energy(gt_spectrogram, target_spectrogram, lower_percentile=10):
    """Scale target spectrogram to match GT lower frequency energy."""
    freq_bins, time_frames = gt_spectrogram.shape
    num_lower_bins = int(freq_bins * lower_percentile / 100)
    
    if num_lower_bins == 0:
        return target_spectrogram
    
    gt_lower_energy = np.mean(gt_spectrogram[:num_lower_bins, :])
    target_lower_energy = np.mean(target_spectrogram[:num_lower_bins, :])
    
    if target_lower_energy > 0:
        scaling_factor = gt_lower_energy / target_lower_energy
    else:
        scaling_factor = 1.0
    
    return target_spectrogram * scaling_factor, scaling_factor


def lsd(est, target):
    """Calculate Log Spectral Distance."""
    if isinstance(est, np.ndarray):
        est = torch.from_numpy(est).float()
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).float()
    
    if est.dim() == 2:
        est = est.unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0)
    
    if est.dim() == 3:
        est = est.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)
    
    lsd_result = torch.log10((target**2/((est + EPS)**2)) + EPS)**2
    lsd_per_frame = torch.mean(lsd_result, dim=2)**0.5
    lsd_per_frame = torch.mean(lsd_per_frame, dim=(0, 1))
    
    return lsd_per_frame


def main():
    """Demonstrate before/after comparison."""
    
    # File paths
    current_dir = Path(__file__).parent
    gt_path = current_dir / "gt_2.wav"
    selected_path = current_dir / "selected_2.wav"
    
    print("=" * 60)
    print("LSD BEFORE/AFTER ENERGY SCALING DEMONSTRATION")
    print("=" * 60)
    
    # Load audio files
    print("Loading audio files...")
    gt_audio, _ = librosa.load(gt_path, sr=SR)
    selected_audio, _ = librosa.load(selected_path, sr=SR)
    
    # Ensure same length
    min_len = min(len(gt_audio), len(selected_audio))
    gt_audio = gt_audio[:min_len]
    selected_audio = selected_audio[:min_len]
    
    print(f"Audio length: {min_len} samples ({min_len/SR:.2f}s)")
    
    # Compute STFT
    print("Computing STFT...")
    stft1 = librosa.stft(gt_audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    stft2 = librosa.stft(selected_audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    
    # Convert to magnitude spectrograms
    mag1 = np.abs(stft1)
    mag2 = np.abs(stft2)
    
    print(f"Spectrogram shape: {mag1.shape}")
    print(f"Lower {LOWER_PERCENTILE}% frequency bins: {int(mag1.shape[0] * LOWER_PERCENTILE / 100)}")
    
    # Calculate energy statistics
    num_lower_bins = int(mag1.shape[0] * LOWER_PERCENTILE / 100)
    gt_lower_energy = np.mean(mag1[:num_lower_bins, :])
    selected_lower_energy = np.mean(mag2[:num_lower_bins, :])
    
    print(f"\nEnergy Analysis:")
    print(f"GT lower frequency energy: {gt_lower_energy:.6f}")
    print(f"Selected lower frequency energy: {selected_lower_energy:.6f}")
    print(f"Energy ratio (selected/gt): {selected_lower_energy/gt_lower_energy:.4f}")
    
    # Calculate LSD WITHOUT scaling
    print("\nCalculating LSD WITHOUT energy scaling...")
    lsd_without_scaling = lsd(mag1, mag2)
    if isinstance(lsd_without_scaling, torch.Tensor):
        lsd_without_scaling = lsd_without_scaling.detach().numpy()
    
    # Apply scaling
    print("Applying energy scaling...")
    mag2_scaled, scaling_factor = unify_lower_frequency_energy(mag1, mag2, LOWER_PERCENTILE)
    print(f"Scaling factor: {scaling_factor:.6f}")
    
    # Calculate LSD WITH scaling
    print("Calculating LSD WITH energy scaling...")
    lsd_with_scaling = lsd(mag1, mag2_scaled)
    if isinstance(lsd_with_scaling, torch.Tensor):
        lsd_with_scaling = lsd_with_scaling.detach().numpy()
    
    # Results comparison
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    
    stats_data = [
        ['Metric', 'Without Scaling', 'With Scaling', 'Improvement'],
        ['Average LSD', f'{np.mean(lsd_without_scaling):.4f}', f'{np.mean(lsd_with_scaling):.4f}', 
         f'{np.mean(lsd_without_scaling) - np.mean(lsd_with_scaling):.4f}'],
        ['Min LSD', f'{np.min(lsd_without_scaling):.4f}', f'{np.min(lsd_with_scaling):.4f}', 
         f'{np.min(lsd_without_scaling) - np.min(lsd_with_scaling):.4f}'],
        ['Max LSD', f'{np.max(lsd_without_scaling):.4f}', f'{np.max(lsd_with_scaling):.4f}', 
         f'{np.max(lsd_without_scaling) - np.max(lsd_with_scaling):.4f}'],
        ['Std LSD', f'{np.std(lsd_without_scaling):.4f}', f'{np.std(lsd_with_scaling):.4f}', 
         f'{np.std(lsd_without_scaling) - np.std(lsd_with_scaling):.4f}'],
        ['Median LSD', f'{np.median(lsd_without_scaling):.4f}', f'{np.median(lsd_with_scaling):.4f}', 
         f'{np.median(lsd_without_scaling) - np.median(lsd_with_scaling):.4f}']
    ]
    
    # Print formatted table
    print(f"{'Metric':<20} {'Without Scaling':<15} {'With Scaling':<15} {'Improvement':<12}")
    print("-" * 70)
    for i in range(1, len(stats_data)):
        print(f"{stats_data[i][0]:<20} {stats_data[i][1]:<15} {stats_data[i][2]:<15} {stats_data[i][3]:<12}")
    
    improvement_pct = ((np.mean(lsd_without_scaling) - np.mean(lsd_with_scaling)) / np.mean(lsd_without_scaling)) * 100
    print(f"\nOverall improvement: {improvement_pct:.2f}%")
    
    # Generate comparison plot
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Spectrograms comparison
        im1 = ax1.imshow(np.log10(mag1 + EPS), aspect='auto', origin='lower', cmap='viridis')
        ax1.set_title('Ground Truth Spectrogram')
        ax1.set_ylabel('Frequency Bin')
        plt.colorbar(im1, ax=ax1)
        
        im2 = ax2.imshow(np.log10(mag2 + EPS), aspect='auto', origin='lower', cmap='viridis')
        ax2.set_title('Selected Spectrogram (Original)')
        plt.colorbar(im2, ax=ax2)
        
        # 2. LSD comparison
        time_frames = len(lsd_without_scaling)
        time_indices = np.arange(time_frames)
        
        ax3.plot(time_indices, lsd_without_scaling, label='Without Scaling', alpha=0.7)
        ax3.plot(time_indices, lsd_with_scaling, label='With Scaling', alpha=0.7)
        ax3.axhline(y=np.mean(lsd_without_scaling), color='blue', linestyle='--', alpha=0.5)
        ax3.axhline(y=np.mean(lsd_with_scaling), color='orange', linestyle='--', alpha=0.5)
        ax3.set_title('LSD Over Time Comparison')
        ax3.set_xlabel('Time Frame')
        ax3.set_ylabel('LSD')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 3. LSD histogram
        ax4.hist(lsd_without_scaling, bins=30, alpha=0.7, label='Without Scaling', density=True)
        ax4.hist(lsd_with_scaling, bins=30, alpha=0.7, label='With Scaling', density=True)
        ax4.axvline(np.mean(lsd_without_scaling), color='blue', linestyle='--', alpha=0.7)
        ax4.axvline(np.mean(lsd_with_scaling), color='orange', linestyle='--', alpha=0.7)
        ax4.set_title('LSD Distribution Comparison')
        ax4.set_xlabel('LSD')
        ax4.set_ylabel('Density')
        ax4.legend()
        
        plt.tight_layout()
        
        # Save plot
        comparison_plot = current_dir / "before_after_comparison.png"
        plt.savefig(comparison_plot, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nComparison plot saved to: {comparison_plot}")
        
    except ImportError:
        print("Matplotlib not available, skipping plot generation.")
    
    # Save detailed results
    results_file = current_dir / "before_after_comparison_results.txt"
    with open(results_file, 'w') as f:
        f.write("LSD Before/After Energy Scaling Comparison\n")
        f.write("=" * 60 + "\n")
        f.write(f"Ground Truth: {gt_path.name}\n")
        f.write(f"Target: {selected_path.name}\n")
        f.write(f"Lower frequency unification: {LOWER_PERCENTILE}%\n")
        f.write(f"Scaling factor: {scaling_factor:.6f}\n\n")
        
        f.write("Energy Analysis:\n")
        f.write(f"GT lower frequency energy: {gt_lower_energy:.6f}\n")
        f.write(f"Selected lower frequency energy: {selected_lower_energy:.6f}\n")
        f.write(f"Energy ratio: {selected_lower_energy/gt_lower_energy:.4f}\n\n")
        
        f.write("LSD Statistics:\n")
        f.write(f"{'Metric':<20} {'Without Scaling':<15} {'With Scaling':<15} {'Improvement':<12}\n")
        f.write("-" * 70 + "\n")
        for i in range(1, len(stats_data)):
            f.write(f"{stats_data[i][0]:<20} {stats_data[i][1]:<15} {stats_data[i][2]:<15} {stats_data[i][3]:<12}\n")
        
        f.write(f"\nOverall improvement: {improvement_pct:.2f}%\n")
    
    print(f"Detailed results saved to: {results_file}")
    print("=" * 60)


if __name__ == "__main__":
    main() 