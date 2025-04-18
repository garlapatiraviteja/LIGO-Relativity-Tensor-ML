import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec

# 1. Generate more realistic LIGO strain data with a gravitational wave signal
np.random.seed(0)
time = np.linspace(0, 16, 4096)  # 16 seconds of data at 256 Hz sample rate

# Create a simulated gravitational wave chirp (increasing frequency and amplitude)
def chirp_signal(t, t0=8.0, f0=20, f1=150, duration=0.5, amplitude=5e-21):
    # Only return signal during the chirp period
    mask = (t > t0) & (t < t0 + duration)
    signal = np.zeros_like(t)
    t_rel = t[mask] - t0
    phase = 2 * np.pi * (f0 * t_rel + 0.5 * (f1 - f0) * t_rel**2 / duration)
    # Amplitude increases as the binary gets closer
    amp_mod = amplitude * (1 + 3 * t_rel/duration)
    signal[mask] = amp_mod * np.sin(phase)
    return signal

# Create a realistic LIGO background with colored noise
def colored_noise(t, fs=256.0, amplitude=1e-22):
    n = len(t)
    noise = np.random.randn(n)
    # Create frequency-dependent filter to simulate LIGO noise curve
    freq = np.fft.rfftfreq(n, 1/fs)
    fft_noise = np.fft.rfft(noise)
    # Simple approximation of LIGO noise curve
    # High sensitivity around 100-300 Hz, reduced sensitivity at low and high frequencies
    noise_curve = 1.0 + 10.0/(freq + 20) + freq/1000.0
    fft_noise *= amplitude / noise_curve
    return np.fft.irfft(fft_noise, n)

# Combine signal and noise
strain = chirp_signal(time) + colored_noise(time)

# 2. Create higher-resolution spectrogram to visualize energy tensor
fs = 256  # Sample rate
frequencies, times, Sxx = spectrogram(strain, fs=fs, nperseg=512, noverlap=384, window='hann')
energy_tensor = Sxx  # LIGO-based ML tensor

# 3. Construct Stress-Energy Tensor components - expanded to 4x4 matrix
# Simplified 2D version for educational illustration
avg_energy = np.mean(energy_tensor)
# Create a more physically motivated stress-energy tensor 
# T^μν = diag(ρc², p, p, p) in simplified form
c_squared = 1  # Set c=1 for relativistic units
energy_density = avg_energy
pressure = avg_energy / 3  # Assume radiation-like equation of state

stress_energy_tensor = np.array([
    [energy_density * c_squared, 0],
    [0, pressure]
])

# 4. Lorentz Transformation with enhanced explanation
def lorentz_transform(tensor, velocity, speed_of_light=1):
    gamma = 1 / np.sqrt(1 - (velocity ** 2) / (speed_of_light ** 2))
    lorentz_matrix = np.array([
        [gamma, -gamma * velocity],
        [-gamma * velocity, gamma]
    ])
    transformed_tensor = lorentz_matrix @ tensor @ lorentz_matrix.T
    return transformed_tensor, lorentz_matrix, gamma

velocity = 0.5
transformed_tensor, lorentz_matrix, gamma = lorentz_transform(stress_energy_tensor, velocity)

# 5. ML Tensor Operations (reshape & transpose)
def ml_tensor_operations(tensor):
    reshaped_tensor = np.reshape(tensor, (-1, tensor.shape[-1]))
    transposed_tensor = tensor[np.newaxis, np.newaxis, :, :]  # (batch=1, channel=1, height, width)
    return reshaped_tensor, transposed_tensor

reshaped_tensor, transposed_tensor = ml_tensor_operations(energy_tensor)

# 6. Enhanced plotting functions
plt.rcParams.update({'font.size': 12, 'font.family': 'serif'})

# Main visualization function
def visualize_ligo_relativity_analysis():
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1.5, 1.5])
    
    # Plot 1: Original strain data with chirp
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time, strain * 1e21, 'k', linewidth=1, alpha=0.8)
    ax1.set_title('Simulated LIGO Strain Data with Gravitational Wave Chirp', fontsize=16)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Strain (×10$^{-21}$)')
    ax1.grid(True, alpha=0.3)
    chirp_time = 8.0
    chirp_duration = 0.5
    ax1.axvspan(chirp_time, chirp_time + chirp_duration, color='red', alpha=0.2, label='GW Chirp')
    ax1.legend()
    
    # Plot 2: Spectrogram (Energy tensor visualization)
    ax2 = fig.add_subplot(gs[1, :2])
    spec_plot = ax2.pcolormesh(times, frequencies, 10 * np.log10(Sxx), 
                              cmap='viridis', shading='gouraud')
    ax2.set_title('LIGO Spectrogram (Energy Tensor)', fontsize=16)
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlabel('Time (s)')
    cbar = plt.colorbar(spec_plot, ax=ax2)
    cbar.set_label('Power/Frequency (dB/Hz)')
    ax2.set_ylim(0, 128)  # Focus on lower frequencies where GW signal is
    
    # Plot 3: Relativistic velocity effect on gamma factor
    ax3 = fig.add_subplot(gs[1, 2])
    v_range = np.linspace(0, 0.99, 100)
    gamma_factors = [1/np.sqrt(1-(v**2)) for v in v_range]
    ax3.plot(v_range, gamma_factors, 'r-', linewidth=2)
    ax3.axvline(x=velocity, color='blue', linestyle='--', 
               label=f'Current v={velocity}c, γ={gamma:.2f}')
    ax3.set_title('Lorentz Factor (γ)', fontsize=16)
    ax3.set_xlabel('Velocity (fraction of c)')
    ax3.set_ylabel('Lorentz Factor (γ)')
    ax3.grid(True)
    ax3.legend()
    
    # Plot 4: Original vs Transformed Tensors with matrix visualization
    ax4 = fig.add_subplot(gs[2, 0])
    im1 = ax4.imshow(stress_energy_tensor, cmap='coolwarm', interpolation='none')
    ax4.set_title('Original Stress-Energy Tensor', fontsize=16)
    for (j, i), label in np.ndenumerate(stress_energy_tensor):
        ax4.text(i, j, f"{label:.2e}", ha='center', va='center', 
                color='black', fontsize=12, fontweight='bold')
    ax4.set_xticks([0, 1])
    ax4.set_yticks([0, 1])
    ax4.set_xticklabels(['t', 'x'])
    ax4.set_yticklabels(['t', 'x'])
    
    # Plot 5: Transformed tensor
    ax5 = fig.add_subplot(gs[2, 1])
    im2 = ax5.imshow(transformed_tensor, cmap='coolwarm', interpolation='none')
    ax5.set_title(f'Lorentz-Transformed Tensor (v={velocity}c)', fontsize=16)
    for (j, i), label in np.ndenumerate(transformed_tensor):
        ax5.text(i, j, f"{label:.2e}", ha='center', va='center', 
                color='black', fontsize=12, fontweight='bold')
    ax5.set_xticks([0, 1])
    ax5.set_yticks([0, 1])
    ax5.set_xticklabels(['t\'', 'x\''])
    ax5.set_yticklabels(['t\'', 'x\''])
    
    # Plot 6: Lorentz transformation matrix
    ax6 = fig.add_subplot(gs[2, 2])
    im3 = ax6.imshow(lorentz_matrix, cmap='Greens', interpolation='none')
    ax6.set_title('Lorentz Transformation Matrix', fontsize=16)
    for (j, i), label in np.ndenumerate(lorentz_matrix):
        ax6.text(i, j, f"{label:.3f}", ha='center', va='center', 
                color='black', fontsize=12, fontweight='bold')
    ax6.set_xticks([0, 1])
    ax6.set_yticks([0, 1])
    ax6.set_xticklabels(['t', 'x'])
    ax6.set_yticklabels(['t\'', 'x\''])
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.show()
    
    # Additional visualization of LIGO energy distribution
    fig2, ax = plt.subplots(figsize=(10, 8))
    
    # Log-scaled spectrogram for better contrast
    pcm = ax.pcolormesh(times, frequencies, Sxx, 
                      norm=LogNorm(vmin=Sxx.max()/1e6, vmax=Sxx.max()),
                      cmap='inferno', shading='gouraud')
    
    # Add contour to highlight energy concentrations
    contour_levels = np.logspace(np.log10(Sxx.max()/1e3), np.log10(Sxx.max()), 5)
    contour = ax.contour(times, frequencies, Sxx, levels=contour_levels, 
                        colors='cyan', alpha=0.6, linewidths=0.5)
    
    # Mark the gravitational wave chirp
    ax.axvline(x=8.0, color='white', linestyle='--', alpha=0.8, label='GW Chirp Start')
    ax.axvline(x=8.5, color='white', linestyle=':', alpha=0.8, label='GW Chirp End')
    
    ax.set_title('LIGO Energy Distribution (Log Scale)', fontsize=18)
    ax.set_ylabel('Frequency (Hz)', fontsize=14)
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylim(0, 128)  # Focus on lower frequencies
    
    cbar = fig2.colorbar(pcm, ax=ax)
    cbar.set_label('Energy Density', fontsize=14)
    
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# Run the visualization
visualize_ligo_relativity_analysis()