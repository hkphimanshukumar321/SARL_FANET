import numpy as np
import math
import scipy.special as sp
import pandas as pd
import matplotlib.pyplot as plt
import os

class BERCalculator:
    """Computes theoretical Bit Error Rate (BER) for various modulation schemes over AWGN."""
    
    MODULATIONS = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "256QAM"]
    
    def __init__(self, modulation="BPSK"):
        if modulation not in self.MODULATIONS:
            raise ValueError(f"Modulation must be one of {self.MODULATIONS}")
        self.modulation = modulation
        
    def get_bits_per_symbol(self):
        if self.modulation == "BPSK": return 1
        if self.modulation == "QPSK": return 2
        if self.modulation == "8PSK": return 3
        if self.modulation == "16QAM": return 4
        if self.modulation == "64QAM": return 6
        if self.modulation == "256QAM": return 8
        return 1
        
    def compute_ber(self, snr_linear_array):
        """
        Computes the theoretical BER for the given linear SNR array (Eb/No).
        Note: snr_linear here is defined as Es/N0 (Signal to Noise Ratio per symbol).
        We convert it to Eb/N0 (Energy per bit to Noise density) internally.
        """
        snr = np.maximum(snr_linear_array, 1e-10) # Avoid division by zero/log issues
        k = self.get_bits_per_symbol()
        ebno = snr / k  # Convert Es/N0 to Eb/N0
        
        # Q-function: Q(x) = 0.5 * erfc(x / sqrt(2))
        def Q(x):
            return 0.5 * sp.erfc(x / np.sqrt(2))
            
        if self.modulation == "BPSK":
            return Q(np.sqrt(2 * ebno))
            
        elif self.modulation == "QPSK":
            # Same as BPSK for Gray-coded QPSK
            return Q(np.sqrt(2 * ebno))
            
        elif self.modulation == "8PSK":
            # Approximation for high SNR
            return 2/3 * Q(np.sqrt(2 * ebno * 3) * np.sin(np.pi/8))
            
        elif self.modulation in ["16QAM", "64QAM", "256QAM"]:
            M = 2**k
            # Symbol error probability for square M-QAM
            P_sq = 2 * (1 - 1/np.sqrt(M)) * Q(np.sqrt(3 * snr / (M - 1)))
            P_s = 1 - (1 - P_sq)**2
            # BER approximation (Gray coding)
            return P_s / k
            
        return np.ones_like(snr)

class FadingChannel:
    """Base class for fading channel models."""
    def __init__(self, name="AWGN"):
        self.name = name

    def sample_gain(self, size, rng):
        """Returns linear power gain coefficients |h|^2 for 'size' links."""
        return np.ones(size) # Default no fading (AWGN)

class AWGNChannel(FadingChannel):
    def __init__(self):
        super().__init__("AWGN")

class RayleighChannel(FadingChannel):
    def __init__(self):
        super().__init__("Rayleigh")
        
    def sample_gain(self, size, rng):
        """Rayleigh fading: h ~ CN(0,1). |h| is Rayleigh, |h|^2 is Exponential(1)."""
        return rng.exponential(scale=1.0, size=size)

class RicianChannel(FadingChannel):
    def __init__(self, K=3.0):
        super().__init__(f"Rician (K={K}dB)")
        self.K_linear = 10 ** (K / 10.0)
        
    def sample_gain(self, size, rng):
        """Rician fading using non-central chi-square distribution for power gain."""
        # Mean line-of-sight (LOS) component power
        mean_power_los = self.K_linear / (self.K_linear + 1)
        # Mean scattered component power
        mean_power_scat = 1 / (self.K_linear + 1)
        
        # h = sqrt(K/(K+1)) * e^{j*theta} + sqrt(1/(K+1)) * (x + jy)/sqrt(2)
        X = rng.normal(np.sqrt(mean_power_los), np.sqrt(mean_power_scat / 2), size)
        Y = rng.normal(0, np.sqrt(mean_power_scat / 2), size)
        
        return X**2 + Y**2

class NakagamiChannel(FadingChannel):
    def __init__(self, m=2.0, omega=1.0):
        super().__init__(f"Nakagami-m (m={m})")
        self.m = m
        self.omega = omega
        
    def sample_gain(self, size, rng):
        """Nakagami-m fading: |h| is Nakagami, |h|^2 is Gamma(m, omega/m)."""
        return rng.gamma(shape=self.m, scale=self.omega/self.m, size=size)


def generate_ber_snr_lut(snr_db_range, modulation="BPSK", save_path="ber_snr_lut.csv"):
    """Generates a CSV Lookup Table for BER vs SNR for a specific modulation under AWGN.
       Note: The instantaneous SNR after fading uses this AWGN curve."""
    snr_linear = 10 ** (snr_db_range / 10.0)
    calc = BERCalculator(modulation)
    ber = calc.compute_ber(snr_linear)
    
    df = pd.DataFrame({
        "SNR_dB": snr_db_range,
        f"BER_{modulation}": ber
    })
    
    df.to_csv(save_path, index=False)
    return df

def plot_ber_vs_snr(save_path, modulation="BPSK", snr_min_db=0, snr_max_db=30, points=100):
    """
    Simulates and plots the average BER vs average SNR for AWGN, Rayleigh, Rician, and Nakagami.
    This effectively numerically integrates the AWGN BER curve over the fading PDFs.
    """
    snr_db = np.linspace(snr_min_db, snr_max_db, points)
    snr_linear = 10 ** (snr_db / 10.0)
    
    channels = [
        AWGNChannel(),
        RayleighChannel(),
        RicianChannel(K=3.0),
        NakagamiChannel(m=2.0)
    ]
    
    calc = BERCalculator(modulation)
    rng = np.random.default_rng(42)
    n_samples = 100000  # Monte Carlo samples to average out the fading distribution
    
    plt.style.use('seaborn-v0_8-whitegrid') if 'seaborn-v0_8-whitegrid' in plt.style.available else plt.style.use('seaborn-whitegrid') if 'seaborn-whitegrid' in plt.style.available else None
    plt.figure(figsize=(10, 7), dpi=150)
    
    markers = ['o', 's', '^', 'D']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    
    for idx, ch in enumerate(channels):
        avg_ber = np.zeros_like(snr_linear)
        for i, snr_val in enumerate(snr_linear):
            # Sample fading gains
            gains = ch.sample_gain(n_samples, rng)
            # Instantaneous SNRs
            inst_snr_linear = snr_val * gains
            # Instantaneous BERs
            inst_ber = calc.compute_ber(inst_snr_linear)
            # Average BER for this average SNR
            avg_ber[i] = np.mean(inst_ber)
            
        plt.semilogy(snr_db, avg_ber, label=ch.name, linewidth=2.5, marker=markers[idx], color=colors[idx], markersize=6, markevery=5)
        
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.ylim([1e-5, 1])
    plt.xlim([snr_min_db, snr_max_db])
    plt.xlabel('Average SNR (dB)', fontsize=12, fontweight='bold')
    plt.ylabel('Bit Error Rate (BER)', fontsize=12, fontweight='bold')
    plt.title(f'BER vs SNR under Various Fading Channels ({modulation})', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=11, frameon=True, shadow=True, borderpad=1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
