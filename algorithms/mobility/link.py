# link.py — Link Model for Many-to-One UAV-to-Sink Communication
# Phase-1: Range-gated binary link
# Phase-2 (optional): Path-loss-based success probability
# Optional: Propagation delay

import numpy as np

SPEED_OF_LIGHT = 3e8  # m/s


def compute_distances(positions, sink_pos):
    """
    Compute Euclidean distance from each UAV to the sink.
    Args:
        positions: (N, 3) array of UAV positions.
        sink_pos:  (3,) array of sink position.
    Returns:
        (N,) array of distances in meters.
    """
    return np.linalg.norm(positions - sink_pos[np.newaxis, :], axis=1)


def compute_link_up(distances, comm_range):
    """
    Phase-1 range-gated binary link.
    Returns (N,) int array: 1 if distance <= comm_range, else 0.
    """
    return (distances <= comm_range).astype(int)


def compute_pathloss_success_prob(distances, k=0.001, eta=2.0):
    """
    Phase-2 distance-based success probability.
    P_succ(d) = exp(-k * d^eta)
    Returns (N,) array of probabilities in [0, 1].
    """
    return np.exp(-k * np.power(distances, eta))


def compute_pathloss_db(distances, eta=2.0, d0=1.0, pl0=46.4):
    """
    Log-distance path loss in dB.
    PL(d) = PL0 + 10*eta*log10(d/d0)
    Args:
        distances: (N,) array in meters.
        eta: path loss exponent (free-space=2, urban=3-5).
        d0: reference distance in meters.
        pl0: path loss at d0 in dB.
    Returns:
        (N,) array of path loss values in dB.
    """
    d_safe = np.maximum(distances, d0)  # avoid log(0)
    return pl0 + 10.0 * eta * np.log10(d_safe / d0)


def compute_fading_success_prob(distances, fading_channel, ber_calc, tx_power_dbm, noise_power_dbm, payload_bits, rng):
    """
    Computes packet success probability combining path-loss, fading, and modulation BER.
    """
    # 1. Path loss (dB)
    pl_db = compute_pathloss_db(distances)
    
    # 2. Fading gain (linear)
    gains = fading_channel.sample_gain(distances.shape[0], rng)
    
    # 3. Rx SNR (dB)
    # SNR = P_tx - PL + Gain_dB - Noise
    # Or in linear: SNR = (P_tx / Noise) * (Gain / PL)
    # Using dB math:
    gain_db = 10.0 * np.log10(np.maximum(gains, 1e-10))
    snr_db = tx_power_dbm - pl_db + gain_db - noise_power_dbm
    
    snr_linear = 10.0 ** (snr_db / 10.0)
    
    # 4. Compute BER for instantaneous SNR
    ber = ber_calc.compute_ber(snr_linear)
    
    # 5. Packet success probability: P_succ = (1 - BER)^payload_bits
    p_succ = np.power(1.0 - ber, payload_bits)
    
    return p_succ



def compute_propagation_delay(distances):
    """
    Optional propagation delay: tau = d / c
    Returns (N,) array of propagation delays in seconds.
    """
    return distances / SPEED_OF_LIGHT
