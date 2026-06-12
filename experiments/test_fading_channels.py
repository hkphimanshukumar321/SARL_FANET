import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from algorithms.mac.baseline import Config, Logger
from algorithms.mac.channel_aware_mac import simulate_tdma_aware, simulate_csma_aware
from algorithms.mobility.link import compute_fading_success_prob
from algorithms.channel.fading import BERCalculator, AWGNChannel, RayleighChannel, RicianChannel, NakagamiChannel
from configs import config as global_cfg

def run_fading_experiment():
    print("=" * 60)
    print(" FADING CHANNEL PERFORMANCE EXPERIMENT ")
    print("=" * 60)
    
    # Configuration
    N = 20
    sim_time_s = 5.0
    payload_bits = 1500 * 8
    traffic_pps_list = np.linspace(100, 1000, 10).astype(int)
    
    # Distance configuration for test (fixed distance to isolate fading effect)
    distances = np.full(N, 200.0) # 200 meters
    link_up_schedule = np.ones((N, 1), dtype=int) # Always up
    
    channels = [
        AWGNChannel(),
        RayleighChannel(),
        RicianChannel(K=3.0),
        NakagamiChannel(m=2.0)
    ]
    
    ber_calc = BERCalculator(global_cfg.MODULATION)
    rng = np.random.default_rng(42)
    
    output_dir = os.path.join(project_root, "results", "fading_test")
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for ch in channels:
        print(f"\nEvaluating Channel: {ch.name}")
        
        # Precompute the success probability for this distance + channel combo
        # Usually this happens dynamically per step in the env, but we fix it here for a clean MAC sweep
        p_succ_sample = compute_fading_success_prob(
            distances, 
            ch, 
            ber_calc, 
            global_cfg.TX_POWER_DBM, 
            global_cfg.NOISE_POWER_DBM, 
            payload_bits, 
            rng
        )
        avg_succ_prob = np.mean(p_succ_sample)
        print(f"  -> Average Link Success Probability: {avg_succ_prob:.4f}")
        
        # Re-shape for the channel-aware MAC simulator (expects N x T matrix)
        sp_sched = np.tile(p_succ_sample[:, np.newaxis], (1, 1))
        
        for pps in traffic_pps_list:
            
            cfg = Config(N=N, sim_time_s=sim_time_s)
            
            # --- TDMA ---
            cfg.seed = 42 + pps
            log_tdma = Logger(load_pps=pps, protocol_name="TDMA")
            simulate_tdma_aware(cfg, pps, log_tdma, link_up_schedule, sp_sched, mobility_dt=sim_time_s)
            
            # --- CSMA ---
            cfg.seed = 42 + pps
            log_csma = Logger(load_pps=pps, protocol_name="CSMA")
            simulate_csma_aware(cfg, pps, log_csma, link_up_schedule, sp_sched, mobility_dt=sim_time_s)
            
            results.append({
                'Channel': ch.name,
                'Offered_Load_pps': pps,
                'Avg_Succ_Prob': avg_succ_prob,
                'TDMA_Throughput_Mbps': log_tdma.get_throughput_bps(sim_time_s) / 1e6,
                'CSMA_Throughput_Mbps': log_csma.get_throughput_bps(sim_time_s) / 1e6,
                'TDMA_Delay_ms': log_tdma.get_avg_end_to_end_delay_s() * 1000,
                'CSMA_Delay_ms': log_csma.get_avg_end_to_end_delay_s() * 1000,
            })
            
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "fading_sweep_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # ---------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------
    markers = ['o', 's', '^', 'D']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    
    # Plot 1: TDMA Throughput
    plt.figure(figsize=(10, 6))
    for i, ch in enumerate(channels):
        subset = df[df['Channel'] == ch.name]
        plt.plot(subset['Offered_Load_pps'], subset['TDMA_Throughput_Mbps'], 
                 label=ch.name, marker=markers[i], color=colors[i], linestyle='-')
    plt.xlabel('Offered Traffic Rate (PPS)')
    plt.ylabel('TDMA Throughput (Mbps)')
    plt.title('TDMA Throughput vs Traffic Load under Various Fading Channels')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tdma_fading_throughput.png"))
    plt.close()
    
    # Plot 2: CSMA Throughput
    plt.figure(figsize=(10, 6))
    for i, ch in enumerate(channels):
        subset = df[df['Channel'] == ch.name]
        plt.plot(subset['Offered_Load_pps'], subset['CSMA_Throughput_Mbps'], 
                 label=ch.name, marker=markers[i], color=colors[i], linestyle='--')
    plt.xlabel('Offered Traffic Rate (PPS)')
    plt.ylabel('CSMA/CA Throughput (Mbps)')
    plt.title('CSMA/CA Throughput vs Traffic Load under Various Fading Channels')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "csma_fading_throughput.png"))
    plt.close()
    
    # Plot 3: Delay
    plt.figure(figsize=(10, 6))
    for i, ch in enumerate(channels):
        subset = df[df['Channel'] == ch.name]
        plt.plot(subset['Offered_Load_pps'], subset['TDMA_Delay_ms'], 
                 label=f"TDMA ({ch.name})", marker=markers[i], color=colors[i], linestyle='-')
        plt.plot(subset['Offered_Load_pps'], subset['CSMA_Delay_ms'], 
                 label=f"CSMA ({ch.name})", marker=markers[i], color=colors[i], linestyle='--')
    plt.xlabel('Offered Traffic Rate (PPS)')
    plt.ylabel('End-to-End Delay (ms)')
    plt.title('End-to-End Delay vs Traffic Load under Various Fading Channels')
    plt.grid(True)
    # Put legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fading_delay_combined.png"))
    plt.close()
    
    print(f"Plots saved to {output_dir}")
    
    # Verdict Check
    awgn_final = df[df['Channel'] == 'AWGN']['TDMA_Throughput_Mbps'].iloc[-1]
    rayleigh_final = df[df['Channel'] == 'Rayleigh']['TDMA_Throughput_Mbps'].iloc[-1]
    
    print("\n" + "=" * 60)
    print(" VERDICT ")
    if rayleigh_final < awgn_final:
        print(" PASS: Rayleigh fading correctly severely degrading throughput compared to AWGN.")
    else:
        print(" FAIL: Fading models not impacting throughput as expected.")
    print("=" * 60)

if __name__ == "__main__":
    run_fading_experiment()
