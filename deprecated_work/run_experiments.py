import os
import sys
import json
import datetime
import subprocess
import numpy as pd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Add the project root to sys.path so we can import algorithms
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from algorithms.mac.baseline import Config, Logger
from algorithms.mac.baseline import simulate_slotted_aloha, simulate_tdma, simulate_csma_ca
from configs import config as params

def get_git_commit_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT).decode('utf-8').strip()
    except Exception:
        return "Unknown"

def make_result_dirs(N, phy_rate_bps, QMAX, rts_cts_enabled, ack_enabled):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    trial_name = f"trial_{timestamp}"
    
    # Example format: N20_PHY2M_Q20
    phy_mbps = int(phy_rate_bps / 1e6)
    config_folder_name = f"N{N}_PHY{phy_mbps}M_Q{QMAX}"
    
    if rts_cts_enabled and ack_enabled:
        config_folder_name += "_RTSCTS_ACK_enabled"
    elif ack_enabled:
        config_folder_name += "_ACK_enabled"
    
    base_dir = os.path.join(project_root, "results", config_folder_name, trial_name)
    
    os.makedirs(os.path.join(base_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "csv"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "logs"), exist_ok=True)
    
    return base_dir, trial_name

def run_experiment():
    N = params.N
    sim_time_s = params.SIM_TIME_S
    slot_time_s = params.SLOT_TIME_S
    phy_rate_bps = params.PHY_RATE_BPS
    payload_bytes = params.PAYLOAD_BYTES
    QMAX = params.QMAX
    
    sweep_min_pps = params.SWEEP_MIN_PPS
    sweep_max_pps = params.SWEEP_MAX_PPS
    sweep_steps = params.SWEEP_STEPS
    
    # Advanced / Generic Tunable Parameters
    cw_min = params.CW_MIN
    cw_max = params.CW_MAX
    difs_slots = params.DIFS_SLOTS
    sifs_slots = params.SIFS_SLOTS
    ack_slots = params.ACK_SLOTS
    ack_timeout_slots = params.ACK_TIMEOUT_SLOTS
    max_retry = params.MAX_RETRY
    rts_cts_enabled = params.RTS_CTS_ENABLED
    ack_enabled = params.ACK_ENABLED
    rts_slots = params.RTS_SLOTS
    cts_slots = params.CTS_SLOTS
    
    log_interval_slots = params.LOG_INTERVAL_SLOTS

    seed = params.SEED
    
    out_dir, trial_name = make_result_dirs(N, phy_rate_bps, QMAX, rts_cts_enabled, ack_enabled)
    
    log_file_path = os.path.join(out_dir, "logs", "run.log")
    
    # Save metadata
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "git_commit": get_git_commit_hash(),
        "script": os.path.basename(__file__),
        "seed_used": seed,
        "output_directory": out_dir,
        "parameters": {k: v for k, v in vars(params).items() if not k.startswith('__')}
    }
    
    with open(os.path.join(out_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    def log_print(msg):
        print(msg)
        with open(log_file_path, "a") as lf:
            lf.write(msg + "\n")

    log_print(f"Starting Experiment targeting logic from config.py")
    log_print(f"Directory: {out_dir}")
    log_print(f"Parameters: N={N}, PHY={phy_rate_bps/1e6}Mbps, Pkt={payload_bytes}B, QMAX={QMAX}")
    log_print(f"RTS/CTS Enabled: {rts_cts_enabled}, ACK Enabled: {ack_enabled}")
    
    cfg = Config(N=N, sim_time_s=sim_time_s, slot_time_s=slot_time_s, 
                 phy_rate_bps=phy_rate_bps, payload_bytes=payload_bytes, QMAX=QMAX,
                 cw_min=cw_min, cw_max=cw_max, difs_slots=difs_slots, sifs_slots=sifs_slots,
                 ack_slots=ack_slots, ack_timeout_slots=ack_timeout_slots, max_retry=max_retry,
                 log_interval_slots=log_interval_slots, rts_cts_enabled=rts_cts_enabled,
                 ack_enabled=ack_enabled, rts_slots=rts_slots, cts_slots=cts_slots)

    traffic_pps_list = np.linspace(sweep_min_pps, sweep_max_pps, sweep_steps).astype(int)
    
    results = []
    
    q_log_aloha = []
    q_log_tdma = []
    q_log_csma = []
    
    for idx, pps in enumerate(traffic_pps_list):
        log_print(f"Testing load {pps} pps ({idx+1}/{len(traffic_pps_list)})")
        
        log_aloha = Logger(load_pps=pps)
        cfg.seed = seed + idx
        simulate_slotted_aloha(cfg, pps, log_aloha)
        q_log_aloha.extend(log_aloha.q_log)
        
        log_tdma = Logger(load_pps=pps)
        cfg.seed = seed + idx
        simulate_tdma(cfg, pps, log_tdma)
        q_log_tdma.extend(log_tdma.q_log)
        
        log_csma = Logger(load_pps=pps)
        cfg.seed = seed + idx
        simulate_csma_ca(cfg, pps, log_csma)
        q_log_csma.extend(log_csma.q_log)
        
        results.append({
            'Offered_Load_pps': pps,
            
            'ALOHA_Throughput_Mbps': log_aloha.get_throughput_bps(sim_time_s) / 1e6,
            'TDMA_Throughput_Mbps': log_tdma.get_throughput_bps(sim_time_s) / 1e6,
            'CSMA_Throughput_Mbps': log_csma.get_throughput_bps(sim_time_s) / 1e6,
            
            'ALOHA_Drops': log_aloha.pkts_dropped_qfull + log_aloha.pkts_dropped_mac,
            'TDMA_Drops': log_tdma.pkts_dropped_qfull + log_tdma.pkts_dropped_mac,
            'CSMA_Drops': log_csma.pkts_dropped_qfull + log_csma.pkts_dropped_mac,
            
            'ALOHA_Q_len': log_aloha.get_avg_q_len(N),
            'TDMA_Q_len': log_tdma.get_avg_q_len(N),
            'CSMA_Q_len': log_csma.get_avg_q_len(N),
            
            'ALOHA_Delay_s': log_aloha.get_avg_end_to_end_delay_s(),
            'TDMA_Delay_s': log_tdma.get_avg_end_to_end_delay_s(),
            'CSMA_Delay_s': log_csma.get_avg_end_to_end_delay_s(),
            
            'ALOHA_Util': log_aloha.get_channel_utilization(),
            'TDMA_Util': log_tdma.get_channel_utilization(),
            'CSMA_Util': log_csma.get_channel_utilization(),
            
            'ALOHA_Collisions': log_aloha.collision_events,
            'TDMA_Collisions': log_tdma.collision_events,
            'CSMA_Collisions': log_csma.collision_events,
        })
        
    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, "csv", "summary_metrics.csv")
    df.to_csv(csv_path, index=False)
    log_print(f"Saved metrics to {csv_path}")
    
    # Save End-to-End Delay standalone CSV
    delay_df = pd.DataFrame()
    delay_rows = []
    for idx, row in df.iterrows():
        delay_rows.append({"traffic_rate": row["Offered_Load_pps"], "traffic_rate_unit": "packets/sec", "end_to_end_delay": row["ALOHA_Delay_s"], "delay_unit": "seconds", "protocol": "ALOHA", "N": N, "trial_id": trial_name})
        delay_rows.append({"traffic_rate": row["Offered_Load_pps"], "traffic_rate_unit": "packets/sec", "end_to_end_delay": row["TDMA_Delay_s"], "delay_unit": "seconds", "protocol": "TDMA", "N": N, "trial_id": trial_name})
        delay_rows.append({"traffic_rate": row["Offered_Load_pps"], "traffic_rate_unit": "packets/sec", "end_to_end_delay": row["CSMA_Delay_s"], "delay_unit": "seconds", "protocol": "CSMA_CA", "N": N, "trial_id": trial_name})
    pd.DataFrame(delay_rows).to_csv(os.path.join(out_dir, "csv", "end_to_end_delay_vs_traffic_rate.csv"), index=False)
    log_print(f"Saved delay data to csv/end_to_end_delay_vs_traffic_rate.csv")
    
    # Save Time-series Queuing Files
    pd.DataFrame(q_log_aloha).to_csv(os.path.join(out_dir, "csv", "buffer_usage_ALOHA.csv"), index=False)
    pd.DataFrame(q_log_tdma).to_csv(os.path.join(out_dir, "csv", "buffer_usage_TDMA.csv"), index=False)
    pd.DataFrame(q_log_csma).to_csv(os.path.join(out_dir, "csv", "buffer_usage_CSMA_CA.csv"), index=False)
    log_print(f"Saved detailed buffer time-series to csv/buffer_usage_PROTOCOL.csv")
    
    # Save ACK Tracking Log
    ack_logs = log_aloha.ack_log + log_tdma.ack_log + log_csma.ack_log
    if ack_logs:
        pd.DataFrame(ack_logs).to_csv(os.path.join(out_dir, "csv", "ack_success_log.csv"), index=False)
        log_print(f"Saved ACK tracking data to csv/ack_success_log.csv")
    
    # Generate Plots
    
    # 1. Throughput
    plt.figure(figsize=(10,6))
    plt.plot(df['Offered_Load_pps'], df['ALOHA_Throughput_Mbps'], label='Slotted ALOHA', marker='o')
    plt.plot(df['Offered_Load_pps'], df['TDMA_Throughput_Mbps'], label='TDMA', marker='s')
    plt.plot(df['Offered_Load_pps'], df['CSMA_Throughput_Mbps'], label='CSMA/CA', marker='^')
    plt.xlabel('Offered Traffic Rate (packets/sec total)')
    plt.ylabel('Throughput (Mbps)')
    plt.title(f'Protocol Comparison: Throughput\n(N={N}, Pkt={payload_bytes}B, PHY={phy_rate_bps/1e6}Mbps)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "images", "throughput_vs_load.png"))
    plt.close()
    
    # 2. Utilization
    plt.figure(figsize=(10,6))
    plt.plot(df['Offered_Load_pps'], df['ALOHA_Util'], label='Slotted ALOHA', marker='o')
    plt.plot(df['Offered_Load_pps'], df['TDMA_Util'], label='TDMA', marker='s')
    plt.plot(df['Offered_Load_pps'], df['CSMA_Util'], label='CSMA/CA', marker='^')
    plt.xlabel('Offered Traffic Rate (packets/sec total)')
    plt.ylabel('Channel Utilization')
    plt.title(f'Protocol Comparison: Channel Utilization (N={N})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "images", "utilization_vs_load.png"))
    plt.close()
    
    # 3. Queues
    plt.figure(figsize=(10,6))
    plt.plot(df['Offered_Load_pps'], df['TDMA_Q_len'], label='TDMA', marker='s')
    plt.plot(df['Offered_Load_pps'], df['CSMA_Q_len'], label='CSMA/CA', marker='^')
    plt.xlabel('Offered Traffic Rate (packets/sec total)')
    plt.ylabel('Average Queue Length (packets)')
    plt.title(f'Protocol Comparison: Queue Occupancy\n(N={N}, Buffer Size={QMAX})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "images", "queues_vs_load.png"))
    plt.close()
    
    # 4. Drops
    plt.figure(figsize=(10,6))
    plt.plot(df['Offered_Load_pps'], df['ALOHA_Drops'], label='Slotted ALOHA', marker='o')
    plt.plot(df['Offered_Load_pps'], df['TDMA_Drops'], label='TDMA', marker='s')
    plt.plot(df['Offered_Load_pps'], df['CSMA_Drops'], label='CSMA/CA', marker='^')
    plt.xlabel('Offered Traffic Rate (packets/sec total)')
    plt.ylabel('Dropped Packets (Buffer + Retries)')
    plt.title(f'Protocol Comparison: Packet Drops (N={N})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "images", "drops_vs_load.png"))
    plt.close()
    
    # 5. End-To-End Delay
    plt.figure(figsize=(10,6))
    plt.plot(df['Offered_Load_pps'], df['ALOHA_Delay_s'], label='Slotted ALOHA', marker='o')
    plt.plot(df['Offered_Load_pps'], df['TDMA_Delay_s'], label='TDMA', marker='s')
    plt.plot(df['Offered_Load_pps'], df['CSMA_Delay_s'], label='CSMA/CA', marker='^')
    plt.xlabel('Offered Traffic Rate (packets/sec total)')
    plt.ylabel('End-to-End Delay (seconds)')
    rts_tag = "ON" if rts_cts_enabled else "OFF"
    ack_tag = "ON" if ack_enabled else "OFF"
    plt.title(f'Protocol Comparison: End-to-End Delay vs Traffic Rate\n(N={N}, QMAX={QMAX}, RTS/CTS={rts_tag}, ACK={ack_tag})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "images", "end_to_end_delay_vs_traffic_rate.png"))
    plt.close()

    # ====================================================
    # Mobility Simulation (Phase-1: independent trace)
    # ====================================================
    if getattr(params, "ENABLE_MOBILITY", False):
        log_print("Running 3D Mobility Simulation...")
        try:
            from algorithms.mobility.manager import MobilityManager
            mob_mgr = MobilityManager(
                n_nodes=N,
                bounds=(getattr(params, 'AREA_X', 1000), getattr(params, 'AREA_Y', 1000), getattr(params, 'AREA_Z', 300)),
                sink_pos=(getattr(params, 'SINK_X', 500), getattr(params, 'SINK_Y', 500), getattr(params, 'SINK_Z', 0)),
                mobility_model=getattr(params, 'MOBILITY_MODEL', 'gauss_markov'),
                speed_mode=getattr(params, 'SPEED_MODE', 'uniform'),
                v_min=getattr(params, 'V_MIN', 5.0),
                v_max=getattr(params, 'V_MAX', 30.0),
                v_mean=getattr(params, 'V_MEAN', None),
                v_std=getattr(params, 'V_STD', None),
                speed_update_interval=getattr(params, 'SPEED_UPDATE_INTERVAL', 5.0),
                comm_range=getattr(params, 'COMM_RANGE_R', 500.0),
                enable_pathloss=getattr(params, 'ENABLE_PATHLOSS', False),
                pathloss_k=getattr(params, 'PATHLOSS_K', 0.001),
                pathloss_eta=getattr(params, 'PATHLOSS_ETA', 2.0),
                enable_prop_delay=getattr(params, 'ENABLE_PROP_DELAY', False),
                dt=getattr(params, 'MOBILITY_DT', 0.1),
                seed=seed,
                gm_alpha=getattr(params, 'GM_ALPHA', 0.5),
                rwp_pause_time=getattr(params, 'RWP_PAUSE_TIME', 1.0),
                circ_radius=getattr(params, 'CIRC_RADIUS', 100.0),
                circ_omega_mean=getattr(params, 'CIRC_OMEGA_MEAN', 0.1),
                circ_omega_std=getattr(params, 'CIRC_OMEGA_STD', 0.02),
                circ_climb_rate=getattr(params, 'CIRC_CLIMB_RATE', 0.5),
            )
            mob_mgr.run(sim_time_s)
            mob_mgr.export_csv(out_dir)
            log_print(f"Saved mobility CSVs to csv/mobility_positions.csv and csv/uav_sink_distance.csv")
            mob_mgr.generate_plots(out_dir, top_k=min(5, N))
            log_print(f"Saved mobility plots to images/")
        except Exception as e:
            log_print(f"ERROR in Mobility Simulation: {e}")
            import traceback
            log_print(traceback.format_exc())

    # Run RL Post-Processing Step
    if getattr(params, "ENABLE_RL_SELECTOR", False):
        log_print("Running RL Post-Processing step...")
        try:
            from algorithms.rl.qlearning_selector import run_rl_postprocess
            run_rl_postprocess(out_dir, df, params, log_print)
        except Exception as e:
            log_print(f"ERROR in RL Post-Processing: {e}")

    log_print(f"Experiment completed successfully. Outputs saved to {out_dir}")

if __name__ == '__main__':
    run_experiment()
