import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import sys

# Allow direct execution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class Config:
    def __init__(self, N=20, sim_time_s=10.0, slot_time_s=9e-6, phy_rate_bps=2e6, payload_bytes=1500, QMAX=20, seed=42,
                 cw_min=15, cw_max=1023, difs_slots=4, sifs_slots=2, ack_slots=2, ack_timeout_slots=10, max_retry=7,
                 log_interval_slots=1000, rts_cts_enabled=False, ack_enabled=False, rts_slots=3, cts_slots=3,
                 tdma_guard_time_s=1e-6):
        self.N = N
        self.sim_time_s = sim_time_s
        self.slot_time_s = slot_time_s
        self.phy_rate_bps = phy_rate_bps
        self.payload_bytes = payload_bytes
        self.QMAX = QMAX
        self.seed = seed
        
        # CSMA tunable parameters
        self.cw_min = cw_min
        self.cw_max = cw_max
        self.difs_slots = difs_slots
        self.sifs_slots = sifs_slots
        self.ack_slots = ack_slots
        self.ack_timeout_slots = ack_timeout_slots
        self.max_retry = max_retry
        
        self.rts_cts_enabled = rts_cts_enabled
        self.ack_enabled = ack_enabled
        self.rts_slots = rts_slots
        self.cts_slots = cts_slots
        
        self.log_interval_slots = log_interval_slots
        
        self.payload_bits = self.payload_bytes * 8
        self.tx_time_s = self.payload_bits / float(self.phy_rate_bps)
        self.TX_slots = int(math.ceil(self.tx_time_s / self.slot_time_s))
        
        self.tdma_guard_time_s = tdma_guard_time_s
        self.TDMA_GUARD_SLOTS = int(math.ceil(self.tdma_guard_time_s / self.slot_time_s))
        self.TDMA_SLOT_TOTAL_SLOTS = self.TX_slots + self.TDMA_GUARD_SLOTS
        
        self.total_slots = int(math.ceil(self.sim_time_s / self.slot_time_s))

class Logger:
    def __init__(self, load_pps=0, protocol_name="Unknown"):
        self.pkts_generated = 0
        self.pkts_dropped_qfull = 0
        self.pkts_dropped_mac = 0
        self.pkts_success = 0
        self.payload_bits_tx = 0
        self.collision_events = 0
        self.tx_attempts = 0
        self.channel_busy_slots = 0
        self.sum_q_len = 0
        self.total_slots = 0
        
        self.sum_end_to_end_delay_s = 0.0
        
        self.load_pps = load_pps
        self.protocol_name = protocol_name
        self.ts_enq = 0
        self.ts_deq = 0
        self.ts_drops = 0
        self.q_log = []
        self.ack_log = []
        self.comm_log = []  # Per-link communication events: [{slot, sender, receiver, status, snr_db, ber}]

    def record_queue_time_series(self, slot_idx, max_cap, current_used):
        self.q_log.append({
            "slot_index": slot_idx,
            "q_max": max_cap,
            "q_allotted": max_cap,  # Hardware max bounds natively match allotted here
            "q_used": current_used,
            "q_wasted": max_cap - current_used,
            "enqueues": self.ts_enq,
            "dequeues": self.ts_deq,
            "drops": self.ts_drops
        })
        self.ts_enq = 0
        self.ts_deq = 0
        self.ts_drops = 0

    def record_ack(self, slot_idx, sender_id, receiver_id, pkt_id, status="success", retry_num=0):
        self.ack_log.append({
            "timestamp_slot": slot_idx,
            "protocol": self.protocol_name,
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "packet_id": pkt_id,
            "status": status,
            "retry_count": retry_num
        })

    def get_throughput_bps(self, sim_time_s):
        return self.payload_bits_tx / float(sim_time_s)

    def get_avg_q_len(self, N):
        if self.total_slots == 0:
            return 0
        return self.sum_q_len / float(self.total_slots * N)

    def get_channel_utilization(self):
        if self.total_slots == 0:
            return 0
        return self.channel_busy_slots / float(self.total_slots)


    def get_avg_end_to_end_delay_s(self):
        if self.pkts_success == 0:
            return 0.0
        return self.sum_end_to_end_delay_s / float(self.pkts_success)

def simulate_slotted_aloha(cfg: Config, offered_total_pps: float, log: Logger):
    rng = np.random.default_rng(cfg.seed)
    # prevent div by zero if lam_node=0
    lam_node = offered_total_pps / float(cfg.N)
    if lam_node <= 0: return

    q = [[] for _ in range(cfg.N)]  # List of lists to store arrival timestamps
    current_q_sum = 0
    channel_busy = 0
    log.total_slots = cfg.total_slots
    
    next_arrival = rng.exponential(1.0 / lam_node, size=cfg.N)
    next_global_arrival = np.min(next_arrival)
    
    max_cap = cfg.N * cfg.QMAX
    
    for s in range(cfg.total_slots):
        if s > 0 and s % cfg.log_interval_slots == 0:
            log.record_queue_time_series(s, max_cap, current_q_sum)
            
        log.sum_q_len += current_q_sum
        
        # 1. Faster Arrivals
        t = s * cfg.slot_time_s
        if t >= next_global_arrival:
            arriving_nodes = np.where(t >= next_arrival)[0]
            for i in arriving_nodes:
                log.pkts_generated += 1
                if len(q[i]) < cfg.QMAX:
                    q[i].append(t)  # Record arrival timestamp
                    current_q_sum += 1
                    log.ts_enq += 1
                else:
                    log.pkts_dropped_qfull += 1
                    log.ts_drops += 1
                next_arrival[i] += rng.exponential(1.0 / lam_node)
            next_global_arrival = np.min(next_arrival)
        
        # 2. Channel check
        if channel_busy > 0:
            channel_busy -= 1
            log.channel_busy_slots += 1
            continue
            
        # 3. Transmission attempt
        if current_q_sum > 0:
            tx_nodes = [i for i in range(cfg.N) if len(q[i]) > 0]
            if len(tx_nodes) == 0:
                continue
                
            log.tx_attempts += len(tx_nodes)
            channel_busy = cfg.TX_slots - 1
            log.channel_busy_slots += 1
            
            if len(tx_nodes) == 1:
                i = tx_nodes[0]
                arrivalTime = q[i].pop(0)
                
                log.pkts_success += 1
                log.payload_bits_tx += cfg.payload_bits
                log.sum_end_to_end_delay_s += ((s + cfg.TX_slots) * cfg.slot_time_s) - arrivalTime
                
                current_q_sum -= 1
                log.ts_deq += 1
            else:
                log.collision_events += 1
                for i in tx_nodes:
                    if len(q[i]) > 0:
                        q[i].pop(0)
                        current_q_sum -= 1
                        log.pkts_dropped_mac += 1
                        log.ts_drops += 1
                    
def simulate_tdma(cfg: Config, offered_total_pps: float, log: Logger):
    rng = np.random.default_rng(cfg.seed)
    lam_node = offered_total_pps / float(cfg.N)
    if lam_node <= 0: return
    
    q = [[] for _ in range(cfg.N)]
    current_q_sum = 0
    channel_busy = 0
    log.total_slots = cfg.total_slots
    
    next_arrival = rng.exponential(1.0 / lam_node, size=cfg.N)
    next_global_arrival = np.min(next_arrival)
    current_tdma_slot_owner = 0
    
    max_cap = cfg.N * cfg.QMAX
    
    for s in range(cfg.total_slots):
        if s > 0 and s % cfg.log_interval_slots == 0:
            log.record_queue_time_series(s, max_cap, current_q_sum)
            
        log.sum_q_len += current_q_sum
        
        t = s * cfg.slot_time_s
        if t >= next_global_arrival:
            arriving_nodes = np.where(t >= next_arrival)[0]
            for i in arriving_nodes:
                log.pkts_generated += 1
                if len(q[i]) < cfg.QMAX:
                    q[i].append(t)
                    current_q_sum += 1
                    log.ts_enq += 1
                else:
                    log.pkts_dropped_qfull += 1
                    log.ts_drops += 1
                next_arrival[i] += rng.exponential(1.0 / lam_node)
            next_global_arrival = np.min(next_arrival)
        
        if channel_busy > 0:
            channel_busy -= 1
            log.channel_busy_slots += 1
            continue
            
        if s % cfg.TDMA_SLOT_TOTAL_SLOTS == 0:
            current_tdma_slot_owner = (s // cfg.TDMA_SLOT_TOTAL_SLOTS) % cfg.N
            i = current_tdma_slot_owner
            if len(q[i]) > 0:
                arrivalTime = q[i].pop(0)
                
                log.tx_attempts += 1
                log.pkts_success += 1
                log.payload_bits_tx += cfg.payload_bits
                log.sum_end_to_end_delay_s += ((s + cfg.TX_slots) * cfg.slot_time_s) - arrivalTime
                
                current_q_sum -= 1
                log.ts_deq += 1
                
                # Channel busy accounts for BOTH TX time and Guard Time to prevent overlap mathematically
                channel_busy = cfg.TDMA_SLOT_TOTAL_SLOTS - 1
                log.channel_busy_slots += cfg.TX_slots # Only attribute actual communication time to utilization

def simulate_csma_ca(cfg: Config, offered_total_pps: float, log: Logger):
    rng = np.random.default_rng(cfg.seed)
    lam_node = offered_total_pps / float(cfg.N)
    if lam_node <= 0: return

    q = [[] for _ in range(cfg.N)]
    current_q_sum = 0
    cw = np.full(cfg.N, cfg.cw_min, dtype=int)
    backoff = np.full(cfg.N, -1, dtype=int)
    retry = np.zeros(cfg.N, dtype=int)
    
    channel_busy = 0
    idle_since = 0
    log.total_slots = cfg.total_slots
    
    # Calculate durations based on configuration
    if cfg.rts_cts_enabled and cfg.ack_enabled:
        busy_success = cfg.rts_slots + cfg.sifs_slots + cfg.cts_slots + cfg.sifs_slots + cfg.TX_slots + cfg.sifs_slots + cfg.ack_slots
        busy_collision = cfg.rts_slots + cfg.ack_timeout_slots # Collision on RTS
    elif cfg.ack_enabled:
        busy_success = cfg.TX_slots + cfg.sifs_slots + cfg.ack_slots
        busy_collision = cfg.TX_slots + cfg.ack_timeout_slots # Collision on Data, wait for ACK timeout
    else:
        busy_success = cfg.TX_slots
        busy_collision = cfg.TX_slots
    
    next_arrival = rng.exponential(1.0 / lam_node, size=cfg.N)
    next_global_arrival = np.min(next_arrival)
    
    max_cap = cfg.N * cfg.QMAX
    
    # Fake packet ID tracker for logs
    pkt_id_counter = 0

    for s in range(cfg.total_slots):
        if s > 0 and s % cfg.log_interval_slots == 0:
            log.record_queue_time_series(s, max_cap, current_q_sum)
            
        log.sum_q_len += current_q_sum
        
        t = s * cfg.slot_time_s
        if t >= next_global_arrival:
            arriving_nodes = np.where(t >= next_arrival)[0]
            for i in arriving_nodes:
                log.pkts_generated += 1
                if len(q[i]) < cfg.QMAX:
                    q[i].append(t)
                    current_q_sum += 1
                    log.ts_enq += 1
                else:
                    log.pkts_dropped_qfull += 1
                    log.ts_drops += 1
                next_arrival[i] += rng.exponential(1.0 / lam_node)
            next_global_arrival = np.min(next_arrival)
        
        if channel_busy > 0:
            channel_busy -= 1
            log.channel_busy_slots += 1
            if channel_busy == 0:
                idle_since = 0
            continue
            
        idle_since += 1
        
        if current_q_sum > 0:
            if idle_since <= cfg.difs_slots:
                continue
                
            has_pkt_arr = np.array([len(q[i]) > 0 for i in range(cfg.N)])
            need_bo = has_pkt_arr & (backoff < 0)
            if need_bo.any():
                backoff[need_bo] = rng.integers(0, cw[need_bo] + 1)
                
            tx_nodes = np.where(has_pkt_arr & (backoff == 0))[0]
            if tx_nodes.size == 0:
                counting = has_pkt_arr & (backoff > 0)
                backoff[counting] -= 1
                continue
                
            idle_since = 0
            log.tx_attempts += tx_nodes.size
            
            if tx_nodes.size == 1:
                i = tx_nodes[0]
                arrivalTime = q[i].pop(0)
                
                log.pkts_success += 1
                log.payload_bits_tx += cfg.payload_bits
                log.sum_end_to_end_delay_s += ((s + busy_success) * cfg.slot_time_s) - arrivalTime
                
                current_q_sum -= 1
                log.ts_deq += 1
                
                pkt_id_counter += 1
                if cfg.ack_enabled:
                    # Log successful ACK
                    log.record_ack(slot_idx=s, sender_id=int(i), receiver_id=-1, pkt_id=pkt_id_counter, status="success", retry_num=retry[i])
                
                cw[i] = cfg.cw_min
                retry[i] = 0
                backoff[i] = -1
                
                channel_busy = busy_success - 1
                log.channel_busy_slots += 1
            else:
                log.collision_events += 1
                for i in tx_nodes:
                    retry[i] += 1
                    if retry[i] > cfg.max_retry:
                        if len(q[i]) > 0:
                            q[i].pop(0)
                            current_q_sum -= 1
                            log.pkts_dropped_mac += 1
                            log.ts_drops += 1
                        retry[i] = 0
                        cw[i] = cfg.cw_min
                    else:
                        cw[i] = min(2 * cw[i] + 1, cfg.cw_max)
                    backoff[i] = -1
                
                channel_busy = busy_collision - 1
                log.channel_busy_slots += 1

def run_baseline_sweep():
    N = 20
    sim_time_s = 10.0
    from configs.config import TDMA_GUARD_TIME_S
    cfg = Config(N=N, sim_time_s=sim_time_s, QMAX=20, tdma_guard_time_s=TDMA_GUARD_TIME_S)
    
    traffic_pps_list = np.linspace(10, 1000, 20).astype(int)
    
    results = []
    
    print("Running Baseline Validation Sweep...")
    for idx, pps in enumerate(traffic_pps_list):
        print(f"Testing load {pps} pps ({idx+1}/{len(traffic_pps_list)})")
        
        log_aloha = Logger(protocol_name="ALOHA")
        cfg.seed = 42 + idx
        simulate_slotted_aloha(cfg, pps, log_aloha)
        
        log_tdma = Logger(protocol_name="TDMA")
        cfg.seed = 42 + idx
        simulate_tdma(cfg, pps, log_tdma)
        
        log_csma = Logger(protocol_name="CSMA_CA")
        cfg.seed = 42 + idx
        simulate_csma_ca(cfg, pps, log_csma)
        
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
            
            'ALOHA_Util': log_aloha.get_channel_utilization(),
            'TDMA_Util': log_tdma.get_channel_utilization(),
            'CSMA_Util': log_csma.get_channel_utilization(),
            
            'ALOHA_Collisions': log_aloha.collision_events,
            'TDMA_Collisions': log_tdma.collision_events,
            'CSMA_Collisions': log_csma.collision_events,
        })
        
    df = pd.DataFrame(results)
    df.to_csv("baseline_results.csv", index=False)
    print("Sweep complete. Generated baseline_results.csv")
    
    plt.figure(figsize=(10,6))
    plt.plot(df['Offered_Load_pps'], df['ALOHA_Throughput_Mbps'], label='Slotted ALOHA', marker='o')
    plt.plot(df['Offered_Load_pps'], df['TDMA_Throughput_Mbps'], label='TDMA', marker='s')
    plt.plot(df['Offered_Load_pps'], df['CSMA_Throughput_Mbps'], label='CSMA/CA', marker='^')
    plt.xlabel('Offered Traffic Rate (packets/sec total)')
    plt.ylabel('Throughput (Mbps)')
    plt.title(f'Baseline Protocol Comparison: Throughput\n(N={N}, Pkt=1500B, 2Mpbs Ideal Channel)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("baseline_throughput.png")
    
    plt.figure(figsize=(10,6))
    plt.plot(df['Offered_Load_pps'], df['ALOHA_Util'], label='Slotted ALOHA', marker='o')
    plt.plot(df['Offered_Load_pps'], df['TDMA_Util'], label='TDMA', marker='s')
    plt.plot(df['Offered_Load_pps'], df['CSMA_Util'], label='CSMA/CA', marker='^')
    plt.xlabel('Offered Traffic Rate (packets/sec total)')
    plt.ylabel('Channel Utilization')
    plt.title(f'Baseline Protocol Comparison: Channel Utilization (N={N})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("baseline_utilization.png")
    
    plt.figure(figsize=(10,6))
    plt.plot(df['Offered_Load_pps'], df['TDMA_Q_len'], label='TDMA', marker='s')
    plt.plot(df['Offered_Load_pps'], df['CSMA_Q_len'], label='CSMA/CA', marker='^')
    plt.xlabel('Offered Traffic Rate (packets/sec total)')
    plt.ylabel('Average Queue Length (packets)')
    plt.title(f'Baseline Protocol Comparison: Queue Occupancy\n(N={N}, Buffer Size={cfg.QMAX})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("baseline_queues.png")
    
    plt.figure(figsize=(10,6))
    plt.plot(df['Offered_Load_pps'], df['ALOHA_Drops'], label='Slotted ALOHA', marker='o')
    plt.plot(df['Offered_Load_pps'], df['TDMA_Drops'], label='TDMA', marker='s')
    plt.plot(df['Offered_Load_pps'], df['CSMA_Drops'], label='CSMA/CA', marker='^')
    plt.xlabel('Offered Traffic Rate (packets/sec total)')
    plt.ylabel('Dropped Packets (Buffer + Retries)')
    plt.title(f'Baseline Protocol Comparison: Packet Drops (N={N})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("baseline_drops.png")
    
def run_buffer_analysis():
    N = 20
    sim_time_s = 10.0
    from configs.config import TDMA_GUARD_TIME_S
    traffic_pps_list = np.linspace(10, 1000, 15).astype(int)
    q_sizes = [5, 20, 50]
    
    results = []
    print("Running Buffer Analysis...")
    
    for q in q_sizes:
        cfg = Config(N=N, sim_time_s=sim_time_s, QMAX=q, tdma_guard_time_s=TDMA_GUARD_TIME_S)
        for idx, pps in enumerate(traffic_pps_list):
            
            log_tdma = Logger()
            cfg.seed = 100 + idx + q
            simulate_tdma(cfg, pps, log_tdma)
            
            log_csma = Logger()
            cfg.seed = 100 + idx + q
            simulate_csma_ca(cfg, pps, log_csma)
            
            results.append({
                'QMAX': q,
                'Offered_Load_pps': pps,
                'TDMA_Throughput_Mbps': log_tdma.get_throughput_bps(sim_time_s) / 1e6,
                'CSMA_Throughput_Mbps': log_csma.get_throughput_bps(sim_time_s) / 1e6,
                'TDMA_Drops': log_tdma.pkts_dropped_qfull + log_tdma.pkts_dropped_mac,
                'CSMA_Drops': log_csma.pkts_dropped_qfull + log_csma.pkts_dropped_mac,
            })
            
    df = pd.DataFrame(results)
    df.to_csv("buffer_analysis_results.csv", index=False)
    print("Buffer Analysis complete. Generated buffer_analysis_results.csv")
    
    plt.figure(figsize=(10,6))
    colors = ['r', 'g', 'b']
    for i, q in enumerate(q_sizes):
        df_q = df[df['QMAX'] == q]
        plt.plot(df_q['Offered_Load_pps'], df_q['TDMA_Throughput_Mbps'], label=f'TDMA (Q={q})', marker='s', linestyle='--', color=colors[i])
        plt.plot(df_q['Offered_Load_pps'], df_q['CSMA_Throughput_Mbps'], label=f'CSMA/CA (Q={q})', marker='^', linestyle='-', color=colors[i])
    plt.xlabel('Offered Traffic Rate (packets/sec total)')
    plt.ylabel('Throughput (Mbps)')
    plt.title('Buffer Analysis: Throughput vs QMAX')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("buffer_analysis_throughput.png")

if __name__ == "__main__":
    run_baseline_sweep()
    run_buffer_analysis()
