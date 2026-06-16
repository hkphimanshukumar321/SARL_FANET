import os
import glob
import json
import torch
import torch.nn as nn
from stable_baselines3 import DQN, PPO, A2C
import qai_hub as hub

# Target API Key configuration
API_KEY = "smh0ph3k95nbyl9a6bbv9d79r1tmkjtnepx67jj2"
os.environ["QAI_HUB_API_TOKEN"] = API_KEY

# Define target devices
# We select 5 distinct edge devices covering IoT, XR, Phone, and Compute
TARGET_DEVICES = [
    "Dragonwing RB3 Gen 2 Vision Kit", # IoT standard (Linux)
    "XR2 Gen 2 (Proxy)", # Extended Reality / Drone Vision (Android)
    "Samsung Galaxy S24", # Samsung Galaxy (Snapdragon 8 Gen 3)
    "Snapdragon 8 Elite QRD", # Latest Reference Phone Design (Android)
    "Snapdragon X Elite CRD" # Next-gen Compute (Windows)
]

# Find the latest parallel run checkpoints
results_dir = r"c:\Users\hkphi\OneDrive\Desktop\SIMULATION PHASES\PHASE8_MARLSARLCENTRALIZED\sarl_on_marl_env\results"
latest_parallel_dir = sorted([d for d in os.listdir(results_dir) if "sarl_comparison_parallel_" in d])[-1]
CHECKPOINTS_DIR = os.path.join(results_dir, latest_parallel_dir, "checkpoints")
PROFILING_DIR = r"c:\Users\hkphi\OneDrive\Desktop\SIMULATION PHASES\PHASE8_MARLSARLCENTRALIZED\sarl_on_marl_env\profiling"
ONNX_DIR = os.path.join(PROFILING_DIR, "onnx_models")

os.makedirs(PROFILING_DIR, exist_ok=True)
os.makedirs(ONNX_DIR, exist_ok=True)

class SB3PolicyWrapper(nn.Module):
    """
    Wraps the SB3 Policy to take flat tensors instead of a Dict.
    This enables clean ONNX export for hardware profilers.
    """
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, scalars, history):
        # Pack back into the Dict observation space expected by SB3 MultiInputPolicy
        obs = {"scalars": scalars, "history": history}
        # Forward pass (predict action)
        # Note: SB3 policies return a tuple (action, value/log_prob) depending on the method.
        # We use policy.forward or policy.extract_features + action_net depending on architecture.
        # For simplicity and profiling, calling forward() gives the full compute graph.
        return self.policy(obs)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def export_and_profile():
    results = {}
    
    zip_files = glob.glob(os.path.join(CHECKPOINTS_DIR, "unified_*_model*.zip"))
    if not zip_files:
        print(f"No unified .zip checkpoints found in {CHECKPOINTS_DIR}")
        return

    for zip_path in zip_files:
        filename = os.path.basename(zip_path)
        algo_name = filename.split("_model.zip")[0].replace("unified_", "")
        
        print(f"\n[{algo_name.upper()}] Processing {filename}...")
        
        # Load correct SB3 class
        if "ppo" in algo_name:
            model_cls = PPO
        elif "a2c" in algo_name:
            model_cls = A2C
        elif "dqn" in algo_name:
            model_cls = DQN
        else:
            print(f"Unknown algorithm class for {algo_name}, skipping.")
            continue
            
        # 1. Load model onto CPU
        print("  -> Loading PyTorch model...")
        model = model_cls.load(zip_path, device="cpu")
        policy = model.policy.eval()
        
        # 2. Compute parameters
        trainable_params = count_parameters(policy)
        print(f"  -> Trainable Parameters: {trainable_params}")
        
        # 3. Create Wrapper and dummy tensors
        wrapper = SB3PolicyWrapper(policy).eval()
        dummy_scalars = torch.zeros(1, 14, dtype=torch.float32)
        dummy_history = torch.zeros(1, 5, 14, dtype=torch.float32)
        
        # 4. Export to ONNX
        onnx_path = os.path.join(ONNX_DIR, f"{algo_name}.onnx")
        print(f"  -> Exporting to ONNX: {onnx_path}")
        
        # ONNX export might fail on complex dict returns, so we wrap it
        # to just trace the graph
        try:
            traced_wrapper = torch.jit.trace(wrapper, (dummy_scalars, dummy_history), strict=False)
            torch.onnx.export(
                traced_wrapper, 
                (dummy_scalars, dummy_history), 
                onnx_path, 
                input_names=["scalars", "history"],
                opset_version=14
            )
            print("  -> ONNX export successful.")
        except Exception as e:
            print(f"  -> ONNX export failed: {e}")
            continue

        # 5. Submit to Qualcomm AI Hub
        results[algo_name] = {
            "trainable_parameters": trainable_params,
            "size_mb": os.path.getsize(onnx_path) / (1024 * 1024),
            "devices": {}
        }
        
        print(f"  -> Uploading {algo_name}.onnx to Qualcomm AI Hub...")
        try:
            uploaded_model = hub.upload_model(onnx_path)
        except Exception as e:
            print(f"  -> Upload failed: {e}")
            continue
            
        jobs = {}
        for device_name in TARGET_DEVICES:
            print(f"  -> Submitting job for {device_name}...")
            try:
                profile_job = hub.submit_profile_job(
                    model=uploaded_model,
                    device=hub.Device(device_name),
                    name=f"Profile_{algo_name}_{device_name.split()[0]}"
                )
                jobs[device_name] = profile_job
            except Exception as e:
                print(f"     Failed to submit on {device_name}: {e}")
                results[algo_name]["devices"][device_name] = {"error": str(e)}
        
        for device_name, profile_job in jobs.items():
            print(f"  -> Waiting for {device_name}...")
            try:
                profile_job.wait()
                profile_data = profile_job.download_profile()
                
                inference_time_ms = profile_data.execution_detail.estimated_inference_time / 1000.0 if profile_data.execution_detail else None
                memory_bytes = profile_data.execution_detail.peak_memory_bytes if profile_data.execution_detail else None
                macs = profile_data.execution_detail.macs if profile_data.execution_detail else None
                
                results[algo_name]["devices"][device_name] = {
                    "inference_time_ms": inference_time_ms,
                    "peak_memory_bytes": memory_bytes,
                    "macs": macs,
                    "gflops": (macs * 2 / 1e9) if macs else None,
                    "job_id": profile_job.job_id
                }
                print(f"     Done! Latency: {inference_time_ms} ms")
            except Exception as e:
                print(f"     Failed to fetch profile on {device_name}: {e}")
                results[algo_name]["devices"][device_name] = {"error": str(e)}
                
        with open(os.path.join(PROFILING_DIR, "profiling_results.json"), "w") as f:
            json.dump(results, f, indent=4)

    print(f"\nAll profiling completed! Data saved to {os.path.join(PROFILING_DIR, 'profiling_results.json')}")
    
    # Generate Table
    print("\n" + "="*80)
    print("PROFILING TABLE (Markdown)")
    print("="*80)
    print("| Model | Param.(M) | Size(MB) | GFLOPs | Mem.(MB) | RB3(ms) | XR2(ms) | S24(ms) | S8 Elite(ms) | X Elite(ms) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for model, data in results.items():
        param_m = data.get("trainable_parameters", 0) / 1e6
        size_mb = data.get("size_mb", 0)
        
        # Get GFLOPs and Mem from the first successful device
        gflops, mem_mb = 0, 0
        for dev, ddata in data.get("devices", {}).items():
            if "gflops" in ddata and ddata["gflops"]:
                gflops = ddata["gflops"]
                mem_mb = ddata["peak_memory_bytes"] / (1024*1024)
                break
                
        devs = data.get("devices", {})
        get_lat = lambda name: f'{devs.get(name, {}).get("inference_time_ms", "Err"):.3f}' if isinstance(devs.get(name, {}).get("inference_time_ms"), float) else 'Err'
        
        rb3 = get_lat("Dragonwing RB3 Gen 2 Vision Kit")
        xr2 = get_lat("XR2 Gen 2 (Proxy)")
        s24 = get_lat("Samsung Galaxy S24")
        s8 = get_lat("Snapdragon 8 Elite QRD")
        xe = get_lat("Snapdragon X Elite CRD")
        
        print(f"| {model} | {param_m:.4f} | {size_mb:.2f} | {gflops:.4f} | {mem_mb:.2f} | {rb3} | {xr2} | {s24} | {s8} | {xe} |")

if __name__ == "__main__":
    export_and_profile()
