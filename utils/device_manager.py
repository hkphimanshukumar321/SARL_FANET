import os
import psutil
import json
from configs import config as global_cfg

def get_cpu_info():
    """Returns the number of physical and logical CPU cores and total RAM."""
    physical = psutil.cpu_count(logical=False)
    logical = psutil.cpu_count(logical=True)
    ram_gb = psutil.virtual_memory().total / (1024**3)
    return {
        "physical_cores": physical,
        "logical_cores": logical,
        "ram_gb": round(ram_gb, 2)
    }

def get_gpu_info():
    """Returns GPU availability and stats if PyTorch or TensorFlow is installed."""
    info = {"available": False, "count": 0, "devices": []}
    
    # Check PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            info["available"] = True
            info["backend"] = "torch"
            info["count"] = torch.cuda.device_count()
            for i in range(info["count"]):
                props = torch.cuda.get_device_properties(i)
                info["devices"].append({
                    "id": i,
                    "name": props.name,
                    "vram_gb": round(props.total_memory / (1024**3), 2)
                })
            return info
    except ImportError:
        pass
        
    # Check TensorFlow fallback
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            info["available"] = True
            info["backend"] = "tensorflow"
            info["count"] = len(gpus)
            for i, g in enumerate(gpus):
                info["devices"].append({
                    "id": i,
                    "name": g.name,
                    "vram_gb": "unknown"  # TF doesn't easily expose total VRAM securely here without init
                })
    except ImportError:
        pass
        
    return info

def resolve_device(purpose="train"):
    """
    Decides the PyTorch device string ('cpu' or 'cuda:0') based on global config
    and hardware availability.
    purpose: 'train' or 'eval'
    """
    if global_cfg.FORCE_CPU or not global_cfg.ENABLE_GPU:
        return "cpu"
        
    gpu_info = get_gpu_info()
    if not gpu_info["available"]:
        return "cpu"
        
    if purpose == "train" and not global_cfg.TRAIN_ON_GPU:
        return "cpu"
        
    if purpose == "eval" and not global_cfg.EVAL_ON_GPU:
        return "cpu"
        
    # Bind to requested GPU
    target = global_cfg.GPU_DEVICE_ID
    if target < gpu_info["count"]:
        return f"cuda:{target}"
    return "cuda:0" # Fallback if requested ID is out of bounds

def get_optimal_worker_count():
    """
    Calculates the number of CPU workers to spawn for multiprocessing runs based on config.
    """
    if not global_cfg.ENABLE_MULTIPROCESSING:
        return 1
        
    if isinstance(global_cfg.NUM_CPU_WORKERS, int) and global_cfg.NUM_CPU_WORKERS > 0:
        return global_cfg.NUM_CPU_WORKERS
        
    # Auto mode: Use configured fraction of logical cores
    logical = psutil.cpu_count(logical=True)
    if logical is None:
        logical = 1
        
    fraction = max(0.1, min(1.0, global_cfg.CPU_UTILIZATION_FRACTION))
    workers = int(logical * fraction)
    return max(1, workers)

def generate_device_report(output_dir):
    """
    Generates a system snapshot and saves it to metadata/device_report.json
    """
    report = {
        "cpu": get_cpu_info(),
        "gpu": get_gpu_info(),
        "allocations": {
            "requested_workers": get_optimal_worker_count(),
            "train_device": resolve_device("train"),
            "eval_device": resolve_device("eval")
        },
        "config_echo": {
            "ENABLE_MULTIPROCESSING": global_cfg.ENABLE_MULTIPROCESSING,
            "ENABLE_GPU": global_cfg.ENABLE_GPU,
            "FORCE_CPU": global_cfg.FORCE_CPU
        }
    }
    
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "device_report.json")
    
    if global_cfg.ENABLE_RESOURCE_LOGGING:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
            
    return report

if __name__ == "__main__":
    print(json.dumps(generate_device_report("."), indent=2))
