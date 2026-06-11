# utils/experiment_tracking.py
# ============================================================
# Weights & Biases integration for FANET experiment tracking.
# Auto-detects wandb installation; all functions gracefully
# no-op if wandb is not installed or not logged in.
# ============================================================

import os

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    wandb = None
    _HAS_WANDB = False

# Environment-variable override: set WANDB_DISABLED=1 to force off
_WANDB_DISABLED = os.environ.get("WANDB_DISABLED", "0") == "1"

_active_run = None
_login_attempted = False


def _read_api_key_from_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def _get_api_key() -> str:
    api_key = os.environ.get("WANDB_API_KEY", "").strip()
    if api_key:
        return api_key
    api_key_file = os.environ.get("WANDB_API_KEY_FILE", "").strip()
    if api_key_file:
        return _read_api_key_from_file(api_key_file)
    return ""


def _has_saved_credentials():
    api_key = _get_api_key()
    if api_key:
        return True
    home = os.path.expanduser("~")
    return os.path.exists(os.path.join(home, ".netrc")) or os.path.exists(os.path.join(home, "_netrc"))


def is_enabled():
    """Check if wandb is available and not explicitly disabled."""
    return _HAS_WANDB and not _WANDB_DISABLED


def _maybe_login():
    """
    Non-interactive wandb auth.

    Practical rule:
    - if WANDB_API_KEY is present, use it
    - otherwise rely on an existing netrc/session
    - never prompt from library import paths
    """
    global _login_attempted
    if not is_enabled() or _login_attempted:
        return
    _login_attempted = True
    api_key = _get_api_key()
    if not api_key:
        return
    try:
        wandb.login(key=api_key, relogin=False)
    except Exception:
        # Tracking should never block the experiment pipeline.
        pass


def init_run(project="fanet-mac-rl", config=None, run_name=None, tags=None,
             group=None, notes=None, reinit=True):
    """Initialise a wandb run. No-op if wandb is unavailable.

    Parameters
    ----------
    project : str
        wandb project name.
    config : dict | None
        Experiment configuration dict (logged as run hyperparameters).
    run_name : str | None
        Human-readable run name (appears in the wandb dashboard).
    tags : list[str] | None
        Tags for filtering runs (e.g. ["ablation", "fading"]).
    group : str | None
        Group name for related runs (e.g. "unified_experiment").
    notes : str | None
        Free-text notes for this run.
    reinit : bool
        Allow re-initialisation within the same process.

    Returns
    -------
    wandb.Run | None
    """
    global _active_run
    if not is_enabled():
        return None
    if not _has_saved_credentials():
        return None
    _maybe_login()

    try:
        _active_run = wandb.init(
            project=project,
            config=config or {},
            name=run_name,
            tags=tags,
            group=group,
            notes=notes,
            reinit=reinit,
        )
    except Exception:
        _active_run = None
    return _active_run


def log_metrics(metrics, step=None, commit=True):
    """Log a dictionary of metrics to the active wandb run.

    Parameters
    ----------
    metrics : dict
        Key-value pairs, e.g. {"reward": 3.14, "epsilon": 0.5}.
    step : int | None
        Global step number (training step or episode).
    commit : bool
        If True, flush immediately; if False, buffer until next commit.
    """
    if not is_enabled() or _active_run is None:
        return
    wandb.log(metrics, step=step, commit=commit)


def log_artifact(filepath, artifact_name=None, artifact_type="result"):
    """Upload a file (CSV, image, checkpoint) as a wandb Artifact.

    Parameters
    ----------
    filepath : str
        Absolute path to the file to upload.
    artifact_name : str | None
        Name for the artifact; defaults to the filename stem.
    artifact_type : str
        One of "result", "model", "dataset", etc.
    """
    if not is_enabled() or _active_run is None:
        return
    if not os.path.exists(filepath):
        return
    name = artifact_name or os.path.splitext(os.path.basename(filepath))[0]
    art = wandb.Artifact(name=name, type=artifact_type)
    art.add_file(filepath)
    _active_run.log_artifact(art)


def log_image(key, image_path):
    """Log an image to wandb under the given key.

    Parameters
    ----------
    key : str
        Metric key for the image panel, e.g. "plots/throughput_comparison".
    image_path : str
        Absolute path to the image file.
    """
    if not is_enabled() or _active_run is None:
        return
    if not os.path.exists(image_path):
        return
    wandb.log({key: wandb.Image(image_path)})


def finish_run():
    """Finalise and close the active wandb run."""
    global _active_run
    if not is_enabled() or _active_run is None:
        return
    wandb.finish()
    _active_run = None


# ============================================================
# SB3-compatible callback
# ============================================================
class WandbSB3Callback:
    """Stable-Baselines3 training callback that logs to wandb.

    Usage:
        from utils.experiment_tracking import WandbSB3Callback
        cb = WandbSB3Callback(algo_name="dqn")
        model.learn(total_timesteps=10000, callback=cb)
    """

    def __init__(self, algo_name="rl", verbose=0):
        self.algo_name = algo_name
        self.verbose = verbose
        self._enabled = is_enabled()
        # Will be set by SB3 when callback is attached
        self.num_timesteps = 0
        self.locals = {}
        self.globals = {}
        self.model = None
        self.logger = None
        self.n_calls = 0

    def init_callback(self, model):
        self.model = model

    def _on_step(self):
        self.n_calls += 1
        self.num_timesteps = self.model.num_timesteps if self.model else self.n_calls

        if not self._enabled:
            return True

        infos = self.locals.get("infos", [{}])
        if infos and "episode" in infos[0]:
            ep = infos[0]["episode"]
            log_metrics({
                f"{self.algo_name}/episode_reward": ep["r"],
                f"{self.algo_name}/episode_length": ep["l"],
            }, step=self.num_timesteps)

        return True

    def _on_training_start(self):
        pass

    def _on_training_end(self):
        pass

    def _on_rollout_start(self):
        pass

    def _on_rollout_end(self):
        pass

    # ------ SB3 BaseCallback interface ------
    def on_step(self):
        return self._on_step()


# ============================================================
# MARL episode logger
# ============================================================
class WandbMARLLogger:
    """Lightweight logger for custom MARL training loops.

    Usage:
        logger = WandbMARLLogger(algo_name="iql")
        for ep in range(episodes):
            ...
            logger.log_episode(ep, reward=ep_reward, epsilon=eps, loss=loss)
        logger.finish()
    """

    def __init__(self, algo_name="marl"):
        self.algo_name = algo_name
        self._enabled = is_enabled()

    def log_episode(self, episode, **kwargs):
        """Log per-episode metrics.

        Parameters
        ----------
        episode : int
            Episode number.
        **kwargs
            Arbitrary metrics, e.g. reward=3.14, epsilon=0.5, loss=0.01.
        """
        if not self._enabled:
            return
        metrics = {f"{self.algo_name}/{k}": v for k, v in kwargs.items()}
        log_metrics(metrics, step=episode)

    def log_eval(self, step, **kwargs):
        """Log evaluation metrics."""
        if not self._enabled:
            return
        metrics = {f"{self.algo_name}/eval_{k}": v for k, v in kwargs.items()}
        log_metrics(metrics, step=step)
