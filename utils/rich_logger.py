# utils/rich_logger.py
# ============================================================
# Rich-based console logging for FANET experiments.
# Auto-detects rich installation; falls back to plain print()
# and tqdm if rich is not available.
# ============================================================

import datetime

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn, TextColumn,
        TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn,
    )
    from rich.table import Table
    from rich.theme import Theme
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False


# ── Shared console instance ─────────────────────────────────

_console = None


def get_console():
    """Return the shared Rich Console (or None if rich unavailable)."""
    global _console
    if not _HAS_RICH:
        return None
    if _console is None:
        theme = Theme({
            "info": "cyan",
            "success": "bold green",
            "warning": "bold yellow",
            "error": "bold red",
            "step": "bold blue",
        })
        _console = Console(theme=theme)
    return _console


# ── Logging helpers ──────────────────────────────────────────

def log_step(msg, style="info"):
    """Print a timestamped log message.

    Parameters
    ----------
    msg : str
        Message text.
    style : str
        Rich style name: "info", "success", "warning", "error", "step".
    """
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    c = get_console()
    if c is not None:
        c.print(f"[dim]{ts}[/dim]  [{style}]{msg}[/{style}]")
    else:
        print(f"[{ts}] {msg}")


def print_banner(title, subtitle=None):
    """Print a prominent experiment banner.

    Parameters
    ----------
    title : str
        Banner title (e.g. "UNIFIED EXPERIMENT RUNNER").
    subtitle : str | None
        Optional subtitle line.
    """
    c = get_console()
    if c is not None:
        content = f"[bold]{title}[/bold]"
        if subtitle:
            content += f"\n{subtitle}"
        c.print(Panel(content, border_style="bright_cyan", expand=False))
    else:
        print("=" * 60)
        print(f" {title}")
        if subtitle:
            print(f" {subtitle}")
        print("=" * 60)


def print_section(title):
    """Print a section header (lighter than a full banner).

    Parameters
    ----------
    title : str
        Section title (e.g. "STEP 1: Baseline MAC Simulations").
    """
    c = get_console()
    if c is not None:
        c.rule(f"[bold cyan]{title}[/bold cyan]")
    else:
        print("\n" + "=" * 60)
        print(f" {title}")
        print("=" * 60)


def print_table(title, columns, rows):
    """Print a nicely formatted table.

    Parameters
    ----------
    title : str
        Table title.
    columns : list[str]
        Column header names.
    rows : list[list[str]]
        Row data (each row is a list of string values).
    """
    c = get_console()
    if c is not None:
        table = Table(title=title, show_lines=True)
        for col in columns:
            table.add_column(col)
        for row in rows:
            table.add_row(*[str(v) for v in row])
        c.print(table)
    else:
        # Simple fallback
        print(f"\n--- {title} ---")
        print("  ".join(columns))
        for row in rows:
            print("  ".join(str(v) for v in row))


# ── Progress bar ─────────────────────────────────────────────

def create_progress(description="Working", total=None):
    """Create a progress bar context manager.

    Usage:
        with create_progress("Training", total=1000) as progress:
            task = progress.add_task("DQN", total=1000)
            for step in range(1000):
                ...
                progress.update(task, advance=1)

    Returns
    -------
    Progress-like context manager.
    """
    if _HAS_RICH:
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=get_console(),
            transient=False,
        )
    else:
        # Fallback: return a tqdm-compatible wrapper
        return _TqdmFallbackProgress()


class _TqdmFallbackProgress:
    """Minimal progress wrapper using tqdm as fallback."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def add_task(self, description, total=None):
        try:
            from tqdm import tqdm
            bar = tqdm(total=total, desc=description, unit="step")
            self._bars = getattr(self, '_bars', {})
            task_id = len(self._bars)
            self._bars[task_id] = bar
            return task_id
        except ImportError:
            self._bars = getattr(self, '_bars', {})
            task_id = len(self._bars)
            self._bars[task_id] = None
            return task_id

    def update(self, task_id, advance=1, **kwargs):
        bars = getattr(self, '_bars', {})
        bar = bars.get(task_id)
        if bar is not None:
            bar.update(advance)
            postfix = {k: v for k, v in kwargs.items() if k not in ('advance',)}
            if postfix:
                bar.set_postfix(postfix)

    def stop(self):
        for bar in getattr(self, '_bars', {}).values():
            if bar is not None:
                bar.close()
