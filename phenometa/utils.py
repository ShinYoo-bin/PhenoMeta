import csv, datetime, json, logging, os, random, threading, time, traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

RESOURCE_COLUMNS = ["ts", "tag", "epoch", "repeat", "fold", "sample_kind", "cpu_proc", "cpu_sys", "rss_gb", "uss_gb",
                    "vms_gb", "threads", "io_read_mb", "io_write_mb", "gpu_alloc_gb", "gpu_reserved_gb",
                    "gpu_peak_alloc_gb", "gpu_peak_reserved_gb"]


def ensure_dir(path): Path(path).mkdir(parents=True, exist_ok=True)
def run_id(): return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
def now(): return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
def save_json(value, path):
    ensure_dir(Path(path).parent)
    with open(path, "w") as stream: json.dump(value, stream, indent=2, default=str)


def atomic_save_json(value, path):
    """Atomically replace a JSON status file so abrupt exits keep the last valid state."""
    path = Path(path); ensure_dir(path.parent)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "w") as stream:
            json.dump(value, stream, indent=2, default=str)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_frame(frame, path):
    """Persist the latest in-memory history without exposing a partial CSV."""
    path = Path(path); ensure_dir(path.parent)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
def append_csv(path, row, columns):
    path = Path(path); ensure_dir(path.parent); new = not path.exists()
    with open(path, "a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        if new: writer.writeheader()
        writer.writerow(row)


def atomic_upsert_csv(path, row, columns, key_columns):
    """Insert or replace one CSV row and atomically remove duplicate keys."""
    import pandas as pd
    path = Path(path); ensure_dir(path.parent)
    incoming = pd.DataFrame([{column: row.get(column, np.nan) for column in columns}])
    if path.exists() and path.stat().st_size:
        current = pd.read_csv(path)
        for column in columns:
            if column not in current:
                current[column] = np.nan
        frame = pd.concat([current[columns], incoming], ignore_index=True)
    else:
        frame = incoming
    frame = frame.drop_duplicates(subset=list(key_columns), keep="last")
    atomic_write_frame(frame[columns], path)


class RuntimeSession:
    """Durable per-invocation wall-time ledger with reboot-safe heartbeats."""

    COLUMNS = ["invocation_id", "dataset", "started_at", "last_heartbeat_at",
               "ended_at", "status", "active_wall_time_sec"]

    def __init__(self, output_dir, invocation_id, dataset):
        self.path = Path(output_dir) / "runtime_sessions.csv"
        self.invocation_id = str(invocation_id)
        self.dataset = str(dataset)
        self.started = datetime.datetime.now()
        self._write("running", self.started, None)

    def _read(self):
        import pandas as pd
        if not self.path.exists() or not self.path.stat().st_size:
            return pd.DataFrame(columns=self.COLUMNS)
        frame = pd.read_csv(self.path)
        for column in self.COLUMNS:
            if column not in frame:
                frame[column] = np.nan
        for column in ("invocation_id", "dataset", "started_at", "last_heartbeat_at", "ended_at", "status"):
            frame[column] = frame[column].astype("object")
        # A prior row left running was terminated without a clean shutdown.
        mask = (frame["status"] == "running") & (frame["invocation_id"].astype(str) != self.invocation_id)
        frame.loc[mask, "status"] = "interrupted"
        frame.loc[mask, "ended_at"] = frame.loc[mask, "last_heartbeat_at"]
        return frame[self.COLUMNS]

    def _write(self, status, heartbeat, ended):
        import pandas as pd
        frame = self._read()
        heartbeat_text = heartbeat.strftime("%Y-%m-%d %H:%M:%S")
        elapsed = max(0.0, (heartbeat - self.started).total_seconds())
        row = {
            "invocation_id": self.invocation_id, "dataset": self.dataset,
            "started_at": self.started.strftime("%Y-%m-%d %H:%M:%S"),
            "last_heartbeat_at": heartbeat_text,
            "ended_at": ended.strftime("%Y-%m-%d %H:%M:%S") if ended else np.nan,
            "status": status, "active_wall_time_sec": elapsed,
        }
        frame = frame[frame["invocation_id"].astype(str) != self.invocation_id]
        incoming = pd.DataFrame([row])
        frame = incoming if frame.empty else pd.concat([frame, incoming], ignore_index=True)
        atomic_write_frame(frame[self.COLUMNS], self.path)

    def heartbeat(self):
        self._write("running", datetime.datetime.now(), None)

    def close(self, status):
        ended = datetime.datetime.now()
        self._write(status, ended, ended)

    def total_active_seconds(self):
        import pandas as pd
        frame = self._read()
        return float(pd.to_numeric(frame["active_wall_time_sec"], errors="coerce").fillna(0).sum())
def logger(path):
    log = logging.getLogger("phenometa"); log.setLevel(logging.INFO); log.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(), logging.FileHandler(path)):
        handler.setFormatter(formatter); log.addHandler(handler)
    return log


class RunDiagnostics:
    """Durable stage state and exception context for long-running PhenoMeta jobs."""

    def __init__(self, output_dir, log, run_ts, dataset):
        self.output_dir = Path(output_dir)
        self.log = log
        self.base = {"run_ts": str(run_ts), "dataset": str(dataset)}
        self.current = {}
        self.failure_recorded = False
        previous_failure = self.output_dir / "failure.json"
        if previous_failure.exists():
            archived = self.output_dir / f"failure.previous.{run_ts}.json"
            os.replace(previous_failure, archived)
        self.update(status="starting", stage="initialization")

    @property
    def state_path(self):
        return self.output_dir / "run_state.json"

    @property
    def failure_path(self):
        return self.output_dir / "failure.json"

    def update(self, **values):
        self.current.update(values)
        payload = {**self.base, **self.current, "updated_at": now()}
        atomic_save_json(payload, self.state_path)
        return payload

    def fail(self, exc, *, stage=None, **context):
        stage = stage or self.current.get("stage", "unknown")
        payload = self.update(
            status="failed",
            stage=stage,
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            traceback=traceback.format_exc(),
            **context,
        )
        atomic_save_json(payload, self.failure_path)
        self.failure_recorded = True
        return payload

    @contextmanager
    def stage(self, name, *, repeat=-1, fold=-1, **context):
        details = {"stage": str(name), "repeat": int(repeat), "fold": int(fold), **context}
        self.failure_recorded = False
        # Stage context is a snapshot, not an accumulating dictionary.  Without
        # clearing it here, a later failure can misleadingly retain e.g. the
        # previous stage's split/input_path/cache_path.
        self.current = {}
        self.update(status="running", **details)
        self.log.info(
            "[stage:start] stage=%s repeat=%s fold=%s context=%s",
            name, repeat, fold, json.dumps(context, sort_keys=True, default=str),
        )
        start = time.time()
        try:
            yield
        except BaseException as exc:
            elapsed = time.time() - start
            self.fail(exc, elapsed_sec=elapsed, **details)
            self.log.exception(
                "[stage:failed] stage=%s repeat=%s fold=%s elapsed_sec=%.3f "
                "exception_type=%s exception=%s",
                name, repeat, fold, elapsed, type(exc).__name__, exc,
            )
            raise
        else:
            elapsed = time.time() - start
            self.log.info(
                "[stage:done] stage=%s repeat=%s fold=%s elapsed_sec=%.3f",
                name, repeat, fold, elapsed,
            )
            self.update(status="running", stage=f"{name}:done", repeat=repeat, fold=fold,
                        elapsed_sec=elapsed, **context)

    def complete(self):
        self.current = {}
        self.update(status="complete", stage="complete", repeat=-1, fold=-1)


class ResourceLogger:
    def __init__(self, path, interval=30, device="cuda", heartbeat=None):
        self.path, self.interval = Path(path), interval
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tag, self.epoch, self.repeat, self.fold = "init", -1, -1, -1
        self.stop_event, self.thread, self.lock = threading.Event(), None, threading.Lock()
        self.heartbeat = heartbeat
        self.columns = RESOURCE_COLUMNS
        try:
            import psutil
            self.psutil, self.process = psutil, psutil.Process(os.getpid())
        except ImportError: self.psutil = self.process = None
        ensure_dir(self.path.parent)
        if not self.path.exists():
            with open(self.path, "w", newline="") as stream: csv.writer(stream).writerow(self.columns)
        else:
            # Preserve append compatibility for legacy runs. New runs always
            # receive the sample_kind column and can separate periodic means
            # from explicit boundary/peak snapshots.
            with open(self.path, newline="") as stream:
                existing = next(csv.reader(stream), [])
            if existing:
                self.columns = existing
    def set_status(self, tag, epoch=-1, repeat=-1, fold=-1):
        with self.lock: self.tag, self.epoch, self.repeat, self.fold = tag, epoch, repeat, fold
    def synchronize_cuda(self):
        if self.device.type == "cuda":
            try:
                index = self.device.index if self.device.index is not None else torch.cuda.current_device()
                torch.cuda.synchronize(index)
            except Exception: pass
    def reset_gpu_peak(self):
        if self.device.type == "cuda":
            try:
                index = self.device.index if self.device.index is not None else torch.cuda.current_device()
                torch.cuda.synchronize(index)
                torch.cuda.reset_peak_memory_stats(index)
            except Exception: pass
    def start(self):
        self.thread = threading.Thread(target=self._loop, daemon=True); self.thread.start()
    def stop(self):
        self.stop_event.set()
        if self.thread: self.thread.join(timeout=5)
        self.record("boundary")
    def _loop(self):
        while not self.stop_event.wait(self.interval): self.record("periodic")
    def record(self, sample_kind="boundary"):
        row = {key: np.nan for key in RESOURCE_COLUMNS}
        with self.lock:
            row.update(ts=now(), tag=self.tag, epoch=self.epoch, repeat=self.repeat,
                       fold=self.fold, sample_kind=str(sample_kind))
        if self.process:
            try:
                mem = self.process.memory_info(); full = self.process.memory_full_info()
                io = self.process.io_counters()
                row.update(rss_gb=mem.rss/2**30, uss_gb=getattr(full, "uss", np.nan)/2**30,
                           vms_gb=mem.vms/2**30, threads=self.process.num_threads(),
                           io_read_mb=io.read_bytes/2**20, io_write_mb=io.write_bytes/2**20)
                if sample_kind == "periodic":
                    row.update(cpu_proc=self.process.cpu_percent(), cpu_sys=self.psutil.cpu_percent())
            except Exception: pass
        if self.device.type == "cuda":
            try:
                i = self.device.index or torch.cuda.current_device()
                row.update(gpu_alloc_gb=torch.cuda.memory_allocated(i)/2**30, gpu_reserved_gb=torch.cuda.memory_reserved(i)/2**30,
                           gpu_peak_alloc_gb=torch.cuda.max_memory_allocated(i)/2**30,
                           gpu_peak_reserved_gb=torch.cuda.max_memory_reserved(i)/2**30)
            except Exception: pass
        append_csv(self.path, row, self.columns)
        if self.heartbeat:
            try: self.heartbeat()
            except Exception: pass


def summarize_resource(path, tag):
    import pandas as pd
    frame = pd.read_csv(path)
    part = frame[frame["tag"].astype(str).str.contains(tag, na=False)]
    mean_part = part
    if "sample_kind" in part:
        periodic = part[part["sample_kind"].astype(str).eq("periodic")]
        if not periodic.empty:
            mean_part = periodic
    result = {}
    for column in ("uss_gb", "gpu_alloc_gb", "gpu_peak_alloc_gb", "gpu_peak_reserved_gb"):
        result[f"{column}_mean"] = float(mean_part[column].mean()) if len(mean_part) else np.nan
        result[f"{column}_max"] = float(part[column].max()) if len(part) else np.nan
    return result
