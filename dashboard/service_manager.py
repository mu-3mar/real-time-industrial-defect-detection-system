#!/usr/bin/env python3
"""Service Manager Daemon for QC-SCM Detection Dashboard.

Allows local client-side Javascript to run, stop, and restart the backend
runtime on port 8080.
"""

import http.server
import json
import subprocess
import os
import signal
import sys
import urllib.request
import urllib.parse
import datetime
import time
import psutil
import socket
import uuid
from pathlib import Path

PORT = 8080

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_DIR = PROJECT_ROOT / "src" / "qc_scm" / "runtime"
MAIN_PY = RUNTIME_DIR / "main.py"
LOG_FILE = Path(__file__).resolve().parent / "service.log"
STATE_FILE = Path(__file__).resolve().parent / "service_manager_state.json"
API_BASE = os.environ.get("QC_SCM_API_URL", "http://127.0.0.1:8000").rstrip("/")
TUNNEL_URL = os.environ.get("QC_SCM_TUNNEL_URL", "").strip().rstrip("/")

CONDA_ENV = os.environ.get("QC_SCM_CONDA_ENV", "qc")


def _run_python_in_flow(cmd_args):
    """Build Popen args: prefer conda run -n qc python ... else fall back to sys.executable.

    Always set cwd=RUNTIME_DIR so legacy runtime imports work.
    Extend PYTHONPATH with RUNTIME_DIR as safety net.
    """
    env = os.environ.copy()
    runtime_path = str(RUNTIME_DIR)
    existing_pp = env.get("PYTHONPATH", "")
    if existing_pp:
        env["PYTHONPATH"] = f"{runtime_path}:{existing_pp}"
    else:
        env["PYTHONPATH"] = runtime_path

    # Try conda first
    try:
        conda_check = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if CONDA_ENV in conda_check.stdout:
            return [
                "conda", "run", "--no-capture-output", "-n", CONDA_ENV,
                "python", str(MAIN_PY), *cmd_args
            ], env
    except Exception:
        pass

    # Fallback: whatever python we are running with
    return [sys.executable, str(MAIN_PY), *cmd_args], env


def _read_state():
    try:
        with STATE_FILE.open(encoding="utf-8") as stream:
            state = json.load(stream)
        return state if isinstance(state, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_state(state):
    temporary = STATE_FILE.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(state, stream)
    temporary.replace(STATE_FILE)


def _clear_state():
    try:
        STATE_FILE.unlink()
    except FileNotFoundError:
        pass


def _owned_process():
    """Return the managed process only when its PID identity still matches."""
    state = _read_state()
    pid = state.get("pid")
    created = state.get("create_time")
    if not isinstance(pid, int) or not isinstance(created, (int, float)):
        return None
    try:
        process = psutil.Process(pid)
        if abs(process.create_time() - float(created)) > 1:
            _clear_state()
            return None
        if str(MAIN_PY) not in " ".join(process.cmdline()):
            _clear_state()
            return None
        return process
    except (psutil.Error, OSError):
        _clear_state()
        return None


def get_service_pid():
    process = _owned_process()
    return process.pid if process else None


def get_process_start_time(pid):
    try:
        output = subprocess.check_output(["ps", "-o", "lstart=", "-p", str(pid)])
        dt = datetime.datetime.strptime(output.decode().strip(), "%a %b %d %H:%M:%S %Y")
        return dt
    except Exception:
        return None


def _get_json(url, timeout=3):
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError):
        return None


def _health_api_status():
    data = _get_json(f"{API_BASE}/api/health")
    if data and data.get("status") == "healthy":
        return {"state": "healthy", "detail": data}
    return {"state": "offline", "detail": "Health endpoint did not report healthy"}


def _storage_status(process):
    if not process:
        return {"state": "unavailable", "detail": "Service is stopped"}
    return _get_json(f"{API_BASE}/api/health/storage") or {
        "state": "error", "detail": "Local storage health endpoint unavailable"
    }


def _tunnel_status(process):
    if not TUNNEL_URL:
        return {"state": "not_configured", "detail": "QC_SCM_TUNNEL_URL is not set"}
    if not process:
        return {"state": "inactive", "detail": TUNNEL_URL}
    data = _get_json(f"{TUNNEL_URL}/api/health")
    if data and data.get("status") == "healthy":
        return {"state": "healthy", "detail": TUNNEL_URL}
    return {"state": "inactive", "detail": TUNNEL_URL}


def check_api_health():
    return _health_api_status()["state"] == "healthy"


_last_net_bytes = 0
_last_net_time = 0


def get_network_speed():
    global _last_net_bytes, _last_net_time
    try:
        counters = psutil.net_io_counters()
        curr_bytes = counters.bytes_sent + counters.bytes_recv
        curr_time = time.time()

        if _last_net_bytes == 0:
            _last_net_bytes = curr_bytes
            _last_net_time = curr_time
            return 0

        elapsed = curr_time - _last_net_time
        if elapsed <= 0:
            return 0

        speed = (curr_bytes - _last_net_bytes) / elapsed
        _last_net_bytes = curr_bytes
        _last_net_time = curr_time
        return int(speed / 1024)  # KB/s
    except Exception:
        return 0


def get_gpu_percent():
    """Try to read GPU utilization via nvidia-smi. Returns 0-100 or None."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            timeout=2,
        )
        vals = [int(v.strip()) for v in out.decode().strip().split("\n") if v.strip()]
        if vals:
            return vals[0]
    except Exception:
        pass
    return None


def get_system_metrics():
    try:
        cpu = psutil.cpu_percent(interval=None)
    except Exception:
        cpu = 0

    try:
        ram = psutil.virtual_memory().percent
    except Exception:
        ram = 0

    try:
        disk = psutil.disk_usage('/').percent
    except Exception:
        disk = 0

    gpu = get_gpu_percent()

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.0)
        s.connect(("8.8.8.8", 53))
        s.close()
        network = "Connected"
    except Exception:
        network = "Disconnected"

    result = {
        "cpu_percent": cpu,
        "ram_percent": ram,
        "disk_percent": disk,
        "gpu_percent": gpu if gpu is not None else 0,
        "network_status": network,
        "network_speed_kb": get_network_speed(),
    }
    return result


def _log(message, level="INFO"):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    with LOG_FILE.open("a", encoding="utf-8", buffering=1) as stream:
        stream.write(f"[{timestamp}] [{level}] [Manager] {message}\n")


def _reset_log():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    session_id = uuid.uuid4().hex
    with LOG_FILE.open("w", encoding="utf-8") as stream:
        stream.write(
            f"[{datetime.datetime.now():%H:%M:%S}] [INFO] [Manager] "
            f"Service session {session_id} started\n"
        )
    return session_id


def _start_service():
    existing = _owned_process()
    if existing:
        return existing, "Service already running"
    popen_args, env = _run_python_in_flow([])
    session_id = _reset_log()
    log_file = LOG_FILE.open("a", encoding="utf-8", buffering=1)
    try:
        process = subprocess.Popen(
            popen_args,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=str(RUNTIME_DIR),
            env=env,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log_file.close()
    _write_state({
        "pid": process.pid,
        "create_time": psutil.Process(process.pid).create_time(),
        "pgid": os.getpgid(process.pid),
        "session_id": session_id,
    })
    _log(f"Detection service started (PID {process.pid})")
    return process, "Service start initiated"


def _stop_service():
    process = _owned_process()
    if not process:
        return False, "Service is already stopped"
    pgid = _read_state().get("pgid")
    manager_pgid = os.getpgrp()
    if not isinstance(pgid, int) or pgid <= 1 or pgid in {manager_pgid, os.getpid()}:
        return False, "Refusing to stop an untrusted service process group"
    try:
        os.killpg(pgid, signal.SIGTERM)
        psutil.wait_procs([process], timeout=8)
        try:
            os.killpg(pgid, 0)
            group_alive = True
        except ProcessLookupError:
            group_alive = False
        if group_alive:
            os.killpg(pgid, signal.SIGKILL)
            time.sleep(0.2)
    except (OSError, psutil.Error) as exc:
        return False, f"Service stop failed: {exc}"
    _clear_state()
    _log("Detection service stopped")
    return True, "Service stopped"


class ServiceManagerHandler(http.server.BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        # Mute standard access log output to keep console clean
        pass

    def _write_json(self, status, payload):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())

    def do_GET(self):
        if self.path == '/status':
            process = _owned_process()
            started_dt = get_process_start_time(process.pid) if process else None
            api = _health_api_status() if process else {"state": "offline"}
            storage = _storage_status(process)
            tunnel = _tunnel_status(process)
            metrics = get_system_metrics()

            res = {
                "status": "running" if process else "stopped",
                "pid": process.pid if process else None,
                "uptime_seconds": int((datetime.datetime.now() - started_dt).total_seconds()) if started_dt else 0,
                "started_at": started_dt.isoformat() if started_dt else None,
                "storage": storage,
                "storage_healthy": storage.get("state") == "healthy",
                "tunnel": tunnel,
                "tunnel_online": tunnel.get("state") == "healthy",
                "api": api,
                "api_healthy": api.get("state") == "healthy",
                **metrics
            }
            self._write_json(200, res)

        elif self.path.startswith('/logs/'):
            # Filtered logs for a specific session
            report_id = urllib.parse.unquote(self.path.split('/')[-1])
            filtered_lines = []
            if LOG_FILE.exists():
                try:
                    with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
                        all_lines = f.readlines()

                        in_session = False
                        for line in all_lines:
                            clean_line = line.rstrip('\r\n').strip()
                            if f"[Session] opened: {report_id}" in clean_line:
                                in_session = True
                                filtered_lines.append(clean_line)
                                continue
                            if f"[Session] closed: {report_id}" in clean_line:
                                filtered_lines.append(clean_line)
                                in_session = False
                                continue

                            if in_session:
                                filtered_lines.append(clean_line)
                            elif f"report_id: {report_id}" in clean_line or f"report_id={report_id}" in clean_line:
                                filtered_lines.append(clean_line)

                    filtered_lines = filtered_lines[-150:]
                except Exception as e:
                    filtered_lines = [f"Error reading logs: {str(e)}"]

            self._write_json(200, {"lines": filtered_lines, "session_id": report_id})

        elif self.path == '/logs':
            lines = []
            if LOG_FILE.exists():
                try:
                    with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
                        all_lines = f.readlines()
                        lines = [line.rstrip('\r\n') for line in all_lines[-250:]]
                except Exception as e:
                    lines = [f"Error reading logs: {str(e)}"]
            self._write_json(200, {"lines": lines, "session_id": _read_state().get("session_id")})

        elif self.path == '/cameras':
            cameras = []
            import glob
            dev_videos = sorted(glob.glob("/dev/video*"))
            for dev in dev_videos:
                cameras.append({"id": dev, "name": f"Camera ({dev.split('/')[-1]})"})

            v4l_path = "/dev/v4l/by-id/*"
            v4l_devs = sorted(glob.glob(v4l_path))
            for dev in v4l_devs:
                cameras.append({"id": dev, "name": dev.split('/')[-1]})

            if not cameras:
                # Common fallbacks so UI is not empty
                cameras = [
                    {"id": "0", "name": "Default Webcam (index 0)"},
                    {"id": "/dev/video0", "name": "/dev/video0"},
                ]

            self._write_json(200, {"cameras": cameras})

        elif self.path.startswith('/session/'):
            report_id = urllib.parse.unquote(self.path.split('/')[-1])
            info = {"report_id": report_id, "exists": False, "camera_source": None,
                    "production_line_id": None, "started_at": None, "telemetry": None,
                    "control": None, "config": None}
            try:
                req = urllib.request.Request(f"{API_BASE}/api/reports")
                with urllib.request.urlopen(req, timeout=2) as response:
                    reports = json.loads(response.read().decode())
                found = next((r for r in reports if r["report_id"] == report_id), None)
                if found:
                    info["exists"] = True
                    info["viewers_count"] = found.get("viewers_count", 0)
            except Exception:
                pass
            self._write_json(200, info)

        else:
            self._write_json(404, {"error": "Not found"})

    def _ensure_log_file(self):
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not LOG_FILE.exists():
            LOG_FILE.touch()

    def do_POST(self):
        try:
            if self.path == '/start':
                process, message = _start_service()
                payload = {"status": "success", "message": message, "pid": process.pid}
            elif self.path == '/stop':
                success, message = _stop_service()
                payload = {"status": "success" if success else "error", "message": message}
            elif self.path == '/restart':
                stopped, stop_message = _stop_service()
                if not stopped and get_service_pid() is not None:
                    payload = {"status": "error", "message": f"Restart aborted: {stop_message}"}
                    self._write_json(200, payload)
                    return
                process, _ = _start_service()
                payload = {"status": "success", "message": "Service restart initiated", "pid": process.pid}
            elif self.path == '/clear_logs':
                LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
                LOG_FILE.write_text("", encoding="utf-8")
                payload = {"status": "success", "message": "Logs cleared"}
            else:
                self._write_json(404, {"error": "Not found"})
                return
        except Exception as exc:
            _log(f"Manager operation failed: {exc}", "ERROR")
            payload = {"status": "error", "message": str(exc)}
        self._write_json(200, payload)


if __name__ == '__main__':
    print(f"[ServiceManager] Project root : {PROJECT_ROOT}")
    print(f"[ServiceManager] Runtime dir   : {RUNTIME_DIR}")
    print(f"[ServiceManager] Main script   : {MAIN_PY}")
    print(f"[ServiceManager] Log file      : {LOG_FILE}")
    print(f"[ServiceManager] API base      : {API_BASE}")
    print(f"[ServiceManager] Starting on port {PORT}...")
    server = http.server.HTTPServer(('0.0.0.0', PORT), ServiceManagerHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
