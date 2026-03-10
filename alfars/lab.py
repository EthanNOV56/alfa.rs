"""
alfars lab - Interactive lab environment for factor research.

This module starts both the backend server and frontend dev server,
providing a Jupyter Lab-like experience for factor research.
"""

import subprocess
import webbrowser
import time
import sys
import os
import signal
import atexit
from typing import Optional, List

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
BACKEND_PORT = 8000
FRONTEND_PORT = 5173


def find_free_port(start_port: int = 8000) -> int:
    """Find a free port starting from start_port."""
    import socket
    port = start_port
    while port < start_port + 100:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            port += 1
    return start_port


def check_cargo_available() -> bool:
    """Check if cargo is available."""
    try:
        subprocess.run(
            ["cargo", "--version"],
            capture_output=True,
            check=True,
            cwd=PROJECT_ROOT
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def start_backend(backend_type: str = "auto") -> Optional[subprocess.Popen]:
    """
    Start the backend server.

    Parameters
    ----------
    backend_type : str
        "rust" for Rust server, "python" for FastAPI server, "auto" to detect

    Returns
    -------
    subprocess.Popen or None
    """
    if backend_type == "auto":
        backend_type = "rust" if check_cargo_available() else "python"

    print(f"[alfars lab] Starting backend ({backend_type})...")

    if backend_type == "rust":
        # Start Rust server
        process = subprocess.Popen(
            ["cargo", "run", "--release", "--bin", "alfars-server"],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid if sys.platform != "win32" else None,
        )
        print(f"[alfars lab] Rust backend started (PID: {process.pid})")
        return process
    else:
        # Start Python FastAPI server
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "alfars.server:app",
             "--host", "0.0.0.0", "--port", str(BACKEND_PORT)],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid if sys.platform != "win32" else None,
        )
        print(f"[alfars lab] Python backend started (PID: {process.pid})")
        return process


def start_frontend() -> Optional[subprocess.Popen]:
    """
    Start the frontend dev server.

    Returns
    -------
    subprocess.Popen or None
    """
    # Check if node_modules exists
    if not os.path.exists(os.path.join(FRONTEND_DIR, "node_modules")):
        print("[alfars lab] Installing frontend dependencies...")
        subprocess.run(
            ["npm", "install"],
            cwd=FRONTEND_DIR,
            check=True,
        )

    print("[alfars lab] Starting frontend dev server...")

    process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=FRONTEND_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid if sys.platform != "win32" else None,
    )

    print(f"[alfars lab] Frontend started (PID: {process.pid})")
    return process


def wait_for_server(port: int, timeout: int = 30) -> bool:
    """Wait for a server to become available."""
    import socket
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.connect(('127.0.0.1', port))
                return True
        except (socket.error, socket.timeout):
            time.sleep(0.5)
    return False


def cleanup_processes(processes: List[subprocess.Popen]) -> None:
    """Terminate all child processes."""
    for process in processes:
        if process and process.poll() is None:
            try:
                if sys.platform == "win32":
                    process.terminate()
                else:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=5)
            except Exception as e:
                print(f"[alfars lab] Warning: Error cleaning up process: {e}")
                try:
                    process.kill()
                except Exception:
                    pass


def print_output(process: subprocess.Popen, name: str) -> None:
    """Print process output in a separate thread."""
    import threading

    def _print():
        try:
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(f"[{name}] {line.rstrip()}")
                if process.poll() is not None:
                    break
        except Exception:
            pass

    thread = threading.Thread(target=_print, daemon=True)
    thread.start()


def main(backend_type: str = "auto", open_browser: bool = True, wait: bool = True):
    """
    Start the alfars lab environment.

    Parameters
    ----------
    backend_type : str
        Backend type: "rust", "python", or "auto" (default: "auto")
    open_browser : bool
        Whether to open browser automatically (default: True)
    wait : bool
        Whether to wait for processes (default: True)
    """
    print("=" * 60)
    print("  alfars lab - Interactive Factor Research Environment")
    print("=" * 60)
    print()

    processes: List[subprocess.Popen] = []

    try:
        # Start backend
        backend_process = start_backend(backend_type)
        if backend_process:
            processes.append(backend_process)
            print_output(backend_process, "backend")

        # Start frontend
        frontend_process = start_frontend()
        if frontend_process:
            processes.append(frontend_process)
            print_output(frontend_process, "frontend")

        # Wait for servers to be ready
        print("[alfars lab] Waiting for servers to start...")

        backend_ready = wait_for_server(BACKEND_PORT, timeout=30)
        frontend_ready = wait_for_server(FRONTEND_PORT, timeout=60)

        if not backend_ready:
            print("[alfars lab] Warning: Backend may not be ready")
        if not frontend_ready:
            print("[alfars lab] Warning: Frontend may not be ready")

        print()
        print("-" * 60)
        print("  Services started successfully!")
        print(f"  Backend API:  http://localhost:{BACKEND_PORT}")
        print(f"  Frontend:    http://localhost:{FRONTEND_PORT}")
        print("-" * 60)
        print()

        # Open browser
        if open_browser:
            time.sleep(1)  # Brief pause before opening browser
            webbrowser.open(f"http://localhost:{FRONTEND_PORT}")

        # Register cleanup handlers
        atexit.register(lambda: cleanup_processes(processes))

        def signal_handler(signum, frame):
            print("\n[alfars lab] Shutting down...")
            cleanup_processes(processes)
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Wait for processes
        if wait:
            print("[alfars lab] Press Ctrl+C to stop all services")
            while processes:
                # Check if any process died
                for i, p in enumerate(processes):
                    if p.poll() is not None:
                        print(f"[alfars lab] Process {i} died, shutting down...")
                        cleanup_processes(processes)
                        return
                time.sleep(1)

    except KeyboardInterrupt:
        print("\n[alfars lab] Interrupted, shutting down...")
        cleanup_processes(processes)
    except Exception as e:
        print(f"[alfars lab] Error: {e}")
        cleanup_processes(processes)
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Start alfars lab environment"
    )
    parser.add_argument(
        "--backend",
        choices=["rust", "python", "auto"],
        default="auto",
        help="Backend type (default: auto)"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically"
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for processes"
    )

    args = parser.parse_args()

    main(
        backend_type=args.backend,
        open_browser=not args.no_browser,
        wait=not args.no_wait
    )
