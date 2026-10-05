"""Interactive Dashboard Runner for LoRA vs LoRA+ Benchmarking.

Serves the Apache ECharts interactive research dashboard on localhost and opens it in the browser.
"""

import argparse
import http.server
import json
import os
import shutil
import socketserver
import sys
import threading
import webbrowser
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Enforce all caches and temp files strictly on D: drive
os.environ["TEMP"] = r"D:\Lora++\.cache\temp"
os.environ["TMP"] = r"D:\Lora++\.cache\temp"
os.environ["TMPDIR"] = r"D:\Lora++\.cache\temp"
os.environ["HF_HOME"] = r"D:\Lora++\.cache\huggingface"
os.environ["TRANSFORMERS_CACHE"] = r"D:\Lora++\.cache\huggingface"
os.environ["TORCH_HOME"] = r"D:\Lora++\.cache\torch"
os.environ["PYTHONPYCACHEPREFIX"] = r"D:\Lora++\.cache\pycache"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.echarts_dashboard import generate_echarts_dashboard


def find_available_port(start_port: int = 8080, max_attempts: int = 20) -> int:
    """Finds an available TCP port starting from start_port."""
    import socket
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start_port


def prepare_dashboard() -> Path:
    """Ensures dashboard.html and index.html are up-to-date."""
    results_path = PROJECT_ROOT / "experiments" / "results" / "comparison_metrics.json"
    plots_dir = PROJECT_ROOT / "experiments" / "plots"
    dashboard_file = plots_dir / "dashboard.html"
    index_file = plots_dir / "index.html"

    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        generate_echarts_dashboard(data, output_dir=str(plots_dir))
    
    if dashboard_file.exists():
        shutil.copyfile(dashboard_file, index_file)

    return plots_dir


def run_server(port: int = 8080, open_browser: bool = True, plots_dir: Path = None):
    """Starts the HTTP server serving the dashboard directory."""
    if plots_dir is None:
        plots_dir = prepare_dashboard()

    port = find_available_port(port)
    url = f"http://localhost:{port}/"

    class DashboardHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(plots_dir), **kwargs)

        def log_message(self, format, *args):
            # Suppress noisy access logs
            pass

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), DashboardHandler) as httpd:
        print("=" * 65)
        print("[*] LoRA+ Empirical Benchmark Dashboard")
        print(f"[*] Serving interactive ECharts dashboard at: {url}")
        print(f"[*] Serving directory: {plots_dir}")
        print("=" * 65)

        if open_browser:
            print(f"[*] Opening {url} in your default browser...")
            threading.Timer(0.6, lambda: webbrowser.open(url)).start()

        print("Press Ctrl+C to stop the dashboard server.\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nDashboard server stopped.")


def main():
    parser = argparse.ArgumentParser(description="Run the interactive LoRA+ research dashboard")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve dashboard on (default: 8080)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open default browser")
    args = parser.parse_args()

    plots_dir = prepare_dashboard()
    run_server(port=args.port, open_browser=not args.no_browser, plots_dir=plots_dir)


if __name__ == "__main__":
    main()
