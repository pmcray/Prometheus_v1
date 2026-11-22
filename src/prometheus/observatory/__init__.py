"""Observatory dashboard for Prometheus v1.

Runs locally on the Chief Scientist's workstation (read-only over the governor
home and synced Google Drive exchange folder). No network exposure by default
(binds strictly to 127.0.0.1) and zero external runtime dependencies.
"""

from .collector import collect_snapshot
from .server import render_dashboard_html, serve

__all__ = ["collect_snapshot", "render_dashboard_html", "serve"]
