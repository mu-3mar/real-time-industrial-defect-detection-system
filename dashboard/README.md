# Dashboard service manager

The dashboard talks to `service_manager.py` on `127.0.0.1:8080`. The manager
starts `src/qc_scm/runtime/main.py` in its own process group and persists the PID,
creation time, process group, and log session ID in
`dashboard/service_manager_state.json`. Stop and restart validate that identity
before sending signals; they do not use port-wide or command-line-wide kills.

## Enable automatic startup on Linux

Run this once from the project checkout:

```bash
./dashboard/install_service_manager.sh
```

The installer creates and enables a per-user `systemd` service with the current
project path and Python interpreter. Set `PYTHON_BIN` to select an interpreter,
and set `QC_SCM_TUNNEL_URL` before running the installer when a tunnel should be
health-checked. The manager itself starts at login; the detection service starts
only when the dashboard requests it.

Useful commands:

```bash
systemctl --user status qc-scm-service-manager.service
systemctl --user restart qc-scm-service-manager.service
```

## Edit the detection region during inference

Open an active report from the dashboard. Press and drag on the live camera
stream to draw a rectangular detection area. The orange rectangle can be moved
from its interior and resized from its four rounded corner handles. The area
inside the rectangle remains clear and is passed to the detector; the area
outside it is dimmed to show that it is disabled.

Changes are applied immediately to that same session through the runtime ROI
API and are not written to the shared YAML configuration. Updating the
rectangle clears the previous tracking result so the next detector pass uses
the new crop. Detection totals, defect/OK counts, and processing FPS appear
directly below the stream.