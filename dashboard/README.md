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

Open an active report from the dashboard and use the orange handles over the
full-width live camera stream. Drag the two vertical lines to change the left
and right limits, or drag the horizontal line to change the top limit. Changes
are applied to that same session immediately through the runtime ROI API and
are not written to the shared YAML configuration. Updating a boundary clears
the previous tracking result so the next detector pass uses the new crop.
Detection totals, defect/OK counts, and processing FPS appear directly below
the stream.