# Dashboard service manager

The dashboard talks to `service_manager.py` on `127.0.0.1:8080`. The manager
starts `Boxes/flow/main.py` in its own process group and persists the PID,
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