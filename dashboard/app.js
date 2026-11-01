const API_BASE = "http://localhost:8000";
const MANAGER_API = "http://localhost:8080";
const PUBLIC_ACCESS_URL = "https://muhammad-ammar-dell-g15-5510.tail5a5979.ts.net";
const serviceUrl = PUBLIC_ACCESS_URL;

let startTime = Date.now();
let serviceState = "stopped";
let isLogPaused = false;
let lastLogsJSON = "";
let servicePid = "--";
let uptimeInterval = null;
let lastStatusKnown = null;
let logRequestGeneration = 0;
let statusRequestInFlight = false;

const reportsTable = document.getElementById("reportsTable");
const eventsContainer = document.getElementById("events");

// --- CLOCK ---
function updateClock() {
    const now = new Date();
    const timeEl = document.getElementById("time");
    const dateEl = document.getElementById("date");
    if (timeEl) timeEl.textContent = now.toLocaleTimeString();
    if (dateEl) dateEl.textContent = now.toLocaleDateString();
}
setInterval(updateClock, 1000);
updateClock();

// --- UPTIME ---
function updateUptime() {
    if (serviceState !== "running") return;
    const diff = Math.floor((Date.now() - startTime) / 1000);
    const h = String(Math.floor(diff / 3600)).padStart(2, "0");
    const m = String(Math.floor(diff % 3600 / 60)).padStart(2, "0");
    const s = String(diff % 60).padStart(2, "0");
    const uptimeEl = document.getElementById("uptime");
    if (uptimeEl) uptimeEl.textContent = `${h}:${m}:${s}`;
}
uptimeInterval = setInterval(updateUptime, 1000);

// --- LOGGING (UI event stream: user actions + errors) ---
function addEvent(message, type = "info") {
    if (isLogPaused && !message.includes("paused") && !message.includes("resumed")) return;
    if (!eventsContainer) return;

    const line = document.createElement("div");
    line.className = "log-line";

    const now = new Date();
    const timeStr = now.toTimeString().split(' ')[0];

    let textClass = "log-text";
    if (type === "success") textClass += " log-green";
    if (type === "link") textClass += " log-link";
    if (type === "error") textClass += " log-green";

    const time = document.createElement("span");
    time.className = "log-time";
    time.textContent = timeStr;
    const text = document.createElement("span");
    text.className = textClass;
    text.textContent = message;
    line.append(time, text);

    eventsContainer.prepend(line);

    const lines = eventsContainer.querySelectorAll(".log-line");
    if (lines.length > 25) {
        lines[lines.length - 1].remove();
    }
}

function renderComponentStatus(elementId, component) {
    const element = document.getElementById(elementId);
    if (!element) return;
    const state = component?.state || "unknown";
    const labels = {
        healthy: ["Connected", "var(--green)"],
        inactive: ["Inactive", "#f59e0b"],
        not_configured: ["Not Configured", "#f59e0b"],
        unavailable: ["Unavailable", "#f59e0b"],
        offline: ["Offline", "var(--red)"],
        error: ["Error", "var(--red)"],
    };
    const [label, color] = labels[state] || ["Unknown", "#f59e0b"];
    element.innerHTML = `<span class="dot" style="background:${color};"></span>${label}`;
    element.style.color = color;
}

// --- STATE MANAGEMENT ---
function updateStatusUI(state) {
    serviceState = state;
    const statusPill = document.getElementById("serviceStatusPill");
    const storageStatus = document.getElementById("storageStatus");
    const tunnelStatus = document.getElementById("tunnelStatus");
    const apiStatus = document.getElementById("apiStatus");
    const healthBox = document.getElementById("healthBox");
    const healthIcon = document.getElementById("healthIcon");
    const healthTitle = document.getElementById("healthTitle");
    const healthSub = document.getElementById("healthSub");
    const pidVal = document.getElementById("servicePid");
    const networkStatus = document.getElementById("networkStatus");

    if (state === "running") {
        if (statusPill) {
            statusPill.innerHTML = '<span class="dot" style="background:var(--green);"></span>Running';
            statusPill.style.background = 'var(--green-light)';
            statusPill.style.color = 'var(--green-dark)';
        }
        if (healthBox) {
            healthBox.style.background = 'var(--green-pale)';
            healthBox.style.borderColor = '#cdeede';
        }
        if (healthIcon) {
            healthIcon.style.background = 'var(--green)';
            healthIcon.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><path d="M20 6L9 17l-5-5"/></svg>';
        }
        if (healthTitle) healthTitle.textContent = "Healthy";
        if (healthSub) healthSub.textContent = "All systems operational";
        if (pidVal) pidVal.innerHTML = `PID: <b>${servicePid}</b>`;
        if (networkStatus) {
            networkStatus.textContent = "Connected";
            networkStatus.style.color = "var(--green-dark)";
        }
    } else if (state === "stopped") {
        if (statusPill) {
            statusPill.innerHTML = '<span class="dot" style="background:var(--red);"></span>Stopped';
            statusPill.style.background = 'var(--red-bg)';
            statusPill.style.color = 'var(--red)';
        }
        if (storageStatus) storageStatus.innerHTML = '<span class="dot" style="background:var(--red);"></span>Offline';
        if (tunnelStatus) tunnelStatus.innerHTML = '<span class="dot" style="background:var(--red);"></span>Not Configured';
        if (apiStatus) apiStatus.innerHTML = '<span class="dot" style="background:var(--red);"></span>Offline';
        if (healthBox) {
            healthBox.style.background = 'var(--red-bg)';
            healthBox.style.borderColor = '#fbcfe8';
        }
        if (healthIcon) {
            healthIcon.style.background = 'var(--red)';
            healthIcon.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><path d="M18 6L6 18M6 6l12 12"/></svg>';
        }
        if (healthTitle) healthTitle.textContent = "Offline";
        if (healthSub) healthSub.textContent = "Service is not running";
        if (pidVal) pidVal.innerHTML = 'PID: <b>--</b>';
        if (networkStatus) {
            networkStatus.textContent = "Disconnected";
            networkStatus.style.color = "var(--red)";
        }

        updateUsageRing("cpu", 0);
        updateUsageRing("ram", 0);
        updateUsageRing("disk", 0);
        updateUsageRing("gpu", 0);
    } else if (state === "restarting") {
        if (statusPill) {
            statusPill.innerHTML = '<span class="dot" style="background:#f59e0b;"></span>Restarting';
            statusPill.style.background = '#fef3c7';
            statusPill.style.color = '#b45309';
        }
        if (healthTitle) healthTitle.textContent = "Restarting...";
        if (healthSub) healthSub.textContent = "Performing self-checks";
    } else if (state === "manager_offline") {
        if (statusPill) {
            statusPill.innerHTML = '<span class="dot" style="background:#f59e0b;"></span>Manager Offline';
            statusPill.style.background = '#fef3c7';
            statusPill.style.color = '#b45309';
        }
        if (healthTitle) healthTitle.textContent = "Unknown";
        if (healthSub) healthSub.textContent = "Service Manager is not running on port 8080";
    }
}

// --- CALLS TO DETECTION API ---
async function loadHealth() {
    const healthBox = document.getElementById("healthBox");
    const healthIcon = document.getElementById("healthIcon");
    const healthTitle = document.getElementById("healthTitle");
    const healthSub = document.getElementById("healthSub");
    const activeCount = document.getElementById("activeReportsCount");
    const apiStatus = document.getElementById("apiStatus");
    const storageStatus = document.getElementById("storageStatus");
    const tunnelStatus = document.getElementById("tunnelStatus");

    if (serviceState !== "running") {
        if (activeCount) activeCount.textContent = "0";
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/api/health`);
        if (!response.ok) throw new Error("Health check API returned error");
        const data = await response.json();

        if (healthBox) {
            healthBox.style.background = 'var(--green-pale)';
            healthBox.style.borderColor = '#cdeede';
        }
        if (healthIcon) {
            healthIcon.style.background = 'var(--green)';
            healthIcon.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><path d="M20 6L9 17l-5-5"/></svg>';
        }
        if (healthTitle) {
            healthTitle.textContent = data.status ? (data.status.charAt(0).toUpperCase() + data.status.slice(1)) : "Healthy";
        }
        if (healthSub) healthSub.textContent = "All systems operational";
        if (activeCount) activeCount.textContent = data.active_reports !== undefined ? data.active_reports : "0";
        if (apiStatus) apiStatus.innerHTML = '<span class="dot" style="background:var(--green);"></span>Healthy';
    } catch (err) {
        if (healthBox) {
            healthBox.style.background = 'var(--red-bg)';
            healthBox.style.borderColor = '#fbcfe8';
        }
        if (healthIcon) {
            healthIcon.style.background = 'var(--red)';
            healthIcon.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><path d="M18 6L6 18M6 6l12 12"/></svg>';
        }
        if (healthTitle) healthTitle.textContent = "Unhealthy";
        if (healthSub) healthSub.textContent = "Detection API not responding";
        if (activeCount) activeCount.textContent = "0";
        if (apiStatus) apiStatus.innerHTML = '<span class="dot" style="background:var(--red);"></span>Offline';
        if (storageStatus) storageStatus.innerHTML = '<span class="dot" style="background:var(--red);"></span>Offline';
        if (tunnelStatus) tunnelStatus.innerHTML = '<span class="dot" style="background:var(--red);"></span>Not Configured';
    }
}

async function loadReports() {
    const reportsTable = document.getElementById("reportsTable");
    const activeCount = document.getElementById("activeReportsCount");

    if (serviceState !== "running") {
        if (reportsTable) {
            reportsTable.innerHTML = `
                <tr>
                    <td colspan="3" style="text-align: center; color: var(--text-sub); padding: 20px 0;">Service not running</td>
                </tr>
            `;
        }
        if (activeCount) activeCount.textContent = "0";
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/api/reports`);
        if (!response.ok) throw new Error("Reports API returned error");
        const reports = await response.json();

        if (reportsTable) {
            reportsTable.innerHTML = "";
            if (reports.length === 0) {
                reportsTable.innerHTML = `
                    <tr>
                        <td colspan="3" style="text-align: center; color: var(--text-sub); padding: 20px 0;">No active reports</td>
                    </tr>
                `;
            } else {
                reports.forEach(report => {
                    const row = document.createElement("tr");
                    row.innerHTML = `
                        <td style="cursor:pointer;">${report.report_id}</td>
                        <td style="cursor:pointer;"><span class="status-active"><span class="dot"></span>Active</span></td>
                        <td style="text-align:right;">
                            <button class="action-btn close-report-btn" data-id="${report.report_id}" title="Close Session">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <line x1="18" y1="6" x2="6" y2="18"></line>
                                    <line x1="6" y1="6" x2="18" y2="18"></line>
                                </svg>
                            </button>
                        </td>
                    `;

                    const cells = row.querySelectorAll("td:not(:last-child)");
                    cells.forEach(cell => {
                        cell.addEventListener("click", () => {
                            window.location.href = `session.html?report_id=${encodeURIComponent(report.report_id)}`;
                        });
                    });

                    const closeBtn = row.querySelector(".close-report-btn");
                    closeBtn.addEventListener("click", async (e) => {
                        e.stopPropagation();
                        const reportId = closeBtn.getAttribute("data-id");
                        if (confirm(`Are you sure you want to close session ${reportId}?`)) {
                            await closeReport(reportId);
                        }
                    });

                    reportsTable.appendChild(row);
                });
            }
        }
        if (activeCount) activeCount.textContent = reports.length;
    } catch (err) {
        if (reportsTable) {
            reportsTable.innerHTML = `
                <tr>
                    <td colspan="3" style="text-align: center; color: var(--text-sub); padding: 20px 0;">No active reports</td>
                </tr>
            `;
        }
        if (activeCount) activeCount.textContent = "0";
    }
}

async function closeReport(reportId) {
    try {
        const response = await fetch(`${API_BASE}/api/reports/close`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ report_id: reportId })
        });
        if (response.ok) {
            showNotification(`Session ${reportId} closed successfully`);
            addEvent(`Session ${reportId} closed`, "success");
            loadReports();
            loadHealth();
        } else {
            showNotification(`Failed to close session ${reportId}`, "error");
        }
    } catch (err) {
        showNotification("Error connecting to Detection API", "error");
    }
}

// --- MODAL & FORM ---
const modalOverlay = document.getElementById("modalOverlay");
const openModalBtn = document.getElementById("openModalBtn");
const closeModalBtn = document.getElementById("closeModalBtn");
const cancelModalBtn = document.getElementById("cancelModalBtn");
const createSessionForm = document.getElementById("createSessionForm");

async function loadCameras() {
    const cameraSelect = document.getElementById("cameraSource");
    if (!cameraSelect) return;

    try {
        const response = await fetch(`${MANAGER_API}/cameras`);
        const data = await response.json();

        const firstOption = cameraSelect.options[0];
        cameraSelect.innerHTML = "";
        cameraSelect.appendChild(firstOption);

        if (data.cameras && data.cameras.length > 0) {
            data.cameras.forEach(cam => {
                const opt = document.createElement("option");
                opt.value = cam.id;
                opt.textContent = cam.name;
                cameraSelect.appendChild(opt);
            });
        } else {
            const opt = document.createElement("option");
            opt.disabled = true;
            opt.textContent = "No cameras discovered";
            cameraSelect.appendChild(opt);
        }
    } catch (err) {
        addEvent("Could not reach Service Manager for camera list. Check that service_manager.py is running.", "info");
    }
}

if (openModalBtn) {
    openModalBtn.addEventListener("click", () => {
        if (serviceState !== "running") {
            showNotification("Start the Detection Service first", "error");
            addEvent("Cannot create session: Detection Service is not running", "error");
            return;
        }
        modalOverlay.style.display = "flex";
        loadCameras();
    });
}

function closeModal() {
    if (modalOverlay) modalOverlay.style.display = "none";
    if (createSessionForm) createSessionForm.reset();
}

if (closeModalBtn) closeModalBtn.addEventListener("click", closeModal);
if (cancelModalBtn) cancelModalBtn.addEventListener("click", closeModal);

if (modalOverlay) {
    modalOverlay.addEventListener("click", (e) => {
        if (e.target === modalOverlay) closeModal();
    });
}

if (createSessionForm) {
    createSessionForm.addEventListener("submit", async (e) => {
        e.preventDefault();

        const payload = {
            report_id: document.getElementById("reportId").value,
            camera_source: document.getElementById("cameraSource").value,
            production_line_id: document.getElementById("lineId").value || "default-line",
            target_speed: parseInt(document.getElementById("targetSpeed").value),
            max_temp: parseInt(document.getElementById("maxTemp").value),
            max_amps: parseInt(document.getElementById("maxAmps").value)
        };

        try {
            addEvent(`Opening session ${payload.report_id} on camera ${payload.camera_source}...`, "info");
            const response = await fetch(`${API_BASE}/api/reports/open`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (response.ok) {
                showNotification(`Session ${payload.report_id} created successfully`);
                addEvent(`[Session] opened: ${payload.report_id}`, "success");
                closeModal();
                loadReports();
                loadHealth();
            } else {
                const data = await response.json();
                showNotification(`Failed: ${data.detail || "Unknown error"}`, "error");
                addEvent(`Failed to open session: ${data.detail || "Unknown error"}`, "info");
            }
        } catch (err) {
            showNotification("Error connecting to Detection API", "error");
            addEvent("Connection error during session open", "info");
        }
    });
}

// --- NOTIFICATIONS ---
function showNotification(message, type = "success") {
    const note = document.getElementById("notification");
    if (!note) return;

    note.textContent = message;
    note.style.display = "block";
    note.style.background = type === "success" ? "var(--green)" : "var(--red)";

    setTimeout(() => {
        note.style.display = "none";
    }, 3000);
}

// --- CONTROLS LISTENERS ---
const startBtn = document.querySelector(".start-btn");
const restartBtn = document.querySelector(".restart-btn");
const stopBtn = document.querySelector(".stop-btn");

if (startBtn) {
    startBtn.addEventListener("click", async () => {
        if (serviceState === "running") {
            addEvent("Detection Service is already running", "info");
            return;
        }
        addEvent("Start Detection Service requested...", "info");
        try {
            updateStatusUI("restarting");
            const res = await fetch(`${MANAGER_API}/start`, { method: "POST" });
            const data = await res.json();
            if (data.status === "success") {
                addEvent("Start command sent successfully. Waiting for service to initialize...", "success");
            } else {
                addEvent(`Failed to start: ${data.message}`, "info");
                updateStatusUI("stopped");
            }
        } catch (err) {
            addEvent("Could not connect to Service Manager on port 8080. Run: python3 dashboard/service_manager.py", "info");
            updateStatusUI("manager_offline");
        }
    });
}

if (restartBtn) {
    restartBtn.addEventListener("click", async () => {
        addEvent("Restarting Detection Service...", "info");
        updateStatusUI("restarting");
        try {
            const res = await fetch(`${MANAGER_API}/restart`, { method: "POST" });
            const data = await res.json();
            if (data.status === "success") {
                addEvent("Restart command sent successfully.", "success");
            } else {
                addEvent(`Failed to restart: ${data.message}`, "info");
                updateStatusUI("stopped");
            }
        } catch (err) {
            addEvent("Could not connect to Service Manager on port 8080.", "info");
            updateStatusUI("manager_offline");
        }
    });
}

if (stopBtn) {
    stopBtn.addEventListener("click", async () => {
        if (serviceState === "stopped") {
            addEvent("Detection Service is already stopped", "info");
            return;
        }
        addEvent("Stop Detection Service requested...", "info");
        try {
            const res = await fetch(`${MANAGER_API}/stop`, { method: "POST" });
            const data = await res.json();
            if (data.status === "success") {
                updateStatusUI("stopped");
                addEvent("Detection Service stopped", "success");
            } else {
                addEvent(`Failed to stop: ${data.message}`, "info");
            }
        } catch (err) {
            addEvent("Could not connect to Service Manager on port 8080.", "info");
            updateStatusUI("manager_offline");
        }
    });
}

// --- AUTO-CHECK STATUS (polls Service Manager) ---
async function checkServiceStatus() {
    if (statusRequestInFlight) return;
    statusRequestInFlight = true;
    try {
        const response = await fetch(`${MANAGER_API}/status`);
        const data = await response.json();
        const detectedState = data.status;

        const storageStatus = document.getElementById("storageStatus");
        const tunnelStatus = document.getElementById("tunnelStatus");
        const apiStatus = document.getElementById("apiStatus");

        renderComponentStatus("storageStatus", data.storage || {
            state: data.storage_healthy ? "healthy" : "offline"
        });
        renderComponentStatus("tunnelStatus", data.tunnel || {
            state: data.tunnel_online ? "healthy" : "not_configured"
        });
        renderComponentStatus("apiStatus", data.api || {
            state: data.api_healthy ? "healthy" : "offline"
        });

        if (data.cpu_percent !== undefined) updateUsageRing("cpu", data.cpu_percent);
        if (data.ram_percent !== undefined) updateUsageRing("ram", data.ram_percent);
        if (data.disk_percent !== undefined) updateUsageRing("disk", data.disk_percent);
        if (data.gpu_percent !== undefined) updateUsageRing("gpu", data.gpu_percent);

        const networkStatus = document.getElementById("networkStatus");
        const networkSub = document.getElementById("networkSub");
        if (networkStatus && data.network_status) {
            networkStatus.textContent = data.network_status;
            networkStatus.style.color = data.network_status === "Connected" ? "var(--green-dark)" : "var(--red)";
        }
        if (networkSub && data.network_speed_kb !== undefined) {
            networkSub.textContent = `Speed: ${data.network_speed_kb} KB/s`;
        }
        const bars = document.querySelectorAll(".bars .bar");
        if (bars.length > 0) {
            const speed = data.network_speed_kb || 0;
            bars.forEach((bar, idx) => {
                const factor = (idx + 1) * 0.25;
                const h = Math.min(100, Math.max(10, Math.floor((speed * factor / 100) * 80) + 20));
                bar.style.height = `${h}%`;
            });
        }

        if (detectedState === "running") {
            servicePid = data.pid || "--";
            if (data.uptime_seconds !== undefined) {
                startTime = Date.now() - (data.uptime_seconds * 1000);
            }
            const pidVal = document.getElementById("servicePid");
            if (pidVal) pidVal.innerHTML = `PID: <b>${servicePid}</b>`;

            if (data.started_at) {
                const startedDate = new Date(data.started_at);
                const timeEl = document.getElementById("lastStartedTime");
                const dateEl = document.getElementById("lastStartedDate");
                if (timeEl) timeEl.textContent = startedDate.toLocaleTimeString();
                if (dateEl) dateEl.textContent = startedDate.toLocaleDateString();
            }
        } else {
            servicePid = "--";
            const pidVal = document.getElementById("servicePid");
            if (pidVal) pidVal.innerHTML = 'PID: <b>--</b>';
            const timeEl = document.getElementById("lastStartedTime");
            const dateEl = document.getElementById("lastStartedDate");
            if (timeEl) timeEl.textContent = "--";
            if (dateEl) dateEl.textContent = "--";
            const uptimeEl = document.getElementById("uptime");
            if (uptimeEl) uptimeEl.textContent = "00:00:00";

            updateUsageRing("cpu", 0);
            updateUsageRing("ram", 0);
            updateUsageRing("disk", 0);
            updateUsageRing("gpu", 0);
            if (bars.length > 0) {
                bars.forEach(bar => { bar.style.height = "10%"; });
            }
        }

        if (serviceState !== "restarting") {
            if (detectedState !== serviceState && detectedState !== lastStatusKnown) {
                updateStatusUI(detectedState);
                addEvent(`Detection Service status: ${detectedState}`, detectedState === "running" ? "success" : "info");
            }
        } else {
            if (detectedState === "running") {
                updateStatusUI("running");
                addEvent("Detection Service started successfully", "success");
            }
        }
        lastStatusKnown = detectedState;
    } catch (err) {
        // Manager offline
        if (serviceState !== "restarting" && serviceState !== "manager_offline") {
            updateStatusUI("manager_offline");
            addEvent("Service Manager unreachable (port 8080). Start with: python3 dashboard/service_manager.py", "info");
        }
    } finally {
        statusRequestInFlight = false;
    }
}
setInterval(checkServiceStatus, 2000);

// --- PUBLIC URL / ACCESS BUTTONS ---
const copyBtn = document.getElementById("copyBtn");
const openBtn = document.getElementById("openBtn");
const docsBtn = document.getElementById("docsBtn");

if (copyBtn) {
    copyBtn.addEventListener("click", async () => {
        try {
            await navigator.clipboard.writeText(serviceUrl);
            addEvent("Detection API URL copied to clipboard", "success");
        } catch (err) {
            addEvent("Failed to copy URL", "info");
        }
    });
}

if (openBtn) {
    openBtn.addEventListener("click", () => {
        window.open(serviceUrl, "_blank");
        addEvent("Opening Detection API URL", "info");
    });
}

if (docsBtn) {
    docsBtn.addEventListener("click", () => {
        window.open(`${serviceUrl}/docs`, "_blank");
        addEvent("Opening API documentation (/docs)", "info");
    });
}

// --- LOG CONTROLS ---
const pauseBtn = document.getElementById("pauseLogBtn");
const clearBtn = document.getElementById("clearLogBtn");

if (pauseBtn) {
    pauseBtn.addEventListener("click", () => {
        isLogPaused = !isLogPaused;
        pauseBtn.innerHTML = isLogPaused ?
            `<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>Resume` :
            `<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>Pause`;
        addEvent(isLogPaused ? "Logs paused" : "Logs resumed", "info");
    });
}

if (clearBtn) {
    clearBtn.addEventListener("click", async () => {
        logRequestGeneration += 1;
        lastLogsJSON = JSON.stringify([]);
        if (eventsContainer) {
            eventsContainer.innerHTML = "";
            try {
                await fetch(`${MANAGER_API}/clear_logs`, { method: "POST" });
            } catch (err) {
                // ignore
            }
        }
    });
}

// --- POLL BACKEND LOGS (reads actual service.log via Service Manager) ---
async function pollBackendLogs() {
    if (isLogPaused) return;
    const requestGeneration = logRequestGeneration;
    try {
        const response = await fetch(`${MANAGER_API}/logs`);
        if (!response.ok) throw new Error("Failed to fetch logs");
        const data = await response.json();
        if (requestGeneration !== logRequestGeneration) return;

        const currentJSON = JSON.stringify(data.lines);
        if (currentJSON === lastLogsJSON) {
            return;
        }
        lastLogsJSON = currentJSON;

        if (eventsContainer) {
            eventsContainer.innerHTML = "";
            if (data.lines && data.lines.length > 0) {
                data.lines.slice().reverse().forEach(line => {
                    const lineEl = document.createElement("div");
                    lineEl.className = "log-line";
                    lineEl.innerHTML = formatLogLine(line);
                    eventsContainer.appendChild(lineEl);
                });
            }
        }
    } catch (err) {
        // manager offline
    }
}

function formatLogLine(line) {
    // Matches 17:20:22 or HH:MM or with ISO format
    const timeMatch = line.match(/^(\d{2}:\d{2}:\d{2}|\d{2}:\d{2})(?:\s+|-|\s+[A-Z]+\s+)\s*(.*)$/);
    let timeStr = "";
    let textStr = line;
    if (timeMatch) {
        timeStr = timeMatch[1];
        textStr = timeMatch[2];
    }

    let textClass = "log-text";
    const lowerText = textStr.toLowerCase();
    if (lowerText.includes("error") || lowerText.includes("failed") || lowerText.includes("exception")) {
        // errors styled differently
        textClass += " log-green";
    } else if (lowerText.includes("success") || lowerText.includes("started") || lowerText.includes("initialized") || lowerText.includes("healthy") || lowerText.includes("online") || lowerText.includes("connected") || lowerText.includes("passed") || lowerText.includes("opened")) {
        textClass += " log-green";
    }
    if (lowerText.includes("http://") || lowerText.includes("https://")) {
        textClass += " log-link";
    }

    const escapeHtml = value => value.replace(/[&<>"']/g, char => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    }[char]));
    return `
        <span class="log-time">${escapeHtml(timeStr)}</span>
        <span class="${textClass}">${escapeHtml(textStr)}</span>
    `;
}

setInterval(pollBackendLogs, 2000);

// --- DYNAMIC RINGS ---
function updateUsageRing(type, percent) {
    const pct = Math.max(0, Math.min(100, Math.round(percent)));
    const label = document.getElementById(`${type}Value`);
    const ring = document.getElementById(`${type}Ring`);
    if (label) {
        label.textContent = `${pct}%`;
    }
    if (ring) {
        const circumference = 251.2;
        const offset = circumference - (circumference * pct / 100);
        ring.style.strokeDashoffset = offset;
    }
}

// --- HEARTBEATS ---
setInterval(() => {
    loadHealth();
    loadReports();
}, 5000);

// Initial triggers
loadHealth();
loadReports();
addEvent("QC-SCM Dashboard initialized. Waiting for Service Manager...", "success");
