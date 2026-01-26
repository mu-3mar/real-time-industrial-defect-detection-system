import sys
import os

# Suppress Logs
os.environ["QT_LOGGING_RULES"] = "*.warning=false"
os.environ["ORT_LOGGING_LEVEL"] = "3"

import subprocess
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFrame
)
from PySide6.QtCore import Qt

ROOT = Path(__file__).resolve().parents[1]
process = None

def run_flow(script_path: str):
    global process

    if process and process.poll() is None:
        status.setText("⚠️ Pipeline already running")
        return

    full_path = ROOT / script_path
    if not full_path.exists():
        status.setText(f"❌ Not found: {script_path}")
        return

    status.setText("🟢 Running...")
    process = subprocess.Popen(
        [sys.executable, str(full_path)],
        cwd=ROOT
    )

# ================== APP ==================
app = QApplication(sys.argv)
window = QWidget()
window.setWindowTitle("QC-SCM | Production Control")
window.setFixedSize(720, 420)

# ================== STYLES ==================
window.setStyleSheet("""
QWidget {
    background-color: #0f172a;
    color: #e5e7eb;
    font-family: Inter;
}
QFrame {
    background-color: #111827;
    border-radius: 14px;
}
QPushButton {
    background-color: #1f2933;
    border-radius: 12px;
    padding: 20px;
    font-size: 16px;
}
QPushButton:hover {
    background-color: #2563eb;
}
""")

# ================== HEADER ==================
title = QLabel("QC-SCM Production Control")
title.setAlignment(Qt.AlignCenter)
title.setStyleSheet("font-size: 24px; font-weight: bold;")

subtitle = QLabel("AI Quality Inspection Pipelines")
subtitle.setAlignment(Qt.AlignCenter)
subtitle.setStyleSheet("color: #9ca3af;")

# ================== CARDS ==================
cards = QHBoxLayout()

def card(title_text, subtitle_text, callback):
    frame = QFrame()
    layout = QVBoxLayout(frame)

    t = QLabel(title_text)
    t.setStyleSheet("font-size: 18px; font-weight: bold;")
    s = QLabel(subtitle_text)
    s.setStyleSheet("color: #9ca3af;")

    btn = QPushButton("Start")
    btn.clicked.connect(callback)

    layout.addWidget(t)
    layout.addWidget(s)
    layout.addStretch()
    layout.addWidget(btn)

    return frame

cards.addWidget(
    card(
        "📦 Boxes Line",
        "Carton & box inspection",
        lambda: run_flow("Boxes/flow/main.py")
    )
)

cards.addWidget(
    card(
        "🧴 Bottles Line",
        "Plastic bottle inspection",
        lambda: run_flow("Bottles/flow/main.py")
    )
)

# ================== STATUS ==================
status = QLabel("⚪ Idle")
status.setAlignment(Qt.AlignCenter)
status.setStyleSheet("color: #9ca3af; font-size: 14px;")

# ================== MAIN LAYOUT ==================
layout = QVBoxLayout(window)
layout.addWidget(title)
layout.addWidget(subtitle)
layout.addSpacing(20)
layout.addLayout(cards)
layout.addStretch()
layout.addWidget(status)

window.show()
sys.exit(app.exec())
