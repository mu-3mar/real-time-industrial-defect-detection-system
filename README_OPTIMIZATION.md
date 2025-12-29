# Pipeline Optimization - Quick Reference

## Performance Tuning Parameters

### Boxes Flow (`Boxes/flow/main.py`)

```python
# Frame Processing
PROCESS_EVERY_N_FRAMES = 3      # Run defect detection every N frames
                                 # Higher = Better performance, lower = More accuracy
                                 # Recommended: 2-5

# Status Logging  
STATUS_UPDATE_INTERVAL = 1.0     # Seconds between console output
                                 # Recommended: 0.5-2.0

# Box Tracking
MIN_FRAMES = 3                   # Frames needed to confirm box presence
                                 # Higher = Fewer false positives
                                 # Recommended: 2-5

MAX_MISSED = 5                   # Frames before considering box exited
                                 # Higher = More forgiving for occlusions
                                 # Recommended: 3-10

# Decision Making
DECISION_FRAMES = 5              # Defect samples needed to lock decision
                                 # Higher = More confident, slower
                                 # Recommended: 3-7

CONFIDENCE_THRESHOLD = 0.6       # Defect ratio to classify as DEFECT
                                 # 0.6 = 60% of samples must show defect
                                 # Recommended: 0.5-0.7
```

### Bottles Flow (`Bottles/flow/main.py`)

```python
PROCESS_EVERY_N_FRAMES = 2  # Process every Nth frame
                             # Recommended: 2-3
```

## Common Adjustments

### High-Speed Conveyor
```python
MIN_FRAMES = 2          # Faster confirmation
MAX_MISSED = 3          # Quicker exit detection
DECISION_FRAMES = 3     # Fewer samples needed
```

### High-Accuracy Requirements
```python
PROCESS_EVERY_N_FRAMES = 2  # More frequent checks
DECISION_FRAMES = 7          # More samples
CONFIDENCE_THRESHOLD = 0.7   # Higher threshold
```

### Performance-Critical
```python
PROCESS_EVERY_N_FRAMES = 5   # Maximum skipping
DECISION_FRAMES = 3           # Quick decisions
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Boxes not detected | Lower `MIN_FRAMES` to 2 |
| False positives | Increase `MIN_FRAMES` to 4-5 |
| Flickering count | Increase `MAX_MISSED` to 7-10 |
| Missing fast boxes | Decrease `PROCESS_EVERY_N_FRAMES` |
| Low FPS | Increase `PROCESS_EVERY_N_FRAMES` |
| Incorrect defect classification | Adjust `CONFIDENCE_THRESHOLD` |

## File Structure

```
QC-SCM/
├── GUI/
│   └── luncher.py              # Main launcher GUI
├── Boxes/
│   └── flow/
│       ├── main.py             # Optimized boxes pipeline
│       ├── tracker/
│       │   ├── __init__.py
│       │   └── box_tracker.py  # State machine & tracking
│       ├── utils/
│       │   ├── __init__.py
│       │   └── performance.py  # Metrics & logging
│       ├── detector/
│       │   └── yolo_detector.py
│       └── config/             # YAML configs (gitignored)
└── Bottles/
    └── flow/
        └── main.py             # Optimized bottles pipeline
```

## Quick Start

1. **Launch GUI**: `python GUI/luncher.py`
2. **Click "Start"** on desired line
3. **Press ESC** to stop pipeline
4. **Check console** for performance stats

## Key Improvements

- 🚀 **67% less processing** on boxes flow
- 🔒 **Stable annotations** - no flickering
- 📊 **1-second status updates** - readable logs
- 📈 **Built-in metrics** - FPS, timing, skip ratio
