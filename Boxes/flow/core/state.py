from collections import deque, defaultdict
from datetime import datetime

class AppState:
    def __init__(self, stability_config):
        # Config
        self.min_frames = stability_config.get("min_frames", 3)
        self.max_missed = stability_config.get("max_missed", 5)
        self.vote_window = stability_config.get("vote_window", 7)
        self.vote_threshold = stability_config.get("vote_threshold", 4)
        
        # Runtime State
        self.box_histories = defaultdict(lambda: deque(maxlen=self.vote_window))
        self.last_defect_results = {} # Cache: box_id -> bool
        
        self.frames_inside = 0
        self.missed_frames = 0
        self.inside = False
        self.final_decision = None # "DEFECT" | "OK"
        
        # Counters
        self.total_count = 0
        self.defect_count = 0
        self.ok_count = 0

    def update_history(self, box_id, is_defect):
        """Adds a detection result to the voting history."""
        self.box_histories[box_id].append(is_defect)

    def get_status(self, box_id):
        """Returns ('STATUS_STR', (B, G, R), 'DECISION_CODE')"""
        votes = sum(self.box_histories[box_id])
        if votes >= self.vote_threshold:
            self.final_decision = "DEFECT"
            return "Defect Box", (0, 0, 255), "DEFECT"
        else:
            self.final_decision = "OK"
            return "Non-Defect Box", (0, 170, 0), "OK"

    def process_entry_exit(self, detected):
        """
        Updates internal state based on whether a box was detected this frame.
        Returns (just_exited: bool, final_decision: str or None)
        """
        just_exited = False
        decision = None

        if detected:
            self.frames_inside += 1
            self.missed_frames = 0
            if self.frames_inside >= self.min_frames:
                self.inside = True
        else:
            self.missed_frames += 1
            if self.missed_frames > self.max_missed and self.inside:
                # Box Exited - capture decision BEFORE reset
                decision = self.final_decision
                self.frame_exit()
                just_exited = True
        
        return just_exited, decision

    def frame_exit(self):
        """Handle exit logic: increment counters, reset state."""
        self.total_count += 1
        if self.final_decision == "DEFECT":
            self.defect_count += 1
        else:
            self.ok_count += 1
        
        # print(f"[{datetime.now().strftime('%H:%M:%S')}] Box Processed: {self.final_decision} | Total: {self.total_count}")

        # Reset
        self.inside = False
        self.frames_inside = 0
        self.final_decision = None
        self.box_histories.clear()
        self.last_defect_results.clear()
