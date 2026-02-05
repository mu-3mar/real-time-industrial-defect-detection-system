class Geometry:
    @staticmethod
    def get_crop(frame, box, offset_x=0):
        """
        Crops the frame based on box coordinates.
        Box is [x1, y1, x2, y2] relative to the frame snippet.
        """
        x1, y1, x2, y2 = map(int, box)
        # Ensure we don't go out of bounds
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        return frame[y1:y2, x1:x2]

    @staticmethod
    def get_stable_id(box):
        """Returns a spatial ID (rounded coordinates) for tracking."""
        x1, y1, _, _ = map(int, box)
        return (x1 // 20, y1 // 20)
