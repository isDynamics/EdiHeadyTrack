import cv2
import os

class Video:
    def __init__(self, video_path):
        """
        Load a video from file and initialize frame extraction.
        :param video_path: Path to the video file.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
        
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def frames(self):
        """
        Generator function to yield frames from the video.
        :return: Iterator over frames as numpy arrays.
        """
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame
        self.cap.release()
    
    def get_frame(self, frame_idx):
        """
        Get a specific frame by index.
        :param frame_idx: Index of the frame to retrieve.
        :return: The requested frame as a numpy array.
        """
        if frame_idx < 0 or frame_idx >= self.frame_count:
            raise IndexError("Frame index out of range")
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("Failed to read frame")
        return frame
    
    def release(self):
        """
        Release the video capture object.
        """
        self.cap.release()