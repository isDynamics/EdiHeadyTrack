import cv2
import numpy as np
import os
from tqdm import tqdm
import xml.etree.ElementTree as ET
try:
    from batch_face import RetinaFace
except ImportError:
    RetinaFace = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# This is the external VGGHeads package, NOT this file
try:
    from head_detector import HeadDetector as VGGHeadDetector
except ImportError:
    VGGHeadDetector = None



class HeadDetector:
    def __init__(self, model_name='retinaface', margin=0.5, model_path=None, gpu_id=0):
        """
        Initialize a head detector using batch-face RetinaFace, YOLO v8, or VGGHeads.
        Supported models: 'retinaface', 'yolo', 'vggheads'
        ...
        """
        self.model_name = model_name
        self.margin = margin
        self.model_path = model_path
        self.gpu_id = gpu_id

        # RetinaFace
        if model_name == 'retinaface':
            if RetinaFace is None:
                raise ImportError(
                    "batch-face / RetinaFace is not installed, "
                    "but model_name='retinaface' was requested."
                )
            self.retinaface_model = RetinaFace(gpu_id=gpu_id)
        else:
            self.retinaface_model = None

        # YOLO
        if model_name == 'yolo':
            if YOLO is None:
                raise ImportError(
                    "ultralytics YOLO is not installed, "
                    "but model_name='yolo' was requested."
                )
            if not model_path:
                package_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(package_dir, 'models', 'yolo.pt')

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"YOLO model file not found: {model_path}")

            self.yolo_model = YOLO(model_path)
        else:
            self.yolo_model = None

        # VGGHeads
        if model_name == 'vggheads':
            if VGGHeadDetector is None:
                raise ImportError(
                    "VGGHeads (external 'head_detector' package) is not installed, "
                    "but model_name='vggheads' was requested."
                )
            self.vgg_head_model = VGGHeadDetector()
            self._vgg_frame_count = 0
        else:
            self.vgg_head_model = None

    
    def _detect_with_retinaface(self, frame):
        """
        Detect faces using batch-face RetinaFace model.
        :param frame: Input frame (numpy array in BGR format)
        :return: List of detections in format compatible with DeepFace
        """
        # Convert BGR to RGB for batch-face
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run detection with batch-face RetinaFace
        faces = self.retinaface_model(frame_rgb, threshold=0.25, resize=1, max_size=1080, return_dict=True)
        
        if not faces:
            return []
        
        detections = []
        
        for face in faces:
            # Extract bounding box from batch-face format
            box = face['box']  # [x1, y1, x2, y2]
            score = face['score']
            
            # Convert to DeepFace-compatible format
            x1, y1, x2, y2 = box
            x = int(x1)
            y = int(y1)
            w = int(x2 - x1)
            h = int(y2 - y1)
            
            # Ensure bounds are within frame
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            
            # Ensure minimum size
            w = max(1, w)
            h = max(1, h)
            
            detection = {
                "facial_area": {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h
                },
                "confidence": float(score)
            }
            detections.append(detection)
        
        return detections
    
    def _detect_with_yolo(self, frame):
        """
        Detect heads using YOLO v8 model.
        :param frame: Input frame (numpy array)
        :return: List of detections in format compatible with DeepFace
        """
        results = self.yolo_model(frame, verbose=False, conf=0.25)
        result = results[0]
        
        if result.boxes is None or len(result.boxes) == 0:
            return []
        
        boxes = result.boxes.cpu().numpy()
        detections = []
        
        for i in range(boxes.shape[0]):
            xyxy = boxes.xyxy[i]
            xmin, ymin, xmax, ymax = map(int, xyxy)
            w = xmax - xmin
            h = ymax - ymin
            
            # YOLO detects heads, but we need face-like regions for consistency
            # More aggressive contraction based on typical face proportions within head
            face_scale_width = 0.6   # Even more aggressive width contraction
            face_scale_height = 0.5 # Very aggressive height contraction
            
            contract_x = int(w * (1 - face_scale_width) / 2)
            contract_y = int(h * (1 - face_scale_height) / 2)
            
            # Shift the face region down and slightly inward (faces are in lower-center of head)
            y_shift = int(h * 0.15)  # Shift down by 15% of head height
            x_shift = int(w * 0.05)  # Shift slightly inward by 5% of head width
            
            face_x = max(0, xmin + contract_x + x_shift)
            face_y = max(0, ymin + contract_y + y_shift)
            face_w = max(1, w - 2 * contract_x - x_shift)
            face_h = max(1, h - 2 * contract_y)
            
            # Ensure we don't exceed frame boundaries
            face_x = min(face_x, frame.shape[1] - 1)
            face_y = min(face_y, frame.shape[0] - 1)
            if face_x + face_w >= frame.shape[1]:
                face_w = frame.shape[1] - face_x - 1
            if face_y + face_h >= frame.shape[0]:
                face_h = frame.shape[0] - face_y - 1
            
            detection = {
                "facial_area": {
                    "x": face_x,
                    "y": face_y,
                    "w": face_w,
                    "h": face_h
                },
                "confidence": float(boxes.conf[i]) if hasattr(boxes, 'conf') and boxes.conf is not None else 1.0
            }
            detections.append(detection)
        
        return detections
    
    def _detect_with_vggheads(self, frame):
        """
        Detect heads using VGGHeads model.
        :param frame: Input frame (numpy array)
        :return: List of detections in format compatible with DeepFace
        """
        
        # Reinitialize model for every frame to prevent any internal state buildup
        # This is aggressive but should ensure consistent performance
        self.vgg_head_model = VGGHeadDetector()
        
        detections = []

        # Force cleanup before processing
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass
        
        # Ensure frame is in correct format (uint8, BGR)
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        
        # VGGHeads expects image in BGR format (which is what OpenCV uses)
        # Make sure frame is contiguous in memory
        frame = np.ascontiguousarray(frame)

        predictions = self.vgg_head_model(frame)

        if not predictions.heads or len(predictions.heads) == 0:
            return []

        for head in predictions.heads:
            # Extract bounding box from head detection
            # VGGHeads provides bbox in format Bbox(x, y, w, h) - already correct format!
            bbox = head.bbox

            x, y, w, h = int(bbox.x), int(bbox.y), int(bbox.w), int(bbox.h)
            
            # Adjust VGGHeads bounding box to better match ground truth face regions
            # Apply empirically determined corrections based on bbox analysis
            center_x_correction = -1.2
            center_y_correction = 0
            width_scale_factor = 0.7
            height_scale_factor = 0.7
            
            # Calculate current center
            current_center_x = x + w / 2
            current_center_y = y + h / 2
            
            # Apply scaling
            new_w = int(w * width_scale_factor)
            new_h = int(h * height_scale_factor)
            
            # Apply center correction and recalculate x, y
            new_center_x = current_center_x + center_x_correction
            new_center_y = current_center_y + center_y_correction
            new_x = int(new_center_x - new_w / 2)
            new_y = int(new_center_y - new_h / 2)
            
            # Ensure bounds - clamp to frame boundaries
            new_x = max(0, min(new_x, frame.shape[1] - new_w))
            new_y = max(0, min(new_y, frame.shape[0] - new_h))
            
            # Additional safety: ensure width and height are positive
            if new_x + new_w > frame.shape[1]:
                new_w = frame.shape[1] - new_x
            if new_y + new_h > frame.shape[0]:
                new_h = frame.shape[0] - new_y
                
            # Final check: ensure minimum size
            new_w = max(1, new_w)
            new_h = max(1, new_h)
            
            # Convert to DeepFace-compatible format
            detection = {
                "facial_area": {
                    "x": new_x,
                    "y": new_y,
                    "w": new_w,
                    "h": new_h
                },
                "confidence": float(head.confidence) if hasattr(head, 'confidence') else 1.0
            }
            detections.append(detection)
        
        # Force GPU memory cleanup to prevent memory accumulation
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass  # If torch not available, skip GPU cleanup
        
        return detections
            
    
    def _calculate_bbox_distance(self, bbox1, bbox2):
        """Calculate the Euclidean distance between centers of two bounding boxes."""
        if not bbox1 or not bbox2:
            return float('inf')
            
        center1_x = bbox1['x'] + bbox1['w'] / 2
        center1_y = bbox1['y'] + bbox1['h'] / 2
        center2_x = bbox2['x'] + bbox2['w'] / 2
        center2_y = bbox2['y'] + bbox2['h'] / 2
        
        return ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
    
    def _smooth_detections(self, detections, max_jump_threshold=100):
        """
        Apply temporal smoothing to prevent large jumps between consecutive frames.
        :param detections: List of detection dictionaries
        :param max_jump_threshold: Maximum allowed jump in pixels between consecutive frames
        :return: Smoothed detections list
        """
        if len(detections) <= 1:
            return detections
        
        smoothed_detections = [detections[0].copy()]  # Start with first detection
        
        for i in range(1, len(detections)):
            current_det = detections[i]
            prev_det = smoothed_detections[i-1]
            
            # Skip smoothing for frames with no detection
            if current_det.get("no_detection", False) or prev_det.get("no_detection", False):
                smoothed_detections.append(current_det.copy())
                continue
            
            current_bbox = current_det.get("facial_area", {})
            prev_bbox = prev_det.get("facial_area", {})
            
            # Calculate distance between current and previous detection
            distance = self._calculate_bbox_distance(current_bbox, prev_bbox)
            
            if distance > max_jump_threshold:
                # Large jump detected - apply smoothing by interpolating
                smoothing_factor = 0.7  # Weight towards previous detection
                
                smoothed_bbox = {
                    "x": int(prev_bbox["x"] * smoothing_factor + current_bbox["x"] * (1 - smoothing_factor)),
                    "y": int(prev_bbox["y"] * smoothing_factor + current_bbox["y"] * (1 - smoothing_factor)),
                    "w": int(prev_bbox["w"] * smoothing_factor + current_bbox["w"] * (1 - smoothing_factor)),
                    "h": int(prev_bbox["h"] * smoothing_factor + current_bbox["h"] * (1 - smoothing_factor))
                }
                
                # Create smoothed detection
                smoothed_det = current_det.copy()
                smoothed_det["facial_area"] = smoothed_bbox
                smoothed_det["smoothed"] = True  # Mark as smoothed for debugging
                smoothed_detections.append(smoothed_det)
            else:
                # No large jump - keep original detection
                smoothed_detections.append(current_det.copy())
        
        return smoothed_detections
    
    def _fill_detection_gaps(self, detections, video_frames):
        """
        Fill gaps where no detection was found using the most recent detection (from immediately before or after).
        :param detections: List of detection dictionaries (may contain gaps)
        :param video_frames: List of video frames for cropping
        :return: Tuple of (filled_detections, filled_cropped_images)
        """
        def find_nearest_valid_detection(frame_idx, detections):
            """Find the nearest valid detection, preferring more recent ones."""
            # First try to find a detection immediately before or after
            search_radius = 1
            max_search_radius = min(10, len(detections) // 4)  # Don't search too far
            
            while search_radius <= max_search_radius:
                # Check frames after current frame first (more recent temporal direction)
                if frame_idx + search_radius < len(detections):
                    future_det = detections[frame_idx + search_radius]
                    if not future_det.get("no_detection", False):
                        return future_det, frame_idx + search_radius
                
                # Then check frames before current frame
                if frame_idx - search_radius >= 0:
                    past_det = detections[frame_idx - search_radius]
                    if not past_det.get("no_detection", False):
                        return past_det, frame_idx - search_radius
                
                search_radius += 1
            
            # If no nearby detection found, fall back to any valid detection
            for i, det in enumerate(detections):
                if not det.get("no_detection", False):
                    return det, i
            
            return None, -1
        
        def interpolate_detection(det1, det2, alpha):
            """Interpolate between two detections based on alpha (0=det1, 1=det2)."""
            if det1 is None or det2 is None:
                return det1 if det1 is not None else det2
                
            area1 = det1.get("facial_area", {})
            area2 = det2.get("facial_area", {})
            
            # Interpolate bounding box coordinates
            interpolated_area = {
                "x": int(area1.get("x", 0) * (1 - alpha) + area2.get("x", 0) * alpha),
                "y": int(area1.get("y", 0) * (1 - alpha) + area2.get("y", 0) * alpha),
                "w": int(area1.get("w", 0) * (1 - alpha) + area2.get("w", 0) * alpha),
                "h": int(area1.get("h", 0) * (1 - alpha) + area2.get("h", 0) * alpha)
            }
            
            # Create interpolated detection
            interpolated_det = det1.copy()
            interpolated_det["facial_area"] = interpolated_area
            interpolated_det["confidence"] = (det1.get("confidence", 1.0) + det2.get("confidence", 1.0)) / 2 * 0.8  # Reduce confidence
            interpolated_det["interpolated"] = True
            
            return interpolated_det
        
        def estimate_motion_and_fill_gap(frame_idx, detections):
            """Estimate head motion and predict position for gap filling."""
            # Look for a sequence of valid detections to estimate motion
            valid_detections = []
            valid_indices = []
            
            # Collect nearby valid detections
            search_range = min(5, len(detections) // 2)
            for i in range(max(0, frame_idx - search_range), min(len(detections), frame_idx + search_range + 1)):
                if not detections[i].get("no_detection", False):
                    valid_detections.append(detections[i])
                    valid_indices.append(i)
            
            if len(valid_detections) < 2:
                # Not enough data for motion estimation, fall back to nearest
                return find_nearest_valid_detection(frame_idx, detections)[0]
            
            # Find the two detections that bracket our target frame
            before_det, after_det = None, None
            before_idx, after_idx = -1, -1
            
            for i, idx in enumerate(valid_indices):
                if idx < frame_idx:
                    if before_det is None or idx > before_idx:
                        before_det = valid_detections[i]
                        before_idx = idx
                elif idx > frame_idx:
                    if after_det is None or idx < after_idx:
                        after_det = valid_detections[i]
                        after_idx = idx
            
            # If we have detections before and after, interpolate
            if before_det and after_det:
                frame_distance = after_idx - before_idx
                alpha = (frame_idx - before_idx) / frame_distance
                return interpolate_detection(before_det, after_det, alpha)
            
            # If we only have one side, try to estimate motion from a sequence
            if len(valid_detections) >= 3:
                # Calculate average motion vector from recent detections
                motion_vectors = []
                for i in range(1, len(valid_detections)):
                    area1 = valid_detections[i-1].get("facial_area", {})
                    area2 = valid_detections[i].get("facial_area", {})
                    
                    dx = area2.get("x", 0) - area1.get("x", 0)
                    dy = area2.get("y", 0) - area1.get("y", 0)
                    dw = area2.get("w", 0) - area1.get("w", 0)
                    dh = area2.get("h", 0) - area1.get("h", 0)
                    
                    frame_diff = valid_indices[i] - valid_indices[i-1]
                    if frame_diff > 0:
                        motion_vectors.append((dx/frame_diff, dy/frame_diff, dw/frame_diff, dh/frame_diff))
                
                if motion_vectors:
                    # Average motion per frame
                    avg_dx = sum(mv[0] for mv in motion_vectors) / len(motion_vectors)
                    avg_dy = sum(mv[1] for mv in motion_vectors) / len(motion_vectors)
                    avg_dw = sum(mv[2] for mv in motion_vectors) / len(motion_vectors)
                    avg_dh = sum(mv[3] for mv in motion_vectors) / len(motion_vectors)
                    
                    # Use the nearest detection and apply motion estimation
                    if before_det:
                        base_det = before_det
                        frame_diff = frame_idx - before_idx
                    else:
                        base_det = after_det
                        frame_diff = frame_idx - after_idx
                    
                    base_area = base_det.get("facial_area", {})
                    
                    # Apply estimated motion
                    predicted_area = {
                        "x": int(base_area.get("x", 0) + avg_dx * frame_diff),
                        "y": int(base_area.get("y", 0) + avg_dy * frame_diff),
                        "w": max(10, int(base_area.get("w", 0) + avg_dw * frame_diff)),
                        "h": max(10, int(base_area.get("h", 0) + avg_dh * frame_diff))
                    }
                    
                    # Create motion-estimated detection
                    motion_det = base_det.copy()
                    motion_det["facial_area"] = predicted_area
                    motion_det["confidence"] = base_det.get("confidence", 1.0) * 0.6  # Reduce confidence more for estimation
                    motion_det["motion_estimated"] = True
                    
                    return motion_det
            
            # Fallback to nearest detection
            return find_nearest_valid_detection(frame_idx, detections)[0]
        
        # Fill gaps and create cropped images
        filled_detections = []
        cropped_images = []
        
        for i, (det, frame) in enumerate(zip(detections, video_frames)):
            if det.get("no_detection", False):
                # Use enhanced motion estimation for gap filling
                gap_filled_det = estimate_motion_and_fill_gap(i, detections)
                
                if gap_filled_det is None:
                    # No valid detection found anywhere - keep the gap
                    filled_detections.append(det)
                    cropped_images.append(None)
                    continue
                
                # Mark as gap-filled and add method info
                gap_filled_det["gap_filled"] = True
                if not gap_filled_det.get("interpolated", False) and not gap_filled_det.get("motion_estimated", False):
                    gap_filled_det["gap_filled_method"] = "nearest"
                elif gap_filled_det.get("motion_estimated", False):
                    gap_filled_det["gap_filled_method"] = "motion_estimated"
                elif gap_filled_det.get("interpolated", False):
                    gap_filled_det["gap_filled_method"] = "interpolated"
                
                filled_detections.append(gap_filled_det)
                
                # Extract cropping coordinates from the gap-filled detection
                facial_area = gap_filled_det.get("facial_area", {})
                x1 = facial_area.get("x", 0)
                y1 = facial_area.get("y", 0)
                x2 = x1 + facial_area.get("w", 0)
                y2 = y1 + facial_area.get("h", 0)
                
                # Ensure bounds are valid for this frame and add some safety margins
                frame_h, frame_w = frame.shape[:2]
                x1 = max(0, min(x1, frame_w - 10))
                y1 = max(0, min(y1, frame_h - 10))
                x2 = max(x1 + 10, min(x2, frame_w))
                y2 = max(y1 + 10, min(y2, frame_h))
                
                # Update the detection with validated coordinates
                gap_filled_det["facial_area"] = {
                    "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1
                }
                
                cropped = frame[y1:y2, x1:x2]
                cropped_images.append(cropped)
            else:
                # Keep original detection
                filled_detections.append(det)
                
                # Extract and crop using original detection
                facial_area = det.get("facial_area", {})
                x1 = facial_area.get("x", 0)
                y1 = facial_area.get("y", 0)
                x2 = x1 + facial_area.get("w", 0)
                y2 = y1 + facial_area.get("h", 0)
                
                cropped = frame[y1:y2, x1:x2]
                cropped_images.append(cropped)
        
        return filled_detections, cropped_images
    
    def _refine_gap_filled_detections(self, detections, video_frames, refinement_method='template_matching'):
        """
        Refine gap-filled detections using template matching or other techniques.
        :param detections: List of detections (some may be gap-filled)
        :param video_frames: List of video frames
        :param refinement_method: Method to use for refinement ('template_matching', 'optical_flow')
        :return: Refined detections
        """
        if refinement_method != 'template_matching':
            return detections
            
        refined_detections = []
        
        for i, det in enumerate(detections):
            if not det.get("gap_filled", False):
                # Not gap-filled, keep as is
                refined_detections.append(det)
                continue
                
            # Try to refine gap-filled detection using template matching
            try:
                frame = video_frames[i]
                facial_area = det.get("facial_area", {})
                x, y, w, h = facial_area.get("x", 0), facial_area.get("y", 0), facial_area.get("w", 0), facial_area.get("h", 0)
                
                # Create a search region around the predicted location
                search_margin = max(w, h) // 2  # Search within half the box size
                search_x1 = max(0, x - search_margin)
                search_y1 = max(0, y - search_margin)
                search_x2 = min(frame.shape[1], x + w + search_margin)
                search_y2 = min(frame.shape[0], y + h + search_margin)
                
                search_region = frame[search_y1:search_y2, search_x1:search_x2]
                
                if search_region.size > 0:
                    # Find a good template from nearby frames
                    template = None
                    template_frame_idx = None
                    
                    # Look for a high-confidence detection nearby to use as template
                    for j in range(max(0, i-3), min(len(detections), i+4)):
                        if j != i and not detections[j].get("no_detection", False) and not detections[j].get("gap_filled", False):
                            temp_conf = detections[j].get("confidence", 0)
                            if temp_conf > 0.7:  # High confidence detection
                                temp_area = detections[j].get("facial_area", {})
                                tx, ty, tw, th = temp_area.get("x", 0), temp_area.get("y", 0), temp_area.get("w", 0), temp_area.get("h", 0)
                                if tw > 10 and th > 10:  # Valid size
                                    template = video_frames[j][ty:ty+th, tx:tx+tw]
                                    template_frame_idx = j
                                    break
                    
                    if template is not None and template.size > 0:
                        # Resize template to match expected size
                        if template.shape[:2] != (h, w):
                            template = cv2.resize(template, (w, h))
                        
                        # Convert to grayscale for template matching
                        if len(template.shape) == 3:
                            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                        else:
                            template_gray = template
                            
                        if len(search_region.shape) == 3:
                            search_gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
                        else:
                            search_gray = search_region
                        
                        # Perform template matching
                        if search_gray.shape[0] >= template_gray.shape[0] and search_gray.shape[1] >= template_gray.shape[1]:
                            result = cv2.matchTemplate(search_gray, template_gray, cv2.TM_CCOEFF_NORMED)
                            _, max_val, _, max_loc = cv2.minMaxLoc(result)
                            
                            # If match is good enough, refine the detection
                            if max_val > 0.3:  # Reasonable match threshold
                                refined_x = search_x1 + max_loc[0]
                                refined_y = search_y1 + max_loc[1]
                                
                                # Update detection with refined location
                                refined_det = det.copy()
                                refined_det["facial_area"] = {
                                    "x": refined_x,
                                    "y": refined_y,
                                    "w": w,
                                    "h": h
                                }
                                refined_det["template_matched"] = True
                                refined_det["template_match_score"] = float(max_val)
                                refined_det["template_source_frame"] = template_frame_idx
                                refined_detections.append(refined_det)
                                continue
                
            except Exception as e:
                # If refinement fails, keep original
                pass
            
            # Keep original gap-filled detection if refinement didn't work
            refined_detections.append(det)
        
        return refined_detections
        
    def predict(self, video, output_dir=None, enable_gap_filling=True, enable_smoothing=True, enable_refinement=False):
        """
        Detect a single head in a video and output cropped images with gap filling and temporal smoothing.
        :param video: Video object (EHT.Video) containing frames
        :param output_dir: Directory to save cropped images
        :param enable_gap_filling: Whether to fill detection gaps using motion estimation/interpolation
        :param enable_smoothing: Whether to apply temporal smoothing to prevent large jumps
        :param enable_refinement: Whether to refine gap-filled detections using template matching
        :return: Tuple of (all_detections, cropped_images) where all_detections is a list of detection results per frame
        """ 
        all_detected_heads = []  # List to store detection results for each frame
        video_frames = []  # Store all frames for gap filling processing
        
        # Create a folder named after the video file
        video_output_dir = os.path.join(output_dir) if output_dir else None
        os.makedirs(video_output_dir, exist_ok=True) if video_output_dir else None

        total_frames = video.frame_count
        
        # First pass: collect all detections and frames
        for i, frame in enumerate(tqdm(video.frames(), total=total_frames, desc="Detecting heads")):
            video_frames.append(frame.copy())  # Store frame for later processing
            
            # Get detections for this frame
            if self.model_name == 'retinaface':
                frame_detections = self._detect_with_retinaface(frame)
            elif self.model_name == 'yolo':
                frame_detections = self._detect_with_yolo(frame)
            elif self.model_name == 'vggheads':
                frame_detections = self._detect_with_vggheads(frame)
            else:
                raise ValueError(f"Unsupported model name: {self.model_name}. Supported models: 'retinaface', 'yolo', 'vggheads'")

            # If we have detections, process the first one (assuming one head per frame)
            if frame_detections:
                head = frame_detections[0]  # Use only the first detected head
                facial_area = head.get("facial_area", {})
                
                x, y, w, h = facial_area.get("x", 0), facial_area.get("y", 0), facial_area.get("w", 0), facial_area.get("h", 0)
                
                # Apply margins consistently for all detectors
                margin_x = int(w * self.margin)
                margin_y = int(h * self.margin)
                
                # Expand bounding box with margins
                x1 = max(x - margin_x, 0)
                y1 = max(y - margin_y, 0)
                x2 = min(x + w + margin_x, frame.shape[1])
                y2 = min(y + h + margin_y, frame.shape[0])
                
                # Store the detection result WITH applied margins for consistent evaluation
                # Update the facial_area to reflect the margin-expanded box
                head_with_margins = head.copy()
                head_with_margins["facial_area"] = {
                    "x": x1,
                    "y": y1, 
                    "w": x2 - x1,
                    "h": y2 - y1
                }
                all_detected_heads.append(head_with_margins)
            else:
                # Add a placeholder for frames with no detections
                all_detected_heads.append({"facial_area": {"x": 0, "y": 0, "w": 0, "h": 0}, "no_detection": True})

        # Second pass: Apply temporal smoothing if enabled
        if enable_smoothing:
            all_detected_heads = self._smooth_detections(all_detected_heads)
        
        # Third pass: Fill gaps and generate cropped images
        if enable_gap_filling:
            all_detected_heads, detected_heads_img = self._fill_detection_gaps(all_detected_heads, video_frames)
            
            # Fourth pass: Refine gap-filled detections if enabled
            if enable_refinement:
                all_detected_heads = self._refine_gap_filled_detections(all_detected_heads, video_frames)
                # Regenerate cropped images after refinement
                detected_heads_img = []
                for i, (det, frame) in enumerate(zip(all_detected_heads, video_frames)):
                    if det.get("no_detection", False):
                        detected_heads_img.append(None)
                    else:
                        facial_area = det.get("facial_area", {})
                        x1 = facial_area.get("x", 0)
                        y1 = facial_area.get("y", 0)
                        x2 = x1 + facial_area.get("w", 0)
                        y2 = y1 + facial_area.get("h", 0)
                        cropped = frame[y1:y2, x1:x2]
                        detected_heads_img.append(cropped)
        else:
            # Generate cropped images without gap filling
            detected_heads_img = []
            for i, (det, frame) in enumerate(zip(all_detected_heads, video_frames)):
                if det.get("no_detection", False):
                    detected_heads_img.append(None)
                else:
                    facial_area = det.get("facial_area", {})
                    x1 = facial_area.get("x", 0)
                    y1 = facial_area.get("y", 0)
                    x2 = x1 + facial_area.get("w", 0)
                    y2 = y1 + facial_area.get("h", 0)
                    
                    cropped = frame[y1:y2, x1:x2]
                    detected_heads_img.append(cropped)
        
        # Save cropped images if output directory is provided
        if video_output_dir:
            for i, cropped in enumerate(detected_heads_img):
                if cropped is not None:
                    cv2.imwrite(os.path.join(video_output_dir, f'{i:04d}.jpg'), cropped)

        return all_detected_heads, detected_heads_img

    @staticmethod
    def load_detections(detections_path):
        """
        Load precomputed head detections from a directory.
        :param detections_path: Path to folder containing cropped head images.
        :return: List of loaded images as numpy arrays.
        """
        images = []
        file_list = sorted(os.listdir(detections_path))
        
        for file in file_list:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(detections_path, file)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
        return images
    
    @staticmethod
    def calculate_occlusions(participant, trial, camera, annotations_file):
        """
        Calculate occlusion ratios for each frame in a given participant-trial-camera.
        Returns a list of occlusion ratios in frame order.
        """
        def rotated_box_to_points(cx, cy, w, h, angle):
            rect = ((cx, cy), (w, h), angle)
            box = cv2.boxPoints(rect)
            return box.astype(np.intp)

        def compute_rotated_intersection_area(box1_pts, box2_pts):
            ret, intersect = cv2.intersectConvexConvex(box1_pts, box2_pts)
            return ret if ret > 0 else 0

        tree = ET.parse(annotations_file)
        root = tree.getroot()
        occlusion_data = {}  # Temporary dict to sort by frame number

        prefix = f"{participant}_{trial}_{camera}_"

        for image in root.findall('image'):
            filename = image.get('name')  # e.g. P003024_L1_F_00012.jpg
            if not filename.startswith(prefix):
                continue

            frame_number = int(filename.split('_')[-1].split('.')[0])
            face_box = None
            face_points = None
            face_area = 0
            occluders = []

            for box in image.findall('box'):
                label = box.get('label')
                xtl = float(box.get('xtl'))
                ytl = float(box.get('ytl'))
                xbr = float(box.get('xbr'))
                ybr = float(box.get('ybr'))
                rotation = float(box.get('rotation', '0.0'))

                w = xbr - xtl
                h = ybr - ytl
                cx = xtl + w / 2
                cy = ytl + h / 2
                points = rotated_box_to_points(cx, cy, w, h, rotation)

                if label == 'Face':
                    face_box = (cx, cy, w, h, rotation)
                    face_points = points
                    face_area = w * h
                elif label in ['Ball', 'Arm/Hand']:
                    occluders.append(points)

            if face_points is not None:
                total_occlusion = sum(
                    compute_rotated_intersection_area(face_points, occ) for occ in occluders
                )
                occlusion_ratio = total_occlusion / face_area if face_area > 0 else 0
                occlusion_data[frame_number] = occlusion_ratio

        # Convert to sorted list by frame number
        sorted_frames = sorted(occlusion_data.keys())
        occlusion_ratios = [occlusion_data[frame] for frame in sorted_frames]
        
        return occlusion_ratios
    

    @staticmethod
    def calculate_visible_face_area(participant, trial, camera, annotations_file):
        """
        Calculate the total visible face area (in pixels) for each frame in a given participant-trial-camera.
        This accounts for overlap with occluding objects, subtracting the overlapping area from the face area.
        If no face bounding box is present, the visible face area is zero.
        Returns a list of visible face areas in frame order.
        """
        def rotated_box_to_points(cx, cy, w, h, angle):
            rect = ((cx, cy), (w, h), angle)
            box = cv2.boxPoints(rect)
            return box.astype(np.intp)

        def compute_rotated_intersection_area(box1_pts, box2_pts):
            ret, intersect = cv2.intersectConvexConvex(box1_pts, box2_pts)
            return ret if ret > 0 else 0

        tree = ET.parse(annotations_file)
        root = tree.getroot()
        visible_area_data = {}  # Temporary dict to sort by frame number

        prefix = f"{participant}_{trial}_{camera}_"

        for image in root.findall('image'):
            filename = image.get('name')  # e.g. P003024_L1_F_00012.jpg
            if not filename.startswith(prefix):
                continue

            frame_number = int(filename.split('_')[-1].split('.')[0])
            face_box = None
            face_points = None
            face_area = 0
            occluders = []

            for box in image.findall('box'):
                label = box.get('label')
                xtl = float(box.get('xtl'))
                ytl = float(box.get('ytl'))
                xbr = float(box.get('xbr'))
                ybr = float(box.get('ybr'))
                rotation = float(box.get('rotation', '0.0'))

                w = xbr - xtl
                h = ybr - ytl
                cx = xtl + w / 2
                cy = ytl + h / 2
                points = rotated_box_to_points(cx, cy, w, h, rotation)

                if label == 'Face':
                    face_box = (cx, cy, w, h, rotation)
                    face_points = points
                    face_area = w * h
                elif label in ['Ball', 'Arm/Hand']:
                    occluders.append(points)

            if face_points is not None:
                total_occlusion = sum(
                    compute_rotated_intersection_area(face_points, occ) for occ in occluders
                )
                visible_face_area = max(face_area - total_occlusion, 0)
                visible_area_data[frame_number] = visible_face_area
            else:
                visible_area_data[frame_number] = 0  # No face detected, visible area is zero

        # Convert to sorted list by frame number
        sorted_frames = sorted(visible_area_data.keys())
        visible_face_areas = [visible_area_data[frame] for frame in sorted_frames]
        
        return visible_face_areas
