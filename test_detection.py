import EdiHeadyTrack as EHT
import os
import xml.etree.ElementTree as ET
import time
import pandas as pd
import shutil  # Add this import for copying files

annotations_file = "../../../Data/HeadDetection/GT/annotations.xml"
participants = ['P003024', 'P003272', 'P003336', 'P003489', 'P003540', 'P003682', 'P003726', 'P003730', 'P003940', 'P004298']
trials = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6']
camera_names = ['F', 'S']
# participants = ['P003940']
# trials = ['R5']
# camera_names = ['S']

margin_percentage = 0.4

def parse_annotations(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    annotations = {}
    for image in root.findall('image'):
        filename = image.get('name')
        boxes = []

        for box in image.findall('box'):
            # Only include boxes labeled as "Face"
            label = box.get('label')
            if label != "Face":
                continue
                
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            width = xbr - xtl
            height = ybr - ytl

            # Determine the camera type based on the filename
            camera_type = 'F' if '_F' in filename else 'S' if '_S' in filename else 'Unknown'
            
            # Initialize camera-specific lists if not already present
            if camera_type not in annotations:
                annotations[camera_type] = {'widths': [], 'heights': []}
            
            # Append width and height to the respective camera type lists
            annotations[camera_type]['widths'].append(width)
            annotations[camera_type]['heights'].append(height)

            margin_x = width * margin_percentage
            margin_y = height * margin_percentage
            
            xtl = max(0, int(xtl - margin_x))
            ytl = max(0, int(ytl - margin_y))
            xbr = int(xbr + margin_x)
            ybr = int(ybr + margin_y)

            boxes.append((xtl, ytl, xbr, ybr))
        if boxes:  # Only add images with bounding boxes
            annotations[filename] = boxes
    return annotations

annotations = parse_annotations(annotations_file)

# Initialize a list to store result rows
results_list = []

# Function to calculate Intersection over Union (IoU)
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Calculate average bounding box height and width for each camera type
average_dimensions = {}
for camera_type, dimensions in annotations.items():
    if camera_type in ['F', 'S']:  # Only consider camera types 'F' and 'S'
        avg_width = sum(dimensions['widths']) / len(dimensions['widths']) if dimensions['widths'] else 0
        avg_height = sum(dimensions['heights']) / len(dimensions['heights']) if dimensions['heights'] else 0
        average_dimensions[camera_type] = {'average_width': avg_width, 'average_height': avg_height}

# Print the average dimensions
for camera_type, avg_dims in average_dimensions.items():
    print(f"Camera {camera_type}: Average Width = {avg_dims['average_width']:.2f}, Average Height = {avg_dims['average_height']:.2f}")

# Available models: yolo, retinaface, vggheads
for detector_model in ['vggheads']:
    detector_results = []  # List to store results for the current detector
    for participant in participants:
        for trial in trials:
            for camera_name in camera_names:
                
                print(f"Processing {participant}, {trial}, {camera_name} with {detector_model} detector...")
                
                video_path = f'../../../Data/Videos/Processed/{participant}_{trial}_{camera_name}.avi'
                head_detections_path = f'../../../Data/HeadDetection/{detector_model}/all/{participant}_{trial}_{camera_name}/'
                successful_detections_path = f'../../../Data/HeadDetection/{detector_model}/successful/{participant}_{trial}_{camera_name}/'
                unsuccessful_detections_path = f'../../../Data/HeadDetection/{detector_model}/unsuccessful/{participant}_{trial}_{camera_name}/'
                
                occlusion_ratios = EHT.HeadDetector.calculate_occlusions(participant, trial, camera_name, annotations_file)

                if not os.path.exists(video_path):
                    print(f"Video not found: {video_path}. Skipping...")
                    continue

                video = EHT.Video(video_path)

                # We need to match each frame with its corresponding annotation
                # Since the annotations use frame-based filenames like P003024_L1_F_frame_0000.jpg
                # We'll build a list of all potential frame annotation keys for this video
                frame_keys = [f"{participant}_{trial}_{camera_name}_frame_{i:04d}.jpg" for i in range(video.frame_count)]
                
                # Get all ground truth boxes for all frames in this video
                all_gt_boxes = {}
                for key in frame_keys:
                    if key in annotations:
                        all_gt_boxes[key] = annotations[key]

                # Measure detection time
                start_time = time.time()
                try:
                    # Detect faces using the specified detector
                    detector = EHT.HeadDetector(detector_model)
                    detected_heads, detected_heads_img = detector.predict(video, output_dir=head_detections_path)
                    detection_time = time.time() - start_time
                    
                    # Check if we have one detection per frame
                    if len(detected_heads) != video.frame_count:
                        print(f"Warning: Expected {video.frame_count} detections, but got {len(detected_heads)}")
                        
                except Exception as e:
                    print(f"Error processing {video_path} with {detector_model}: {e}")
                    detection_time = None
                    detected_heads = []
                    detected_heads_img = []
                    bbox = None
                

                # Evaluate detections
                true_positives = 0
                false_positives = 0
                
                # Count total ground truth boxes across all frames
                total_gt_boxes = sum(len(boxes) for boxes in all_gt_boxes.values())
                false_negatives = total_gt_boxes  # Start with all GT as false negatives
                
                iou_scores = []
                video_name = f"{participant}_{trial}_{camera_name}"

                # Print some debugging information
                valid_detections = sum(1 for det in detected_heads if not det.get("no_detection", False))
                print(f"  Video stats: {len(detected_heads)} frames, {valid_detections} valid detections, {len(all_gt_boxes)} GT frames")

                # Process each frame's detection
                for frame_idx, det in enumerate(detected_heads):
                    # Construct the corresponding frame key for this detection
                    frame_key = f"{participant}_{trial}_{camera_name}_frame_{frame_idx:04d}.jpg"
                    
                    # Get ground truth boxes for this specific frame
                    frame_gt_boxes = all_gt_boxes.get(frame_key, [])
                    
                    # Check if this frame has ground truth but no detection
                    if len(frame_gt_boxes) > 0 and det.get("no_detection", False):
                        print(f"  ✗ FALSE NEGATIVE - Frame {frame_idx:04d}: No detection but GT available: {frame_gt_boxes}")
                        continue
                    
                    # Skip frames with no detection
                    if det.get("no_detection", False):
                        continue
                        
                    facial_area = det.get('facial_area', {})
                    if facial_area:
                        x = facial_area.get('x', 0)
                        y = facial_area.get('y', 0)
                        w = facial_area.get('w', 0)
                        h = facial_area.get('h', 0)

                        # Use the detector's bounding box directly (margins are handled internally)
                        det_box = (x, y, x + w, y + h)
                        best_iou = 0
                        best_gt_box = None
                        
                        for gt_box in frame_gt_boxes:
                            iou = calculate_iou(det_box, gt_box)
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_box = gt_box
                                
                        if best_iou >= 0.3:  # IoU threshold for a correct detection
                            true_positives += 1
                            false_negatives -= 1
                            iou_scores.append(best_iou)
                            # Only show low IoU true positives for debugging
                            if best_iou < 0.5:
                                print(f"  ⚠ LOW IoU TRUE POSITIVE - Frame {frame_idx:04d}: IoU={best_iou:.3f}")
                        else:
                            false_positives += 1
                            if len(frame_gt_boxes) > 0:
                                print(f"  ✗ FALSE NEGATIVE (Low IoU) - Frame {frame_idx:04d}: IoU={best_iou:.3f}, Det={det_box}, GT={best_gt_box}")
                            else:
                                print(f"  ✗ FALSE POSITIVE - Frame {frame_idx:04d}: IoU={best_iou:.3f}, Det={det_box}, No GT available")
                    else:
                        if len(frame_gt_boxes) > 0:
                            print(f"  ✗ FALSE NEGATIVE - Frame {frame_idx:04d}: No facial_area but GT available: {frame_gt_boxes}")
                        else:
                            print(f"  ⚠ No facial_area in detection for frame {frame_idx:04d} (no GT either)")
                        
                avg_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0
                
                # Calculate detection time per successful detection (only frames with actual detections)
                successful_detections = sum(1 for det in detected_heads if not det.get("no_detection", False))
                detection_time_per_detection = detection_time / successful_detections if detection_time is not None and successful_detections > 0 else None
                
                # Append results to the detector's list
                detector_results.append({
                    "Image": video_name,
                    "Detection Time (s)": detection_time,
                    "Detection Time Per Detection (s)": detection_time_per_detection,
                    "Successful Detections": successful_detections,
                    "Total Frames": len(detected_heads),
                    "True Positives": true_positives,
                    "False Positives": false_positives,
                    "False Negatives": false_negatives,
                    "Average IoU": avg_iou
                })
                
                # Print video summary for quick identification of problematic videos
                if false_positives > 0 or false_negatives > 0:
                    print(f"  📊 ISSUES FOUND - Video: {video_name}, FP: {false_positives}, FN: {false_negatives}, IoU: {avg_iou:.3f}")

                # Check if the video has exactly 6 true positives and no false positives or false negatives
                is_successful_video = (true_positives == 6 and false_positives == 0 and false_negatives == 0)

                if is_successful_video:
                    print(f"  ✅ All frames successfully detected for {participant}_{trial}_{camera_name}")

                    # Copy detections to the 'successful' folder
                    if not os.path.exists(successful_detections_path):
                        os.makedirs(successful_detections_path)

                    for file_name in os.listdir(head_detections_path):
                        full_file_name = os.path.join(head_detections_path, file_name)
                        if os.path.isfile(full_file_name):
                            shutil.copy(full_file_name, successful_detections_path)
                else:
                    print(f"  ❌ Not all frames successfully detected for {participant}_{trial}_{camera_name}")

                    # Copy detections to the 'unsuccessful' folder
                    if not os.path.exists(unsuccessful_detections_path):
                        os.makedirs(unsuccessful_detections_path)

                    for file_name in os.listdir(head_detections_path):
                        full_file_name = os.path.join(head_detections_path, file_name)
                        if os.path.isfile(full_file_name):
                            shutil.copy(full_file_name, unsuccessful_detections_path)
                

    # Calculate summary statistics for the current detector
    total_videos = len(detector_results)
    total_images = len(detector_results) * 6
    total_tp = sum(res["True Positives"] for res in detector_results)
    total_fp = sum(res["False Positives"] for res in detector_results)
    total_fn = sum(res["False Negatives"] for res in detector_results)
    mean_iou = sum(res["Average IoU"] for res in detector_results) / total_videos if total_videos else 0
    
    # Calculate mean detection time per successful detection
    valid_detection_times_per_detection = [res["Detection Time Per Detection (s)"] for res in detector_results if res["Detection Time Per Detection (s)"] is not None]
    mean_detection_time_per_detection = sum(valid_detection_times_per_detection) / len(valid_detection_times_per_detection) if valid_detection_times_per_detection else 0
    
    # Calculate totals for successful detections and frames
    total_successful_detections = sum(res["Successful Detections"] for res in detector_results)
    total_frames = sum(res["Total Frames"] for res in detector_results)

    # Calculate precision, recall and F1-score
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Print summary for the current detector
    print(f"\nSummary for {detector_model} detector:")
    print(f"Total Videos Evaluated: {total_videos}")
    print(f"Total Frames: {total_frames}")
    print(f"Total Successful Detections: {total_successful_detections}")
    print(f"Detection Success Rate: {total_successful_detections/total_frames:.4f}" if total_frames > 0 else "Detection Success Rate: 0.0000")
    print(f"Total True Positives: {total_tp}")
    print(f"Total False Positives: {total_fp}")
    print(f"Total False Negatives: {total_fn}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean Detection Time Per Successful Detection (s): {mean_detection_time_per_detection:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    
    # Append detector results to the overall results list
    results_list.extend([{"Detector": detector_model, **res} for res in detector_results])

# Save results_list to a CSV file
results_df = pd.DataFrame(results_list)
results_df.to_csv("../../../Data/HeadDetection/detection_results.csv", index=False)
print("Results saved to detection_results.csv")