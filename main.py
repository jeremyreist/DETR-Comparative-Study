import os, re, cv2, torch, pickle
import numpy as np
from PIL import Image
from testing_models.detr.detr import detect_objects
from utils import visualize_detection_results
from tqdm import tqdm
from scipy.io import loadmat
from transformers import AutoImageProcessor, DeformableDetrForObjectDetection


SELECTED_CLASS = 3 # 3 is for car, 1 is for person

# Applies DETR object detection on a sequence of frames, visualizes the results, and saves them as a video.
def detr_yt_objects(input_folder, output_video, frame_limit):
    # Load the DETR model
    model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
    model.eval()

    # Get a list of frame files in the input folder, sorted in numerical order
    frame_files = sorted(
        [f for f in os.listdir(input_folder) if f.endswith('.jpg')],
        key=lambda x: int(re.findall(r'\d+', x)[0])
    )
    # Check if there are frames in the input folder
    if not frame_files:
        print("No frames found in the input folder.")
        return

    # Read the first frame to get the dimensions
    first_frame = cv2.imread(os.path.join(input_folder, frame_files[0]))
    frame_height, frame_width, _ = first_frame.shape

    # Create a VideoWriter object to write the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, 30, (frame_width, frame_height))

    predicted_boxes = []
    frame_files = frame_files[0:frame_limit]

    for frame_file in tqdm(frame_files):
        frame_path = os.path.join(input_folder, frame_file)

        # Load the frame as a PIL image
        pil_img = Image.open(frame_path)

        # Call the detect_objects function with the frame and the model
        _, prob, boxes = detect_objects(frame_path, model)
        predicted_boxes.append(boxes)

        # Get the annotated image using the plot_results function
        annotated_image = visualize_detection_results(pil_img, prob, boxes, SELECTED_CLASS, return_image=True, display_output=False)

        # Convert the annotated image to BGR format (required by OpenCV)
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        # Resize the annotated image to match the dimensions of the first frame
        annotated_image = cv2.resize(annotated_image, (frame_width, frame_height))

        # Write the processed frame to the output video
        out.write(annotated_image)

    # Release the VideoWriter object and close all windows
    out.release()
    cv2.destroyAllWindows()

    return predicted_boxes


def deformable_detr_yt_objects(input_folder, output_video, frame_limit):
    # Load the Deformable DETR model
    model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr")
    model.eval()

    image_processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")

    # Get a list of frame files in the input folder, sorted in numerical order
    frame_files = sorted(
        [f for f in os.listdir(input_folder) if f.endswith('.jpg')],
        key=lambda x: int(re.findall(r'\d+', x)[0])
    )

    if not frame_files:
        print("No frames found in the input folder.")
        return

    first_frame = cv2.imread(os.path.join(input_folder, frame_files[0]))
    frame_height, frame_width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, 30, (frame_width, frame_height))

    predicted_boxes = []
    frame_files = frame_files[0:frame_limit]

    for frame_file in tqdm(frame_files):
        frame_path = os.path.join(input_folder, frame_file)

        pil_img = Image.open(frame_path)

        inputs = image_processor(images=pil_img, return_tensors="pt")
        outputs = model(**inputs)

        target_sizes = torch.tensor([pil_img.size[::-1]])
        results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]

        prob = []
        boxes = []
        
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if label.item() == SELECTED_CLASS:
                prob.append([0, 0, 0, score.item()])
                boxes.append(box.tolist())

        predicted_boxes.append(boxes)
        annotated_image = visualize_detection_results(pil_img, prob, np.array(boxes), SELECTED_CLASS, return_image=True, display_output=False)

        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        annotated_image = cv2.resize(annotated_image, (frame_width, frame_height))

        out.write(annotated_image)

    out.release()
    cv2.destroyAllWindows()

    return predicted_boxes

# Applies a baseline method on a sequence of frames using ground truth bounding boxes, visualizes the results, and saves them as a video.
def yt_objects_baseline(input_folder, output_video, gt_boxes, frame_limit):
    frame_count = 0

    # Get a list of frame files in the input folder, sorted in numerical order
    frame_files = sorted(
        [f for f in os.listdir(input_folder) if f.endswith('.jpg')],
        key=lambda x: int(re.findall(r'\d+', x)[0])
    )
    # Check if there are frames in the input folder
    if not frame_files:
        print("No frames found in the input folder.")
        return

    # Read the first frame to get the dimensions
    first_frame = cv2.imread(os.path.join(input_folder, frame_files[0]))
    frame_height, frame_width, _ = first_frame.shape

    # Create a VideoWriter object to write the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, 30, (frame_width, frame_height))
    boxes = np.array([])

    for frame_file in tqdm(frame_files):
        frame_path = os.path.join(input_folder, frame_file)

        # Load the frame as a PIL image
        pil_img = Image.open(frame_path)

        
        try:
            boxes = np.array([gt_boxes['car'+frame_file.rstrip('.jpg')]])
        except KeyError:
            boxes = np.array([])

        annotated_image = visualize_detection_results(pil_img, [[0,0,0,1]], boxes, SELECTED_CLASS, return_image=True, display_output=False)

        # Convert the annotated image to BGR format (required by OpenCV)
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        # Resize the annotated image to match the dimensions of the first frame
        annotated_image = cv2.resize(annotated_image, (frame_width, frame_height))

        # Write the processed frame to the output video
        out.write(annotated_image)
    
        if frame_count == frame_limit:
            break

        frame_count += 1 

    # Release the VideoWriter object and close all windows
    out.release()
    cv2.destroyAllWindows()


#  Calculates the Mean Intersection over Union (IoU) between the predicted bounding boxes and the ground truth bounding boxes.
def calculateMIoU(predictions, gt_boxes):
    ious = []
    # Calculate the intersection over union (IoU) for each predicted bounding box
    for i in range(len(predictions)):
        num = f"car{i:08}"
        if num in gt_boxes.keys():
            max_iou = 0
            # For each bounding box in the frame
            for j in range(len(predictions[i])):
                # Calculate the intersection area
                xA = max(predictions[i][j][0], gt_boxes[num][0])
                yA = max(predictions[i][j][1], gt_boxes[num][1])
                xB = min(predictions[i][j][2], gt_boxes[num][2])
                yB = min(predictions[i][j][3], gt_boxes[num][3])
                intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

                # Calculate the union area
                box1_area = (predictions[i][j][2] - predictions[i][j][0] + 1) * (predictions[i][j][3] - predictions[i][j][1] + 1)
                box2_area = (gt_boxes[num][2] - gt_boxes[num][0] + 1) * (gt_boxes[num][3] - gt_boxes[num][1] + 1)
                union_area = box1_area + box2_area - intersection_area

                # Calculate the IoU
                iou = intersection_area / union_area

                if iou > max_iou:
                    max_iou = iou

            # If there is a ground truth bounding box for the current frame, add the IoU to the list
            ious.append(max_iou)

    # Calculate the mean IoU for the current ground truth set
    mean_iou = np.mean(ious)

    return mean_iou


# Processes a sequence of frames using either the DETR object detection model or a baseline method and calculates the Mean IoU for the ground truth.
def process_shot(number_of_frames, model_type):
    # Define the input folder path containing YouTube-Objects dataset
    input_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'data/YouTube-Objects-2.2/car/')
    
    # Load ground truth bounding boxes from .mat files
    gt_boxes_1 = loadmat(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'data/YouTube-Objects-2.2/GroundTruth/GroundTruth/car/bb_gtTest_car.mat'))['bb_gtTest'][0]
    gt_boxes_2 = loadmat(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'data/YouTube-Objects-2.2/GroundTruth/GroundTruth/car/bb_gtTraining_car.mat'))['bb_gtTraining'][0]
    
    # Combine ground truth bounding boxes into a single dictionary
    gt_boxes = {}
    for i in gt_boxes_1:
        gt_boxes[i[0][0]] = i[1][0]
    for i in gt_boxes_2:
        gt_boxes[i[0][0]] = i[1][0]

    # Define output video and pickle file paths
    output_video = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'results/YT-Obj-2.2-{number_of_frames}-{model_type}.mp4')
    pickle_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'results/YT-Obj-2.2-{number_of_frames}-{model_type}.pkl')

    # Check if the model type is 'detr'
    if model_type == 'detr':
        # Load precomputed DETR predictions if available, otherwise run DETR on the dataset
        if os.path.exists(pickle_path):
            detr_preds = pickle.load(open(pickle_path, 'rb'))
        else:
            detr_preds = detr_yt_objects(input_folder, output_video, number_of_frames)
            pickle.dump(detr_preds, open(pickle_path, 'wb'))

        # Calculate Mean IoU for ground truth and DETR predictions
        miou = calculateMIoU(detr_preds, gt_boxes)
        print("Mean IoU for ground truth: {}".format(miou))

    elif model_type == 'deformable-detr':
        # Load precomputed deformable DETR predictions if available, otherwise run it on the dataset
        if os.path.exists(pickle_path):
            detr_preds = pickle.load(open(pickle_path, 'rb'))
        else:
            detr_preds = deformable_detr_yt_objects(input_folder, output_video, number_of_frames)
            pickle.dump(detr_preds, open(pickle_path, 'wb'))

        # Calculate Mean IoU for ground truth and DETR predictions
        miou = calculateMIoU(detr_preds, gt_boxes)
        print("Mean IoU for ground truth: {}".format(miou))

    # Check if the model type is 'baseline'
    elif model_type == 'baseline':
        # Run the baseline method on the dataset
        yt_objects_baseline(input_folder, output_video, gt_boxes, number_of_frames)


# Example usage:
# TODO: add timing to this
process_shot(20, 'deformable-detr') # MIOU 0.887
process_shot(20, 'detr') # MIOU 0.905
process_shot(500, 'deformable-detr') # 27:44 - MIOU 0.864
process_shot(500, 'detr') # 10:57 - MIOU 0.9065