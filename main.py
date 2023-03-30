import os, re, cv2, torch
from PIL import Image
from testing_models.detr.detr import detect_objects
from utils import visualize_detection_results
from tqdm import tqdm

def detr_test():
    # set path to test image
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/test-image2.jpg')

    # Load the DETR model
    model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
    model.eval()

    # Call the detect_objects function with an image path and the model
    pil_img, prob, boxes = detect_objects(image_path, model)

    # Visualize the results
    visualize_detection_results(pil_img, prob, boxes)


def detr_yt_objects(input_folder, output_video):
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

    for frame_file in tqdm(frame_files):
        frame_path = os.path.join(input_folder, frame_file)

        # Load the frame as a PIL image
        pil_img = Image.open(frame_path)

        # Call the detect_objects function with the frame and the model
        _, prob, boxes = detect_objects(frame_path, model)

        # Get the annotated image using the plot_results function
        annotated_image = visualize_detection_results(pil_img, prob, boxes, return_image=True)

        # Convert the annotated image to BGR format (required by OpenCV)
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        # Resize the annotated image to match the dimensions of the first frame
        annotated_image = cv2.resize(annotated_image, (frame_width, frame_height))

        # Write the processed frame to the output video
        out.write(annotated_image)

    # Release the VideoWriter object and close all windows
    out.release()
    cv2.destroyAllWindows()


input_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/YouTube-Objects/car/data/0001/shots/002/')
output_video = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/output_video_002.mp4')
detr_yt_objects(input_folder, output_video)