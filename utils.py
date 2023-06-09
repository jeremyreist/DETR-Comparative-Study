import matplotlib.pyplot as plt
import io, os
from PIL import Image
import numpy as np

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def visualize_detection_results(pil_img, prob, boxes, selected_class, display_output=False, return_image=False):
    """
    This function visualizes the results of object detection by drawing bounding
    boxes around detected objects and displaying class labels with their
    probabilities. Optionally, it can also return the annotated image as a NumPy array.

    Args:
        pil_img (PIL.Image): The input image.
        prob (List[float]): List of probabilities for each detected object.
        boxes (List[List[float]]): List of bounding boxes for each detected object.
        display_output (bool, optional): Whether to display the annotated image. Defaults to False.
        return_image (bool, optional): Whether to return the annotated image as a NumPy array. Defaults to False.
        selected_class (str): The class to be detected.

    Returns:
        np.ndarray: The annotated image as a NumPy array (if return_image is True).
    """

    # Create a new figure with a specified size and display the input image in the figure
    plt.figure(figsize=(16, 9))
    plt.imshow(pil_img)

    # Get the current axes
    ax = plt.gca()

    # Create a list of colors for bounding boxes (repeated to ensure enough colors)
    colors = COLORS * 100

    # Iterate through probabilities, bounding boxes, and colors
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        # Determine the class with the highest probability
        cl = np.argmax(p)

        if selected_class == cl:
            # Create a text label with the class name and probability
            text = f'{CLASSES[cl]}: {p[cl]:0.2f}'

            # Add the text label to the image at the top-left corner of the bounding box
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))
            
            # Draw a bounding box on the image using the specified color and linewidth
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=c, linewidth=3))
            
    # Remove the axes from the figure
    plt.axis('off')

    # Show the annotated image if display_output is True
    if display_output:
        plt.show()

    # Return the annotated image as a NumPy array if return_image is True
    if return_image:
        # Save the plot to a buffer as an RGB image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        im = Image.open(buf)

        # Convert the PIL image to a NumPy array
        img_array = np.array(im)

        # Close the buffer and the figure to release resources
        buf.close()
        plt.close()

        return img_array
    
def compress_mp4_files(input_folder, output_folder, crf_value=28):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Delete all files in the output folder before compressing new files
    for filename in os.listdir(output_folder):
        os.remove(os.path.join(output_folder, filename))

    # Loop through files in input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.mp4'):
            # Define input and output paths
            save_path = os.path.join(input_folder, filename)
            compressed_path = os.path.join(output_folder, filename)

            # Compress using FFmpeg if the compressed file doesn't already exist
            if not os.path.exists(compressed_path):
                os.system(f"ffmpeg -i {save_path} -vcodec libx264 -crf {crf_value} {compressed_path}")


compress_mp4_files('results', 'results/compressed', 40)
