import os
import sys
import torch
from testing_models.detr.detr import detect_objects
from utils import plot_results

def test_detr():
    # set path to test image
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/test-image2.jpg')

    # Load the DETR model
    model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
    model.eval()

    # Call the detect_objects function with an image path and the model
    pil_img, prob, boxes = detect_objects(image_path, model)

    # Visualize the results
    plot_results(pil_img, prob, boxes)

test_detr()