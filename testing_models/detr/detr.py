from PIL import Image
import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# This function is for output bounding box post-processing. It converts bounding boxes
# from center-x, center-y, width, height (cxcywh) format to x_min, y_min, x_max, y_max (xyxy) format.
def box_cxcywh_to_xyxy(x):
    # Split the input tensor into its components: center-x, center-y, width, and height
    x_c, y_c, w, h = x.unbind(1)

    # Calculate the coordinates by subtracting/adding half of the width and height from the center coordinates
    x_min = x_c - 0.5 * w
    y_min = y_c - 0.5 * h
    x_max = x_c + 0.5 * w
    y_max = y_c + 0.5 * h

    # Combine the calculated coordinates into a list of bounding box values
    b = [x_min, y_min, x_max, y_max]

    # Stack the bounding box values along the second dimension to create the output tensor
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    # Rescale bounding boxes from [0; 1] to the original image scale
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def detect_objects(image_path, model):
    # standard PyTorch mean-std input image normalization
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load image
    im = Image.open(image_path)

    # Mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # Propagate through the model
    outputs = model(img)

    # Keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.9

    # Convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)

    return im, probas[keep], bboxes_scaled
