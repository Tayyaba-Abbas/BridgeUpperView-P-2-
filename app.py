import logging
from flask import Flask, request, jsonify, send_file, render_template
import os
from werkzeug.utils import secure_filename
from tabulate import tabulate
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import pandas as pd
import numpy as np
from PIL import Image
import random
import torch.nn.functional as F
import math

from math import radians, sin, cos, sqrt, atan2
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Capture all levels from DEBUG to CRITICAL
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app2.log"),  # Log to file
        logging.StreamHandler()  # Log to console
    ]
)

# Initialize the Flask app
app = Flask(__name__)
logging.info(f"Model file exists: {os.path.exists(r'model/BridgeUpperView_maskrcnn_resnet50_fpn_v2_BridgeUpperView(P#3).pth')}")
# Increase the request timeout
@app.before_request
def set_timeout():
    request.environ['werkzeug.request_timeout'] = 600  # Set timeout to 10 minutes (600 seconds)
# Set the upload folder and allowed extensions
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')  # uploads in the static folder
app.config['RESULTS_FOLDER'] = os.path.join('static', 'results')  # save results here
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG'}
dtype = torch.float32
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB
def resize_with_padding(image, target_size=(640, 640)):
    orig_width, orig_height = image.size
    ratio = min(target_size[0] / orig_width, target_size[1] / orig_height)

    new_width = int(orig_width * ratio)
    new_height = int(orig_height * ratio)

    resized_image = image.resize((new_width, new_height), Image.LANCZOS)


    # Create a blank image with target size and paste the resized image
    new_image = Image.new("RGB", target_size, (0, 0, 0))  # Black padding
    new_image.paste(resized_image, ((target_size[0] - new_width) // 2, (target_size[1] - new_height) // 2))

    return new_image

# Ensure necessary folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Device configuration (use CUDA if available, otherwise use CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")
# Define class names (for the number of classes)
# At the top of your script
class_namess = ['Background', 'Barrier', 'Bridges', 'Divider1', 'Divider2', 'Lane', 'LaneMarker']


# Initialize Mask R-CNN model with pretrained weights
model = maskrcnn_resnet50_fpn_v2(weights=None)

# Get the number of input features for the classifier
in_features_box = model.roi_heads.box_predictor.cls_score.in_features
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

# Get the number of output channels for the Mask Predictor
dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels

# Replace the box predictor
model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features_box, num_classes=len(class_namess))

# Replace the mask predictor
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels=in_features_mask, dim_reduced=dim_reduced, num_classes=len(class_namess))

# Set the model's device and data type
model.to(device=device)

# Add attributes to store the device and model name for later reference
model.device = device
model.name = 'complete_tower_maskrcnn_resnet50_fpn_v2_augmented706_imgs_epochs320'

model.load_state_dict(torch.load(r'model/BridgeUpperView_maskrcnn_resnet50_fpn_v2_BridgeUpperView(P#3).pth', map_location=device))

print("Model weights loaded successfully.")

logging.info("Model weights loaded successfully.")




def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to process and predict on the uploaded image
def process_image(image_path, start_lat, start_lon, end_lat, end_lon):
    try:
        # Load the image
        logging.info(f"Loading image from path: {image_path}")
        test_img = Image.open(image_path).convert("RGB")
        # Resize image while maintaining aspect ratio
        test_img = resize_with_padding(test_img)  # Resize and pad
        print(f"Resized Image Size: {test_img.size}")

        if test_img is None:
            logging.error("Image loading failed. Check image file format or path.")
            raise ValueError("Image loading failed. Check image file format.")

        logging.info(f"Image mode: {test_img.mode}")
        logging.info(f"Type of test_img: {type(test_img)}")

        logging.info(f"Image size before transformation: {test_img.size}")

        # Verify image object
        if test_img:
            logging.info("Image loaded successfully!")
        else:
            logging.error("Failed to load image properly.")
            return

        # Apply transformation
        logging.info("Applying transformation...")
        try:
            logging.info("Manually applying transformation...")
            # Resize the image to (640, 640)
            #resized_img = test_img.resize((640, 640))
            #logging.info(f"Image size after resizing: {resized_img.size}")

            try:
                # Manually convert the resized image to a tensor
                # Convert image to a numpy array (H, W, C)
                img_array = np.array(test_img)
                # Normalize the pixel values to [0, 1] and convert to float32
                img_array = img_array.astype(np.float32) / 255.0

                # Convert to a torch tensor (C, H, W)
                input_tensor = torch.from_numpy(img_array).permute(2, 0, 1)

                logging.info(f"Tensor shape after transformation: {input_tensor.shape}")

                # Convert to batch format (1, C, H, W)
                input_tensor = input_tensor.unsqueeze(0).to(device)  # Assuming model takes batch input
                logging.info(f"Tensor shape after unsqueeze: {input_tensor.shape}")
            except Exception as e:
                logging.error(f"Error during tensor conversion: {str(e)}")
                input_tensor = None

            # Check tensor
            if isinstance(input_tensor, torch.Tensor):
                logging.info("Transformation successful, tensor type: torch.Tensor")
            else:
                logging.error("Transformation failed, input_tensor is not a Tensor.")

            # Check tensor shape
            logging.info(f"Tensor size after transformation: {input_tensor.size()}")

            # Convert to batch format if needed
            #input_tensor = input_tensor.unsqueeze(0).to(device)  # Assuming model takes batch input

            logging.info(f"Tensor shape after unsqueeze: {input_tensor.shape}")
        except Exception as e:
            logging.error(f"Error during transformation: {str(e)}")
        # If transformation succeeds, you can process further
        if input_tensor is not None and input_tensor.size(0) > 0:
             logging.info(f"Input tensor size: {input_tensor.size()}, dtype: {input_tensor.dtype}")
        else:
             logging.error("Input tensor is None or empty.")

        if input_tensor is not None:
            try:
                model.eval()
                torch.set_num_threads(1)  # Avoid deadlocks
                model.to("cpu")
                input_tensor = input_tensor.to("cpu")

                #logging.info(f"Memory usage: {psutil.virtual_memory().percent}%")


                logging.info("Starting inference...")

                with torch.no_grad():
                    model_output = model(input_tensor)
                logging.info("Inference completed.")

                if model_output:
                    logging.info(f"Model output keys: {model_output[0].keys()}")
                    boxes = model_output[0]['boxes']
                    labels = model_output[0]['labels']
                    scores = model_output[0]['scores']
                    logging.info(f"Raw Prediction : Boxes: {boxes}, Labels: {labels}, Scores: {scores}")
                    """
                    # Filter low-confidence predictions
                    high_confidence_preds = scores > 0.9
                    if high_confidence_preds.any():
                        logging.info("Filtered predictions based on threshold.")
                        boxes = boxes[high_confidence_preds]
                        labels = labels[high_confidence_preds]
                        scores = scores[high_confidence_preds]
                        logging.info(f"Filtered boxes: {boxes}, Labels: {labels}, Scores: {scores}")
                    else:
                        logging.warning("No high-confidence predictions.")
                else:
                    logging.error("Empty model output.")
                    """
            except RuntimeError as e:
                logging.error(f"Runtime error during inference: {str(e)}")
            except Exception as e:
                logging.error(f"General inference error: {str(e)}")

        else:
            logging.error("Transformation failed, tensor is None.")


        # Set the confidence threshold
        threshold = 0.85
        # Filter out predictions based on the threshold
        scores_mask = model_output[0]['scores'] > threshold
        logging.info(f"Filtered {scores_mask.sum()} predictions based on threshold.")
        print(f"Filtered masks, remaining predictions: {scores_mask.sum()}")
        class_names = ['Background', 'Barrier', 'Bridges', 'Divider1', 'Divider2', 'Lane', 'LaneMarker']
        pred_masks = model_output[0]['masks'][scores_mask]
        pred_labels = [class_names[label] for label in model_output[0]['labels'][scores_mask]]
        pred_bboxes = model_output[0]['boxes'][scores_mask]

        # Resize the masks to the original image size
        target_size = (test_img.size[1], test_img.size[0])  # (W, H)
        pred_masks = F.interpolate(pred_masks, size=target_size, mode='bilinear', align_corners=False)
        pred_masks = (pred_masks >= threshold).squeeze(1).bool()
        logging.info(f"Processing predictions: {len(pred_labels)} components detected.")

        print(f"Predictions processed. Number of masks: {len(pred_labels)}")


        print("Annotating the image with predictions...")
        logging.info(f"Annotating the image with predictions...")
        # Define custom color mapping for each class
        # Assuming class_names is a list of category names
        
        class_names = ['Background', 'Barrier', 'Bridges', 'Divider1', 'Divider2', 'Lane', 'LaneMarker']


        # Color mapping for each class

        # Color mapping for each class
        int_colors = {
           0: (0, 0, 0),         # Background - black
           1: (255, 0, 0),       # Barrier - red
           2: (128, 0, 128),     # Bridges - purple
           3: (0, 255, 255),     # Divider1 - cyan
           4: (255, 165, 0),     # Divider2 - orange
           5: (0, 255, 0),       # Lane - green
           6: (255, 255, 0)      # LaneMarker - yellow
        }
         # Extract predicted colors for the predicted labels
        pred_colors = []
        for label in pred_labels:
            try:
                index = class_names.index(label)
                if index > 0:  # Ignore background
                    pred_colors.append(int_colors[index])
                else:
                    print(f"Label {label} is background and ignored.")
            except ValueError:
                print(f"Label {label} is not valid. Available classes: {class_names[1:]}")

        # Annotate the image with masks, bounding boxes, and labels for all components
        # Convert tensor from float32 to uint8 (values must be in range 0-255)
        annotated_tensor = (input_tensor[0].cpu() * 255).clamp(0, 255).byte()
        logging.info(f"Converted tensor dtype: {annotated_tensor.dtype}")  # Debug dtype

        for i, (label, mask, bbox) in enumerate(zip(pred_labels, pred_masks, pred_bboxes)):

           # Get the class index from class_names (ensure it matches the class label)
           if label in class_names:
              class_idx = class_names.index(label)
           else:
              class_idx = 0  # Default to 'background' if the label is not found

            # Get the color for this class
           color = int_colors.get(class_idx, (0, 0, 0))  # Default to black if the color is not defined


            # Draw the mask using the defined color
           annotated_tensor = draw_segmentation_masks(image=annotated_tensor, masks=mask.unsqueeze(0), alpha=0.2, colors=[color])
           logging.info(f"draw_segmentation_masks")
           annotated_tensor = draw_bounding_boxes(
             image=annotated_tensor,
             boxes=pred_bboxes,
             labels=[f"{label}\n{prob*100:.2f}%" for label, prob in zip(pred_labels, scores_mask)],  # Ensure class names and confidence scores appear
             colors=pred_colors,
             width=3  # Increase this number to make the boxes thicker
            )
           

        print("Finished annotating the image.")
        logging.info(f"Finished annotating the image.")
        # Convert the annotated tensor back to an image
        annotated_img = transforms.ToPILImage()(annotated_tensor)
        logging.info("Annotation complete.")
      
        # Initialize bridge dimensions
        bridge_width = None
        bridge_height = None
        lane_marker_width = None
        lane_marker_height = None
       
        # Initialize lane width
        lane_widths = []
        for label, bbox in zip(pred_labels, pred_bboxes):
            x, y, width, height = bbox
            if label == "Bridges" and bridge_width is None:
                bridge_width = width
                bridge_height = height
            elif label == "Lane":
                lane_widths.append(width)  # Save all Lane widths
            elif label == "LaneMarker" and lane_marker_width is None:
               lane_marker_width = width
               lane_marker_height = height
       
        # Print all lane widths
        for i, lw in enumerate(lane_widths):
           print(f"Lane {i+1} Width: {lw:.2f} pixels")
        # Output first LaneMarker dimensions
        if lane_marker_width is not None:
            print(f"Width of detected LaneMarker: {lane_marker_width:.2f} pixels")
        if lane_marker_height is not None:
            print(f"Height of detected LaneMarker: {lane_marker_height:.2f} pixels")
        if bridge_width is not None:
            print(f"Width of detected bridge: {bridge_width:.2f} pixels")
        if bridge_height is not None:
            print(f"Height of detected bridge: {bridge_height:.2f} pixels")
       
       
        # Optionally, save to a dictionary or DataFrame
        dimension_data = {
           "Lane Index": list(range(1, len(lane_widths) + 1)),
           "Lane Width (pixels)": lane_widths
        }
       
        import pandas as pd
        df_lanes = pd.DataFrame(dimension_data)
        print(df_lanes)
        # Output
        if bridge_width is not None:
            print(f"Width of detected Bridge: {bridge_width:.2f} pixels")
        if bridge_height is not None:
            print(f"Height of detected Bridge: {bridge_height:.2f} pixels")
        
        else:
            print("No 'Lane' detected in the predictions.")
        lane_count = pred_labels.count("Lane")
        # Check for Divider1 and Divider2 in predictions
        if "Divider2" in pred_labels:
            divider_type= "General barrier of rectangular geometry"
            print("Divider Type: General barrier of rectangular geometry")
        
        if "Divider1" in pred_labels:
            divider_type= "Barrier with extended bottom outside"
            print("Divider Type: Barrier with extended bottom outside")
        if lane_widths:  # check if the list is not empty
            total_lane_width = sum(lane_widths)
            print(f"Total Lane Width: {total_lane_width:.2f} pixels")
        else:
            total_lane_width = None
            print("No lanes detected.")
        standard_lanemarker_height=3
        width = ((standard_lanemarker_height / min(lane_marker_width, lane_marker_height)) * bridge_width) + 0.12    #both sides         yellow marker
        R = 6371.0

        # Convert degrees to radians
        lat1_rad = math.radians(start_lat)
        lon1_rad = math.radians(start_lon)
        lat2_rad = math.radians(end_lat)
        lon2_rad = math.radians(end_lon)
    
        # Differences
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
    
        # Haversine formula
        a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
        bridge_length_km = R * c
        bridge_length_meters = bridge_length_km * 1000
        print(f"The total length of the bridge is approximately {bridge_length_meters:.2f} meters.")
        #print(f"Estimated real-world width of the bridge: {width.item():.2f} meters")
        dimension_data = {
            "Component": ["Bridge"],
            "Height (pixels)": [f"{bridge_height:.2f}"],
            "Width (pixels)": [f"{bridge_width:.2f}"],
            "Estimated Length (m)": [f"{bridge_length_meters:.2f}"],
            "Estimated Width (m)": [f"{width:.2f}"]
        }
        
        
        dimension_df = pd.DataFrame(dimension_data)
        print("\n")
        # Display the DataFrame in a nice table format using tabulate
        print("Final Dimensions Table:")
        print(tabulate(dimension_df, headers='keys', tablefmt='grid'))
        print("\n")
        # Return the test image, annotated image, and dimension table
        # Inside process_image function before returning
        print(f"Returning images and dimensions: {test_img}, {annotated_img}, {dimension_df.shape}")
        logging.info("Dimension data prepared.")
    
        logging.info(f"Dimensions Table:\n{tabulate(dimension_df, headers='keys', tablefmt='grid')}\n")
        return test_img, annotated_img, dimension_df, lane_count, divider_type
        

    except Exception as e:
        logging.error(f"Error processing image: {e}")
        print(f"Error in processing image: {str(e)}")
        raise





@app.route('/predict', methods=['POST'])
def predict():
    try:
        logging.info("Received prediction request.")
        print("Received POST request")
        print(f"Request Files: {request.files}")

        # Check if file is part of the request
        if 'file' not in request.files:
            print("No file part")
            logging.warning("No file part in request.")
            return jsonify({'error': 'No file part'}), 400

        # Get the file from the request
        file = request.files['file']
        start_lat = float(request.form["start_lat"])
        start_lon = float(request.form["start_lon"])
        end_lat = float(request.form["end_lat"])
        end_lon = float(request.form["end_lon"])
        
        if file.filename == '':
            print("No selected file")
            logging.warning("No selected file.")
            return jsonify({'error': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            print(f"File received: {file.filename}")

            # Secure the filename and save it in the correct folder
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)  # Save the uploaded file to the uploads folder
            logging.info(f"File saved at {file_path}")

            original_img, annotated_img, dimensions, lane_count, divider_count = process_image(file_path,start_lat, start_lon, end_lat, end_lon )
            print(f"original_img: {original_img}, annotated_img: {annotated_img}, dimensions: {dimensions}")
            # Ensure original_img and annotated_img are PIL images
            original_img = Image.fromarray(original_img.numpy()) if isinstance(original_img, torch.Tensor) else original_img
            annotated_img = Image.fromarray(annotated_img.numpy()) if isinstance(annotated_img, torch.Tensor) else annotated_img

            # Save the original image to uploads folder
            original_img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            original_img.save(original_img_path)
            print("def")

            # Save the annotated image to the results folder
            annotated_img_path = os.path.join(app.config['RESULTS_FOLDER'], f"annotated_{filename}")
            annotated_img.save(annotated_img_path)

            # Assuming dimensions is a DataFrame, convert tensor values to floats and truncate to 2 decimal places
            dimensions['Height (pixels)'] = dimensions['Height (pixels)'].apply(
               lambda x: round(float(x), 2) if isinstance(x, torch.Tensor) else x)

            dimensions['Width (pixels)'] = dimensions['Width (pixels)'].apply(
               lambda x: round(float(x), 2) if isinstance(x, torch.Tensor) else x)

            dimensions['Estimated Length (m)'] = dimensions['Estimated Length (m)'].apply(
               lambda x: round(float(x), 2) if isinstance(x, torch.Tensor) else x)

            dimensions['Estimated Width (m)'] = dimensions['Estimated Width (m)'].apply(
               lambda x: round(float(x), 2) if isinstance(x, torch.Tensor) else x)

            # Convert the DataFrame to a list of tuples
            dimensions_list = dimensions.values.tolist()
            logging.info("Prediction successfully completed.")


            # Return the result page with images and dimensions
            return render_template('after_pred.html',
                                   original_img=filename,
                                   annotated_img=f"annotated_{filename}",
                                   dimensions=dimensions_list,
                                   lane_count=lane_count,
                                   divider_count=divider_count)

        return jsonify({'error': 'Invalid file format'}), 400

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        logging.error(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500

# Route for the home page
@app.route('/')
def index():
    return render_template('home.html')  # Ensure this file exists in the 'templates' folder

if __name__ == '__main__':
    logging.info("Starting Flask app...")
    app.run(debug=True)