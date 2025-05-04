"""
Bayer Image Processing Web Application
====================================
This Flask web application receives raw Bayer images from a smartphone,
processes them using a placeholder deep learning function, and returns
the processed images.
"""

import os
import uuid
from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
from PIL import Image
import io
import logging
import time
import json
import cv2
import torch
from ispnet import LiteISPNet

# set up logging functionalites
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
BASE_DIR = app.root_path  
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['PROCESSED_FOLDER'] = os.path.join(BASE_DIR, 'processed')
app.config['RAW_VISUALIZED'] = os.path.join(BASE_DIR, 'raw_visualized')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# create upload and processsed image folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
os.makedirs(app.config['RAW_VISUALIZED'], exist_ok=True)


# load the model once (since it's costly to do it again and again)
model = LiteISPNet()
# load the trained model
device = 'cpu'
model.load_state_dict(torch.load('trained_models/ispnet_model.pth')['state_dict'])
# model.load_state_dict(torch.load('trained_models/ispnet_model.onnx')['state_dict'])
model.to(device).eval()

# some helper function for data preprocessing
def remove_black_level(img, black_lv=63, white_lv=4*255):
    img = np.maximum(img.astype(np.float32)-black_lv, 0) / (white_lv-black_lv)
    return img

def extract_bayer_channels(raw):
    ch_R  = raw[0::2, 0::2]
    ch_Gb = raw[0::2, 1::2]
    ch_Gr = raw[1::2, 0::2]
    ch_B  = raw[1::2, 1::2]
    raw_combined = np.stack((ch_B, ch_Gb, ch_R, ch_Gr), axis=2)
    return np.ascontiguousarray(raw_combined.transpose(2,0,1))

def get_coord(H, W, x=1, y=1):
    xs = np.linspace(-x + x/W, x - x/W, W, dtype=np.float32)
    ys = np.linspace(-y + y/H, y - y/H, H, dtype=np.float32)
    x_grid = np.repeat(xs[np.newaxis, :], H, axis=0)
    y_grid = np.repeat(ys[:, np.newaxis], W, axis=1)
    return np.stack((x_grid, y_grid), axis=0)

# the core inference function
def run_liteisp_on_raw(model, raw_png_path=None, raw_array=None, device='cpu'):
    if raw_array is None:
        raw = cv2.imread(raw_png_path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"Can't read {raw_png_path}")
    else:
        raw = raw_array
    if raw.ndim == 3:
        raw = raw[..., 0]
    raw_norm = remove_black_level(raw)
    raw_combined = extract_bayer_channels(raw_norm)
    _, H, W = raw_combined.shape
    coord_t = torch.from_numpy(get_coord(H, W)).unsqueeze(0).to(device)
    raw_t   = torch.from_numpy(raw_combined).unsqueeze(0).to(device)
    with torch.no_grad():
        out_t = model(raw_t, coord_t)
    out_np = out_t[0].permute(1,2,0).cpu().numpy()
    return np.clip(out_np,0,1), raw_norm

def color_to_rggb(img, format='RGB'):
    """
    Convert a demosaicked color image back to RGGB Bayer pattern.
    Works with either RGB or BGR input format.
    
    Parameters:
    -----------
    img : numpy.ndarray
        Color image with shape [height, width, 3]
    format : str
        Input image format, either 'RGB' or 'BGR' (default: 'RGB')
        
    Returns:
    --------
    numpy.ndarray
        RGGB Bayer pattern image with shape [height, width]
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Input must be a color image with shape [height, width, 3]")
    
    if format.upper() not in ['RGB', 'BGR']:
        raise ValueError("Format must be either 'RGB' or 'BGR'")
    
    height, width, _ = img.shape
    bayer = np.zeros((height, width), dtype=img.dtype)
    
    if format.upper() == 'RGB':
        # R channel is at index 0
        r_idx, g_idx, b_idx = 0, 1, 2
    else:  # BGR
        # R channel is at index 2
        r_idx, g_idx, b_idx = 2, 1, 0
    
    # R at positions (even row, even column)
    bayer[0::2, 0::2] = img[0::2, 0::2, r_idx]
    
    # G at positions (even row, odd column)
    bayer[0::2, 1::2] = img[0::2, 1::2, g_idx]
    
    # G at positions (odd row, even column)

    bayer[1::2, 0::2] = img[1::2, 0::2, g_idx]    
    # B at positions (odd row, odd column)
    bayer[1::2, 1::2] = img[1::2, 1::2, b_idx]

    return bayer

def deep_learing_processing(raw_img, device='cpu'):
    """
    Placeholder for deep learning processing function.
    
    Args:
        bayer_image (numpy.ndarray): Raw Bayer image as numpy array
        This is supposed to be a 2D array. (can be a 3D too but each channel is the other's duplicate)
        
    Returns:
        numpy.ndarray: Processed image
    """
    logger.info("Processing image with dimensions: %s", raw_img.shape)
    if raw_img.ndim == 3: # if it is a 3D array
        if np.sum(raw_img[..., 0]) == np.sum(raw_img[..., 1] ): # all the channels are duplicate      
            raw_img = raw_img[..., 0]
            logger.info(f"All three channels are duplicate. Extracted the first channel. Dimensions: {raw_img.shape}")
        else:
            logger.info('Converting R, G, B image back to RGGB mosaic')
            raw_img = color_to_rggb(raw_img, format="BGR")
            logger.info(f'Conversion successful, new shape: {raw_img.shape}')

    H, W = raw_img.shape
    H_pad = ((H + 7) // 8) * 8
    W_pad = ((W + 7) // 8) * 8
    pad_bottom = H_pad - H
    pad_right  = W_pad - W

    raw_padded = np.pad(raw_img, ((0, pad_bottom), (0, pad_right)), mode='constant', constant_values=0)
    logger.info(f'Padded raw image from shape {H, W} to {raw_padded.shape}')
    # this function call returns the raw image demosaiced
    # and the processed image (the raw image processed by the model)
    logger.info('Passing the image through the deep learning model')
    output_padded, raw_norm_padded = run_liteisp_on_raw(model, raw_array=raw_padded, device=device)

    # now, unpad the images (since they were most likely padded. if no padding was done, they the 2 lines below do not make any changes)
    logger.info('unpadding any applied padding')
    processed_img = output_padded[:H, :W, :]
    raw_demosaiced = raw_norm_padded[:H, :W]

    # scale + convert
    proc8 = (processed_img * 255.0).round().astype(np.uint8)
    raw8  = (raw_demosaiced * 255.0).round().astype(np.uint8)
    # RGB→BGR for OpenCV
    proc8_bgr = cv2.cvtColor(proc8, cv2.COLOR_RGB2BGR)
    return proc8_bgr, raw8



@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process_image():
    """
    API endpoint to process raw Bayer images.
    
    Expects:
        - A POST request with form data containing a 'bayer_image' file
        - Optionally metadata as JSON in the 'metadata' field
        
    Returns:
        - JSON response with the URL to the processed image
    """
    try:
        logger.info("Received image processing request")
        start_time = time.time()
        
        # Check if image file is present in request
        if 'bayer_image' not in request.files:
            logger.error("No file part in the request")
            return jsonify({'error': 'No file part in the request'}), 400
        
        file = request.files['bayer_image']
        logger.info(f"request.files['bayer_image']: {request.files['bayer_image']}")
        
        # Check if file is selected
        if file.filename == '':
            logger.error("No file selected")
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if metadata is provided
        metadata = {}
        if 'metadata' in request.form:
            metadata_str = request.form.get('metadata', '{}')
            try:
                metadata = json.loads(metadata_str)
                logger.info(f"Received metadata: {metadata}")
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing metadata JSON: {e}")

        
        # Generate unique filename for the uploaded and processed images
        unique_id = str(uuid.uuid4())
        input_filename = f"{unique_id}_input.png"
        output_filename = f"{unique_id}_output.png"
        raw_vis_filename = f"{unique_id}_raw_vis.png"

        # Save the uploaded file
        input_filepath = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        file.save(input_filepath)
        logger.info(f"Saved input file to {input_filepath}")
        
        
        try:
            # Read the image with IMREAD_UNCHANGED to preserve all channels and bit depth
            raw_img = cv2.imread(input_filepath, cv2.IMREAD_UNCHANGED).astype(np.float32)
            logger.info(f"Image dimensions: {raw_img.shape}, ndim: {raw_img.ndim}")
            
            # Check if image values are in [0,1] range and normalize if needed
            max_val = np.max(raw_img)
            logger.info(f"Max value in image: {max_val}")
            
            if max_val <= 1.0:
                logger.info("Image values in [0,1] range, scaling to [0,255]")
                raw_img = raw_img * 255.0
                
            logger.info(f"Read raw image from disk {raw_img.shape}")
        except Exception as e:
            logger.error(f"Error reading raw image from disk {str(e)}")
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
        
        # Process the image
        try:
            processed_img, raw_demosaiced = deep_learing_processing(raw_img)
            logger.info(f"Processed image to shape {processed_img.shape}")
            logger.info(f"Demosaiced raw image to shape {raw_demosaiced.shape}")
        except Exception as e:
            logger.error(f"Error in deep learning processing: {str(e)}")
            return jsonify({'error': f'Error in deep learning processing: {str(e)}'}), 500
        
        # Save the processed image
        processed_outpath = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
        raw_vis_outpath = os.path.join(app.config['RAW_VISUALIZED'], raw_vis_filename)
        # Image.fromarray(processed_img).save(output_filepath)
        cv2.imwrite(processed_outpath, processed_img)
        cv2.imwrite(raw_vis_outpath, raw_demosaiced)
        logger.info(f"Saved processed image to {processed_outpath}")
        logger.info(f"Saved raw demosaiced image to {raw_vis_outpath}")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Return the URL to the processed image. this will then be used in the https request to be downloaded
        image_url = f"/processed/{output_filename}"
        raw_vis_url = f"/raw_visualized/{raw_vis_filename}"

        return jsonify({
            'success': True,
            'image_url': image_url,
            'raw_vis_url': raw_vis_url,
            'processing_time': processing_time,
            'message': 'Image processed successfully'
        })
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/processed/<filename>')
def processed_file(filename):
    """Serve processed image files."""
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route('/raw_visualized/<filename>')
def raw_visualized_file(filename):
    """Server the raw image visuzlied via demosaicking"""
    return send_from_directory(app.config['RAW_VISUALIZED'], filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded image files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/api/test', methods=['GET'])
def test_api():
    """Simple endpoint to test if the API is working."""
    return jsonify({'status': 'API is working!', 'timestamp': time.time()})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
