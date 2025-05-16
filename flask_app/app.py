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
import rawpy
import imageio

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

def rgb_to_bayer(rgb_image, pattern='RGGB', gains=(1.0, 1.0, 1.0)):
    """
    Convert an RGB image back to a Bayer pattern single‑channel image,
    applying the inverse of any channel gains.

    Args:
        rgb_image: Input RGB image (HxWx3), assumed uint8 or float32 in [0,255]
        pattern: Bayer pattern arrangement ('RGGB', 'BGGR', 'GRBG', 'GBRG')
        gains:  Tuple of (R_gain, G_gain, B_gain) that were applied in the forward pass

    Returns:
        Single‑channel Bayer pattern image, same dtype as input
    """
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("Input must be an RGB image with 3 channels")

    # Unpack inverse gains
    inv_gains = (1.0 / gains[0], 1.0 / gains[1], 1.0 / gains[2])

    height, width = rgb_image.shape[:2]
    # Preserve dtype (uint8 or float32)
    bayer_image = np.zeros((height, width), dtype=rgb_image.dtype)

    # Map pattern letters to channel indices
    pattern_map = {'R': 0, 'G': 1, 'B': 2}

    if pattern not in ['RGGB', 'BGGR', 'GRBG', 'GBRG']:
        raise ValueError("Pattern must be one of: RGGB, BGGR, GRBG, GBRG")

    # Build 2×2 lookup of which channel goes where
    pattern_matrix = [[pattern[0], pattern[1]],
                      [pattern[2], pattern[3]]]

    for y in range(height):
        for x in range(width):
            color = pattern_matrix[y & 1][x & 1]
            c = pattern_map[color]
            val = rgb_image[y, x, c]
            # apply the inverse gain for that channel
            val = val * inv_gains[c]
            # clip & cast back if needed
            if np.issubdtype(bayer_image.dtype, np.integer):
                val = np.clip(val, 0, 255)
            bayer_image[y, x] = val

    return bayer_image


def bgr_to_bayer(bgr_image, pattern='RGGB'):
    """
    Convert a BGR image (OpenCV default) back to a Bayer pattern single-channel image.
    
    Args:
        bgr_image: Input BGR image (HxWx3)
        pattern: Bayer pattern arrangement ('RGGB', 'BGGR', 'GRBG', 'GBRG')
        
    Returns:
        Single-channel Bayer pattern image
    """
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    # Then convert RGB to Bayer
    return rgb_to_bayer(rgb_image, pattern)


def deep_learing_processing(raw_img, device='cpu', scale_down=True):
    logger.info("Processing image with dimensions: %s", raw_img.shape)
    # collapse or convert 3-channel duplicates
    if raw_img.ndim == 3:
        if np.allclose(raw_img[..., 0], raw_img[..., 1]) and np.allclose(raw_img[..., 1], raw_img[..., 2]):
            raw_img = raw_img[..., 0]
            logger.info("All three channels duplicate; using channel 0 only.")
        else:
            raw_img = bgr_to_bayer(raw_img, pattern="RGGB")
            logger.info("Converted RGB back to RGGB mosaic.")
    H_orig, W_orig = raw_img.shape
    
    # Ensure dimensions are multiples of 8 before any scaling
    pad_bottom = (8 - (H_orig % 8)) % 8
    pad_right = (8 - (W_orig % 8)) % 8
    raw_padded = np.pad(raw_img,
                        ((0, pad_bottom), (0, pad_right)),
                        mode='constant', constant_values=0)
    H_pad, W_pad = raw_padded.shape
    logger.info(f"Padded from {(H_orig, W_orig)} to {(H_pad, W_pad)}")
    
    # Scale down if needed
    if scale_down:
        max_dim = 512
        if max(H_pad, W_pad) > max_dim:
            sf = max_dim / float(max(H_pad, W_pad))
            # Calculate new dimensions that are multiples of 8
            new_H = int(H_pad * sf)
            new_W = int(W_pad * sf)
            # Make them multiples of 8
            new_H = new_H - (new_H % 8)
            new_W = new_W - (new_W % 8)
            # Resize directly to dimensions that are multiples of 8
            raw_padded = cv2.resize(raw_padded, (new_W, new_H), interpolation=cv2.INTER_AREA)
            H_pad, W_pad = raw_padded.shape
            logger.info(f"Resized to {(H_pad, W_pad)} (multiples of 8)")
    
    # run through the model
    logger.info("Passing the image through the deep learning model")
    output_padded, raw_norm_padded = run_liteisp_on_raw(model, raw_array=raw_padded, device=device)
    
    # un-pad back to the padded (and possibly resized) size
    logger.info("Unpadding to padded/resized size")
    processed_img = output_padded[:H_pad, :W_pad, :]
    raw_demosaiced = raw_norm_padded[:H_pad, :W_pad]
    
    # convert to 8-bit and BGR
    proc8 = (processed_img * 255.0).round().astype(np.uint8)
    raw8 = (raw_demosaiced * 255.0).round().astype(np.uint8)
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

        
        # Generate unique filename for the processed images
        unique_id = str(uuid.uuid4())
        output_filename = f"{unique_id}_output.png"
        raw_vis_filename = f"{unique_id}_raw_vis.png"

        # Check if the file is a DNG raw file
        is_dng = file.filename.lower().endswith('.dng')

        try:
            if is_dng:
                logger.info("Processing DNG raw file")
                
                upload_filename = f"{unique_id}_upload.dng" 
                upload_filepath = os.path.join(app.config['UPLOAD_FOLDER'], upload_filename)
                  
                # Save the file content to a BytesIO object
                file_content = file.read()
                file_stream = io.BytesIO(file_content)
                
                with open(upload_filepath, 'wb') as f:
                    f.write(file_content)
                logger.info(f"Saved original DNG file to {upload_filepath}")

                # Process the DNG file directly from memory
                with rawpy.imread(file_stream) as raw:
                    # Get raw data and metadata
                    data = raw.raw_image_visible.copy().astype(np.float32)
                    pat = raw.raw_pattern  # e.g. array([[3,2],[0,1]])
                    desc = raw.color_desc.decode()  # e.g. "RGBG", "GRBG", etc.
                    
                    logger.info(f"Source pattern: {pat}")
                    logger.info(f"Color description: {desc}")
                    
                    # Create a mapping from the color description to standard RGBG indices
                    # Standard mapping: R=0, G=1, B=2, G2=1 (second green is also green)
                    color_map = {}
                    for i, color in enumerate(desc):
                        if color == 'R':
                            color_map[i] = 0  # R maps to 0
                        elif color == 'G':
                            color_map[i] = 1  # G maps to 1
                        elif color == 'B':
                            color_map[i] = 2  # B maps to 2
                    
                    # Define RGGB target pattern: 0=R, 1=G, 2=B
                    target_pattern = np.array([[0, 1],
                                            [1, 2]], dtype=np.uint8)
                    
                    # Get pattern height and width
                    pat_h, pat_w = pat.shape
                    
                    # Create output array with same dimensions
                    raw_img = np.zeros_like(data)
                    
                    # For each color in the RGGB target pattern
                    for tgt_y in range(pat_h):
                        for tgt_x in range(pat_w):
                            # Get the target color (0=R, 1=G, 2=B)
                            target_color = target_pattern[tgt_y, tgt_x]
                            
                            # Find where this color appears in the source pattern
                            for src_y in range(pat_h):
                                for src_x in range(pat_w):
                                    src_idx = pat[src_y, src_x]
                                    if color_map.get(src_idx) == target_color:
                                        # Calculate the stride between the original pattern and the target pattern
                                        y_stride = (pat_h + src_y - tgt_y) % pat_h
                                        x_stride = (pat_w + src_x - tgt_x) % pat_w
                                        
                                        # Copy the pixels from source to target position
                                        raw_img[tgt_y::pat_h, tgt_x::pat_w] = data[y_stride::pat_h, x_stride::pat_w]
                    # raw_img = raw_img.astype(np.uint10)
                    #raw_img = raw_img / 1024
                    # raw_img = raw_img * 255
                    raw_img = raw_img.astype(np.uint8)
                    logger.info("Successfully converted the Bayer pattern")
                    logger.info(np.max(raw_img))
            
            else:
                # For non-DNG files, read directly using OpenCV
                upload_filename = f"{unique_id}_upload.png"
                file_bytes = np.frombuffer(file.read(), np.uint8)
                raw_img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED).astype(np.float32)
                cv2.imwrite(upload_filename, raw_img)
                logger.info("Successfully saved 3D raw image to disk")
                logger.info("Read non-DNG image directly from memory")
            
            logger.info(f"Image dimensions: {raw_img.shape}, ndim: {raw_img.ndim}")
            
            # Check if image values are in [0,1] range and normalize if needed
            max_val = np.max(raw_img)
            logger.info(f"Max value in image: {max_val}")
            if max_val <= 1.0:
                logger.info("Image values in [0,1] range, scaling to [0,255]")
                raw_img = raw_img * 255.0
            
            logger.info(f"Processed raw image {raw_img.shape}")

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
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
        cv2.imwrite(processed_outpath, processed_img)
        cv2.imwrite(raw_vis_outpath, raw_demosaiced)
        logger.info(f"Saved processed image to {processed_outpath}")
        logger.info(f"Saved raw demosaiced image to {raw_vis_outpath}")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Return the URL to the processed image
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
