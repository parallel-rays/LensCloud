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
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 16MB max upload

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
    logger.info(f"In def remove_black_level() np.max(img): {np.max(img)}")
    return img

# def remove_black_level(img, black_lv=64, white_lv=1023):
#     # img.black_level_per_channel   = [64, 64, 64, 63]
#     # img.white_level               = 1023

#     bl_map = np.zeros_like(img)
#     bl_map[0::2, 0::2] = 64    # Gr
#     bl_map[1::2, 0::2] = 64    # R
#     bl_map[0::2, 1::2] = 64    # B
#     bl_map[1::2, 1::2] = 63    # Gb
#     img = img - bl_map

#     img = np.maximum(img.astype(np.float32), 0) / (white_lv-black_lv)
#     logger.info(f"In def remove_black_level()  np.max(img): {np.max(img)}")
#     return img

def white_balance(raw_norm, cam_wb):
    # r_gain, g_gain, b_gain, _ = [1.6075353622436523, 1.0, 1.8028168678283691, 0.0]
    r_gain, g_gain, b_gain, _ = cam_wb
    logger.info(f"r_gain, g_gain, b_gain: {r_gain, g_gain, b_gain}")
    raw_norm[0::2, 0::2] *= g_gain
    raw_norm[0::2, 1::2] *= b_gain
    raw_norm[1::2, 0::2] *= r_gain
    raw_norm[1::2, 1::2] *= g_gain
    
    return raw_norm

def extract_bayer_channels(raw, bayer_format='GBGR'):
    logger.info(f"bayer_format: {bayer_format}")
    if bayer_format == 'GBGR':

        # R G
        # G B

        ch_R  = raw[0::2, 0::2]
        ch_Gb = raw[0::2, 1::2]
        ch_Gr = raw[1::2, 0::2]
        ch_B  = raw[1::2, 1::2]
    elif bayer_format == 'RGBG':
        
        # G B
        # R G

        # 2 3
        # 0 1  # the order rawpy reads a bayer filter in

        #ch_Gr = raw[0::2, 0::2]
        #ch_B  = raw[0::2, 1::2]
        #ch_R  = raw[1::2, 0::2]
        #ch_Gb = raw[1::2, 1::2]
        
        ch_Gb = raw[0::2, 0::2]
        ch_B  = raw[0::2, 1::2]
        ch_R  = raw[1::2, 0::2]
        ch_Gr = raw[1::2, 1::2]
        
    else:
        logger.warning("Bayer format not one of 'GBGR' or 'RGBG'")

    raw_combined = np.stack((ch_B, ch_Gb, ch_R, ch_Gr), axis=2)
    logger.info(f"np.sum(ch_R), np.sum(ch_Gb), np.sum(ch_Gr), np.sum(ch_B): {np.sum(ch_R), np.sum(ch_Gb), np.sum(ch_Gr), np.sum(ch_B)}")
    # assert 0, 'Bye bye'
    return np.ascontiguousarray(raw_combined.transpose(2,0,1))

def get_coord(H, W, x=1, y=1):
    xs = np.linspace(-x + x/W, x - x/W, W, dtype=np.float32)
    ys = np.linspace(-y + y/H, y - y/H, H, dtype=np.float32)
    x_grid = np.repeat(xs[np.newaxis, :], H, axis=0)
    y_grid = np.repeat(ys[:, np.newaxis], W, axis=1)
    return np.stack((x_grid, y_grid), axis=0)

# the core inference function
def run_liteisp_on_raw(model, raw=None, device='cpu', bayer_format='GBGR', cam_wb=[1,1,1,1]):
    logger.info(f"Image being passed through the DL Model")
    if raw.ndim == 3:
        raw = raw[..., 0]
    raw_norm = remove_black_level(raw) # trying to normalise after white balance
    # raw_norm = white_balance(raw_norm, cam_wb) # name kept as raw norm since it is used later
    # raw_norm = (raw/255).astype('float32')
    raw_combined = extract_bayer_channels(raw_norm, bayer_format)
    _, H, W = raw_combined.shape
    coord_t = torch.from_numpy(get_coord(H, W)).unsqueeze(0).to(device)
    raw_t   = torch.from_numpy(raw_combined).unsqueeze(0).to(device)
    with torch.no_grad():
        out_t = model(raw_t, coord_t)
    out_np = out_t[0].permute(1,2,0).cpu().numpy()
    return np.clip(out_np,0,1), raw_norm


def deep_learing_processing(raw_img, device='cpu', bayer_format='GBGR', scale_down=False, cam_wb=[1,1,1,1]):
    logger.info("In def deep_learing_processing()")
    logger.info(f"Preparing Processing of image with DL Model with dimensions: {raw_img.shape}",)
    # collapse or convert 3-channel duplicates
    if raw_img.ndim == 3:
        if np.allclose(raw_img[..., 0], raw_img[..., 1]) and np.allclose(raw_img[..., 1], raw_img[..., 2]):
            raw_img = raw_img[..., 0]
            logger.info("All three channels duplicate; using channel 0 only.")
            logger.warning(f"You should NOT be providing an input image with 3 dimensions")
    else:
        logger.info(f"Received RAW Image with {raw_img.ndim} dimension/s") # remove this later

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
    output_padded, raw_norm_padded = run_liteisp_on_raw(model, raw=raw_padded, device=device, bayer_format=bayer_format, cam_wb=cam_wb)
    
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
                bayer_format = 'RGBG'
                logger.info("Processing DNG raw file")
                
                upload_filename = f"{unique_id}_upload.dng" 
                upload_filepath = os.path.join(app.config['UPLOAD_FOLDER'], upload_filename)
                  
                # convert FileStorage obj to bytes
                file_content = file.read()
                file_stream = io.BytesIO(file_content)
                
                with open(upload_filepath, 'wb') as f:
                    f.write(file_content)
                logger.info(f"Saved original DNG file to {upload_filepath}")

                # Read the .dng image directly from memory for speed
                rawpy_obj = rawpy.imread(file_stream)
                cam_wb = rawpy_obj.camera_whitebalance
                logger.info(f"White balance of .dng image: {cam_wb}")
                raw_img = rawpy_obj.raw_image
                # Get raw data and metadata
                pat = rawpy_obj.raw_pattern  # should be array([[3,2],[0,1]])
                desc = rawpy_obj.color_desc.decode()  # expected to be 'RGBG' from the android camera
                if desc != 'RGBG':
                    logger.warning(f"Raw image Bayer format expected to be RGBG, received : {desc}")
                
                logger.info(f"Source pattern: {pat}")
                logger.info(f"Color description: {desc}")
                
                # raw_img = raw_img[1800:2248, 2000:2448]
                # logger.info(f'Cropped the raw dng image to size: {raw_img.shape}')
                # raw_img = raw_img.astype(np.uint8)
                logger.info("Successfully converted the Bayer pattern")
                logger.info(f'np.max(raw_img) in if_dng: {np.max(raw_img)}')
            
            else:
                cam_wb = [1, 1, 1, 1]
                bayer_format = 'GBGR'
                # handle jpg, png and such images
                upload_filename = f"{unique_id}_upload.png"
                upload_filepath = os.path.join(app.config['UPLOAD_FOLDER'], upload_filename)
                file_bytes = file.read()
                raw_img = cv2.imdecode(
                    np.frombuffer(file_bytes, dtype=np.uint8), # this np.uin8 isn't the image's depth
                    cv2.IMREAD_UNCHANGED
                ) # this is a RAW 2D image. note the in the original ZRR dataset, using cv2.imread_unchanged gives a 2D raw array

                cv2.imwrite(upload_filepath, raw_img)
                
                logger.info("Successfully saved 2D raw image to disk in png format. use cv2.imread_unchanged to get back the 2D raw image")
            
            logger.info(f"Done with the is_dng or other file format logic")
            logger.info(f"Raw Image dimensions: {raw_img.shape}, ndim: {raw_img.ndim}")
            
            # Check if image values are in [0,1] range and normalize if needed
            max_val = np.max(raw_img)
            logger.info(f"raw_img.dtype: {raw_img.dtype}")
            logger.info(f"Max value in image: {max_val}")
            if max_val <= 1.0:
                logger.info("Image values in [0,1] range, scaling to [0,255]")
                raw_img = raw_img * 255.0
                # raw_img = raw_img.astype(np.uint8)
                logger.info(f"New: np.max(raw_img): {np.max(raw_img)}")            

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500

        # Process the image
        try:
            # assert 0, 'Cannot pass through the model now'
            processed_img, raw_demosaiced = deep_learing_processing(raw_img, bayer_format=bayer_format, cam_wb=cam_wb)
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
