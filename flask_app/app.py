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

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
BASE_DIR = app.root_path  
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['PROCESSED_FOLDER'] = os.path.join(BASE_DIR, 'processed')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create necessary folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)


def placeholder_deep_learning_process(bayer_image):
    """
    Placeholder for deep learning processing function.
    
    Args:
        bayer_image (numpy.ndarray): Raw Bayer image as numpy array
        This is supposed to be a 2D array
        
    Returns:
        numpy.ndarray: Processed image
    """
    logger.info("Processing image with dimensions: %s", bayer_image.shape)
    bayer_image = bayer_image[:, :, 0] # during development only since input is a 3D array. bayer is supposed to be a 2D one 
    # This is where you would implement your deep learning processing
    # For now, we'll just create a simple debayering simulation
    # In a real implementation, you would replace this with your deep learning model
    
    # Simulate processing time
    time.sleep(3)
    
    # Simple debayering simulation (not accurate, just for demonstration)
    # Assuming RGGB Bayer pattern
    height, width = bayer_image.shape[:2]
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Very simplified debayering (this is not correct debayering, just a placeholder)
    # In reality, you would use a proper demosaicing algorithm or your deep learning model
    rgb_image[:, :, 0] = bayer_image  # Red channel
    rgb_image[:, :, 1] = bayer_image  # Green channel
    rgb_image[:, :, 2] = bayer_image  # Blue channel
    
    return rgb_image


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
        input_filename = f"{unique_id}_input.raw"
        output_filename = f"{unique_id}_output.png"
        
        # Save the uploaded file
        input_filepath = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
        file.save(input_filepath)
        logger.info(f"Saved input file to {input_filepath}")
        
        # Read the Bayer image
        # Note: In a real app, you might need more metadata about the image dimensions
        # and Bayer pattern to correctly read and process it
        img = cv2.imread(input_filepath)

        try:
            # If metadata contains width and height, use those
            if 'width' in metadata and 'height' in metadata:
                width = int(metadata['width'])
                height = int(metadata['height'])
            else:
                width, height = img.shape[:2]
            
            # Create numpy array from raw bytes
            bayer_array = np.array(img, dtype=np.uint8)
            logger.info(f"Converted Bayer data to array of shape {bayer_array.shape}")
        except Exception as e:
            logger.error(f"Error converting Bayer data to array or error getting shape: {str(e)}")
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
        
        # Process the image
        try:
            processed_image = placeholder_deep_learning_process(bayer_array)
            logger.info(f"Processed image to shape {processed_image.shape}")
        except Exception as e:
            logger.error(f"Error in deep learning processing: {str(e)}")
            return jsonify({'error': f'Error in deep learning processing: {str(e)}'}), 500
        
        # Save the processed image
        output_filepath = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
        Image.fromarray(processed_image).save(output_filepath)
        logger.info(f"Saved processed image to {output_filepath}")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Return the URL to the processed image
        image_url = f"/processed/{output_filename}"
        
        return jsonify({
            'success': True,
            'image_url': image_url,
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


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded image files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/api/test', methods=['GET'])
def test_api():
    """Simple endpoint to test if the API is working."""
    return jsonify({'status': 'API is working!', 'timestamp': time.time()})


if __name__ == '__main__':
    # Run the Flask application
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
