import rawpy
import numpy as np
import cv2

def extract_bayer_channels(raw):
    """
    Extracts the Bayer channels from a raw image.  This function assumes
    the input 'raw' is a 2D array representing the raw pixel data.

    Args:
        raw (numpy.ndarray): 2D array of raw pixel data.

    Returns:
        tuple: (ch_R, ch_Gr, ch_B, ch_Gb) - the separated color channels.
    """
    ch_B  = raw[1::2, 1::2] 
    ch_Gb = raw[0::2, 1::2] 
    ch_R  = raw[0::2, 0::2] 
    ch_Gr = raw[1::2, 0::2] 
    return ch_R, ch_Gr, ch_B, ch_Gb

def apply_white_balance_only(dng_file_path):
    """
    Applies white balance correction to raw pixel data from a DNG file,
    handling RGBG Bayer pattern, separating channels, applying WB, and
    recombining into a single 2D array.

    Args:
        dng_file_path (str): The path to the DNG file.

    Returns:
        tuple: A tuple containing:
            - raw_data_wb (numpy.ndarray): The white balance-corrected 2D raw pixel data.
            - original_shape (tuple): The original shape (height, width) of the raw image.
            Returns None on error.
    """
    try:
        raw  = rawpy.imread(dng_file_path)
        
        # raw_data = raw.raw_image
        # camera_wb = raw.camera_whitebalance
        # height, width = original_shape

        # raw_data_2d = raw_data.reshape(original_shape)
        raw_data_2d = raw.raw_image
        original_shape = raw_data_2d.shape
        # Extract channels *before* applying white balance
        # ch_R, ch_Gr, ch_B, ch_Gb = extract_bayer_channels(raw_data_2d)

        # # Apply white balance to each channel
        # ch_R_wb  = ch_R  * camera_wb[0]
        # ch_Gr_wb = ch_Gr * camera_wb[1] 
        # ch_Gb_wb = ch_Gb * camera_wb[1]
        # ch_B_wb  = ch_B  * camera_wb[2]

        # # Recombine the channels into a single 2D array
        # raw_data_wb = np.zeros_like(raw_data_2d) # Create an empty array with the same shape
        # raw_data_wb[0::2, 0::2] = ch_R_wb
        # raw_data_wb[0::2, 1::2] = ch_Gb_wb
        # raw_data_wb[1::2, 0::2] = ch_Gr_wb
        # raw_data_wb[1::2, 1::2] = ch_B_wb

        return raw_data_2d, original_shape

    except Exception as e:
        print(f"Error processing DNG: {e}")
        return None, None

def display_raw_data(raw_data, original_shape, filename="white_balanced_raw.npy"):
    """
    Saves the white balanced raw data as a single 2D NumPy array.

    Args:
        raw_data (numpy.ndarray): The white balance-corrected 2D raw pixel data.
        original_shape (tuple): The original shape of the raw image.
        filename (str): Base filename.
    """
    if raw_data is not None:
        # np.save(filename, raw_data)
        # print(f"White balanced raw data saved to {filename} with shape {original_shape}")
        out_im = [raw_data] * 3
        # out_im = ((np.stack(out_im, axis=2) / np.max(out_im)) * 255).astype(np.uint8)
        out_im = np.stack(out_im, axis=2)
        cv2.imwrite('wb.png', out_im)
        print(f'Wrote the np array as a .png image after replicating it three times')
        top_left     = raw_data[0::2, 0::2]
        top_right    = raw_data[0::2, 1::2]
        bottom_left  = raw_data[1::2, 0::2]
        bottom_right = raw_data[1::2, 1::2]
        
        print(np.sum(top_left), np.sum(top_right), np.sum(bottom_left), np.sum(bottom_right))
    
    else:
        print("No raw data to save.")

if __name__ == "__main__":
    dng_file_path = "real_camera_raw_images/white.dng"

    white_balanced_data, original_shape = apply_white_balance_only(dng_file_path)

    display_raw_data(white_balanced_data, original_shape)

    print("Processing complete.")
