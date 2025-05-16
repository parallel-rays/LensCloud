import rawpy
import numpy as np
import struct
import datetime

def write_dng(filename, image_data, bayer_pattern='RGGB', color_filter_array='RGGB', white_level=4095, black_level=0):
    """
    Writes a 2D numpy array to a DNG file.

    Args:
        filename (str): The name of the output DNG file.
        image_data (numpy.ndarray): The 2D numpy array containing the image data.
            The data type should be a 16-bit unsigned integer (uint16).
        bayer_pattern (str, optional): The Bayer pattern. Defaults to 'RGGB'.
            Corresponds to DNG CFAPattern2 (50717) if used as BYTE array.
            The original code uses this for CFAPattern (50710) effectively.
            Note: DNG requires CFAPattern (50710) to be BYTE type, and CFARepeatPatternDim (50709) to define dimensions.
            The original tag 50717 is CFAPattern2, used when CFALayout=7 or 8.
            For simplicity, we'll keep the original tag usage but this might need DNG spec alignment for full compatibility.
        color_filter_array (str, optional): The color filter array pattern. Defaults to 'RGGB'.
            Original code uses this for DNG tag 50718 (CFALayout), but CFALayout is a SHORT, not a string.
            This argument is not directly used in the corrected IFD construction below in a standard way,
            but retained for signature compatibility. CFAPattern derived from bayer_pattern is used.
        white_level (int, optional): The white level of the sensor. Defaults to 4095.
        black_level (int, optional): The black level of the sensor. Defaults to 0.

    Raises:
        TypeError: If image_data is not a numpy array or its dtype is not uint16.
        ValueError: If the bayer pattern is invalid based on original code's length check.
    """
    if not isinstance(image_data, np.ndarray):
        raise TypeError("image_data must be a numpy array.")
    if image_data.dtype != np.uint16:
        raise TypeError("image_data must have dtype uint16.")

    height, width = image_data.shape

    if len(bayer_pattern) != 4: # Based on original code's use of bayer_pattern.encode() for a 4-byte field
        raise ValueError("bayer_pattern must be a 4-character string (e.g., 'RGGB').")
    # color_filter_array validation removed as its usage for tag 50718 was non-standard.

    # DNG Header
    dng_header = b'\x49\x49\x2a\x00\x08\x00\x00\x00'  # Little endian, MagicNumber=42, IFDOffset=8

    # --- Corrected/Placeholder DNG Tag Definitions ---
    # Note: Some tag IDs and types in the original list were not standard DNG or were mixed up.
    # This version tries to make them writable but for full DNG compliance, review against the DNG spec.
    # For CFAPattern, DNG expects byte values (0=R, 1=G, 2=B, etc.)
    # Assuming a simple mapping for RGGB: R=0, G=1, B=2
    cfa_pattern_map = {'R': 0, 'G': 1, 'B': 2, 'C':3, 'M':4, 'Y':5, 'K':6} # Extend as needed
    try:
        cfa_byte_pattern = tuple(cfa_pattern_map[c] for c in bayer_pattern)
    except KeyError:
        raise ValueError(f"Invalid characters in bayer_pattern '{bayer_pattern}'. Only R,G,B,C,M,Y,K supported for this example.")


    ifd0_tags_definitions = [
        (254, 4, 1, 0),  # NewSubfileType (0 for full-resolution image)
        (256, 3, 1, width),        # ImageWidth
        (257, 3, 1, height),       # ImageLength
        (258, 3, 4, (16, 16, 16, 16)),  # BitsPerSample (e.g. 16 for uint16 raw data, 4 samples if RGGB or similar) - DNG requires 1 sample for CFA
                                        # For CFA, SamplesPerPixel=1, BitsPerSample refers to that one sample.
                                        # Let's assume 1 sample of 16 bits.
        (258, 3, 1, 16), # BitsPerSample (corrected for CFA: 1 sample, 16 bits)
        (259, 3, 1, 1),           # Compression (1=Uncompressed)
        (262, 3, 1, 32803),       # PhotometricInterpretation (32803 for Color Filter Array)
        (270, 2, len("Created by Gemini") + 1, "Created by Gemini"),  # ImageDescription
        (271, 2, len("Google") + 1, "Google"),  # Make
        (272, 2, len("Python DNG Writer") + 1, "Python DNG Writer"),  # Model
        (273, 4, 1, 0), # StripOffsets - Placeholder, will be updated
        (277, 3, 1, 1),           # SamplesPerPixel (1 for CFA)
        (278, 3, 1, height),       # RowsPerStrip
        (279, 4, 1, 0),  # StripByteCounts - Placeholder, will be updated
        (282, 5, 1, (72, 1)),  # XResolution (num, den)
        (283, 5, 1, (72, 1)),  # YResolution (num, den)
        (284, 3, 1, 1),           # PlanarConfiguration (1=chunky for CFA)
        (296, 3, 1, 2),           # ResolutionUnit (2=inches)
        (305, 2, len("Python DNG Writer v0.1") + 1, "Python DNG Writer v0.1"), # Software
        (306, 2, 20, datetime.datetime.now().strftime('%Y:%m:%d %H:%M:%S')), # DateTime
        (34665, 4, 1, 0),  # ExifTag (Offset to Exif IFD) - Placeholder
        # DNG Specific Tags
        (50706, 1, 4, (1, 4, 0, 0)), # DNGVersion (e.g., 1.4.0.0) - Type BYTE
        (50707, 1, 4, (1, 1, 0, 0)), # DNGBackwardVersion (e.g., 1.1.0.0 for compatibility) - Type BYTE
        (50708, 2, len("UniqueCameraModelStr") + 1, "UniqueCameraModelStr"), # UniqueCameraModel - ASCII String
        (50709, 3, 2, (2,2)), # CFARepeatPatternDim (e.g., 2x2 for RGGB) - SHORT
        (50710, 1, 4, cfa_byte_pattern), # CFAPattern - BYTE values. For RGGB: (R,G,G,B) e.g. (0,1,1,2)
        (50714, 3, 1, black_level), # BlackLevel - SHORT or LONG. Count SamplesPerPixel.
        (50717, 3, 1, white_level), # WhiteLevel - SHORT or LONG. Count SamplesPerPixel.
        # Colorimetry (example values, should be calibrated for actual sensor)
        (50721, 5, 3, (10000,10000, 0,10000, 0,10000)), # AsShotNeutral (XYZ, for R,G,B multipliers usually) - RATIONAL
        (50722, 5, 1, (255,1)), # AsShotWhiteXY (x,y) - RATIONAL
        (50727, 5, 9, (7152,10000, -2152,10000, -800,10000, # ColorMatrix1 (XYZ to Camera Native) - SRATIONAL
                           -1669,10000, 10000,10000, -200,10000,
                           -560,10000, -1750,10000, 10000,10000)),
        # Add other essential DNG tags like CalibrationIlluminant1, ForwardMatrix1 etc.
    ]
    num_ifd0_entries = len(ifd0_tags_definitions)

    exif_ifd_tags_definitions = [
        (33434, 5, 1, (1, 100)),  # ExposureTime (e.g., 1/100s)
        (33437, 5, 1, (28, 10)),  # FNumber (e.g., F2.8)
        (34855, 3, 1, 100),       # ISOSpeedRatings (Type SHORT)
        (36867, 2, 20, datetime.datetime.now().strftime('%Y:%m:%d %H:%M:%S')),  # DateTimeOriginal
        (37377, 10, 1, (10,1)), # ShutterSpeedValue APEX (e.g. for 1/100 s, log2(100) approx 6.64) - SRATIONAL
        (37378, 5, 1, (28,10)), # ApertureValue APEX (e.g. for F2.8, 2*log2(2.8) approx 2.97) - RATIONAL
    ]
    num_exif_ifd_entries = len(exif_ifd_tags_definitions)

    # Calculate sizes of IFD entry blocks
    ifd0_entry_block_size = 2 + num_ifd0_entries * 12 + 4  # NumEntries + Entries + NextIFDOffset
    exif_ifd_entry_block_size = 2 + num_exif_ifd_entries * 12 + 4

    # Define start offsets for major blocks
    # IFD0 itself starts at offset 8 (defined by dng_header)
    exif_ifd_start_offset = 8 + ifd0_entry_block_size
    image_data_start_offset = exif_ifd_start_offset + exif_ifd_entry_block_size # If Exif IFD exists
    if not exif_ifd_tags_definitions: # If no Exif IFD
        image_data_start_offset = 8 + ifd0_entry_block_size
        exif_ifd_start_offset = 0 # No Exif IFD

    # Auxiliary data (for values > 4 bytes or specific types like strings/rationals)
    # will be placed after the image data.
    aux_data_start_offset_base = image_data_start_offset + image_data.nbytes
    
    aux_data_buffer = bytearray()
    current_aux_offset = aux_data_start_offset_base

    processed_ifd0_tags = []
    processed_exif_tags = []

    def process_tags(tag_definitions, is_exif_ifd=False):
        nonlocal current_aux_offset # Allow modification of the outer scope variable
        processed_tags_list = []
        for tag_id, tag_type, tag_count, tag_value in tag_definitions:
            # Update dynamic offsets
            if not is_exif_ifd:
                if tag_id == 273: # StripOffsets
                    tag_value = image_data_start_offset
                elif tag_id == 279: # StripByteCounts
                    tag_value = image_data.nbytes
                elif tag_id == 34665: # ExifTag / GPSTag / InteropOffset
                    if exif_ifd_tags_definitions: # Only set if Exif IFD will be written
                        tag_value = exif_ifd_start_offset
                    else: # If no Exif IFD, this tag should not exist or be 0
                        tag_value = 0
                        if not exif_ifd_tags_definitions: continue # Skip adding Exif IFD pointer if no Exif IFD

            data_bytes_for_aux = None
            packed_direct_value_bytes = None # Bytes for direct packing (max 4 bytes)

            # Determine element size (bytes per count element)
            # TIFF types: 1:BYTE, 2:ASCII, 3:SHORT, 4:LONG, 5:RATIONAL
            # 6:SBYTE, 7:UNDEFINED, 8:SSHORT, 9:SLONG, 10:SRATIONAL, 11:FLOAT, 12:DOUBLE
            type_sizes = {1:1, 2:1, 3:2, 4:4, 5:8, 6:1, 7:1, 8:2, 9:4, 10:8, 11:4, 12:8}
            element_size = type_sizes.get(tag_type, 0)
            if element_size == 0:
                raise ValueError(f"Unknown TIFF tag type: {tag_type} for tag ID {tag_id}")
            
            total_data_size = element_size * tag_count

            if tag_type == 2: # ASCII String
                data_bytes_for_aux = tag_value.encode() # Null terminator included in count by definition
                                                        # Ensure tag_count is len(str)+1 in definition
                if len(data_bytes_for_aux) != tag_count:
                     # print(f"Warning: Tag {tag_id} ASCII count mismatch. Expected {tag_count}, got {len(data_bytes_for_aux)}")
                     # Adjust count or ensure string definition is correct. Forcing count to actual.
                     tag_count = len(data_bytes_for_aux) # Use actual length
                total_data_size = tag_count # Recalculate total_data_size for strings based on actual length

            elif tag_type == 5 or tag_type == 10: # RATIONAL or SRATIONAL
                data_bytes_for_aux = b''
                pack_char = '<Ll' if tag_type == 10 else '<LL' # Signed or Unsigned Long
                if tag_count == 1:
                    data_bytes_for_aux = struct.pack(pack_char, tag_value[0], tag_value[1])
                else:
                    for i in range(tag_count): # Assumes tag_value is list of tuples [(n1,d1), (n2,d2)...]
                                               # Or flat list [n1,d1,n2,d2...]
                        if isinstance(tag_value[0], tuple): # List of tuples
                             data_bytes_for_aux += struct.pack(pack_char, tag_value[i][0], tag_value[i][1])
                        else: # Flat list
                             data_bytes_for_aux += struct.pack(pack_char, tag_value[i*2], tag_value[i*2+1])
            
            elif total_data_size <= 4: # Data fits directly
                temp_bytes = b''
                if isinstance(tag_value, (list, tuple)): # e.g. SHORT, count=2, value=(s1,s2)
                    if tag_type == 1 or tag_type == 6 or tag_type == 7: # BYTE, SBYTE, UNDEFINED
                        for v in tag_value: temp_bytes += struct.pack('<b' if tag_type==6 else '<B', v)
                    elif tag_type == 3 or tag_type == 8: # SHORT, SSHORT
                        for v in tag_value: temp_bytes += struct.pack('<h' if tag_type==8 else '<H', v)
                    elif tag_type == 4 or tag_type == 9: # LONG, SLONG (only if count=1 for total_size<=4)
                         if tag_count == 1: temp_bytes = struct.pack('<l' if tag_type==9 else '<L', tag_value[0])
                    elif tag_type == 11: # FLOAT (count=1)
                         if tag_count == 1: temp_bytes = struct.pack('<f', tag_value[0])
                    else:
                        raise ValueError(f"Unhandled tuple value for direct packing: tag {tag_id}, type {tag_type}")
                else: # Single integer/float value
                    if tag_type == 1 or tag_type == 6 or tag_type == 7: temp_bytes = struct.pack('<b' if tag_type==6 else '<B', tag_value)
                    elif tag_type == 3 or tag_type == 8: temp_bytes = struct.pack('<h' if tag_type==8 else '<H', tag_value)
                    elif tag_type == 4 or tag_type == 9: temp_bytes = struct.pack('<l' if tag_type==9 else '<L', tag_value)
                    elif tag_type == 11: temp_bytes = struct.pack('<f', tag_value) # Single float
                    else: # Default to packing as LONG if type unknown but fits (original behavior's risk)
                        temp_bytes = struct.pack('<L', tag_value) # This is risky if not actually a long
                
                packed_direct_value_bytes = temp_bytes.ljust(4, b'\x00') # Pad to 4 bytes

            else: # Data is > 4 bytes or forced out-of-line (already handled for string/rational)
                if not isinstance(tag_value, (list, tuple)):
                    raise ValueError(f"Tag {tag_id}: Data size > 4 but value is not list/tuple.")
                
                data_bytes_for_aux = b''
                pack_formats = {1:'<B', 3:'<H', 4:'<L>', 6:'<b', 8:'<h', 9:'<l', 11:'<f', 12:'<d', 7:'<B'} # Type 7 as BYTE
                pack_fmt = pack_formats.get(tag_type)
                if not pack_fmt: raise ValueError(f"Unsupported tag type {tag_type} for array data.")
                for v_item in tag_value:
                    data_bytes_for_aux += struct.pack(pack_fmt, v_item)

            # Finalize IFD entry value (offset or direct packed value)
            if data_bytes_for_aux is not None:
                value_for_ifd_entry = current_aux_offset
                aux_data_buffer.extend(data_bytes_for_aux)
                if len(data_bytes_for_aux) % 2 != 0: # Pad aux data to word boundary
                    aux_data_buffer.extend(b'\x00')
                current_aux_offset += (len(data_bytes_for_aux) + (len(data_bytes_for_aux) % 2))
            elif packed_direct_value_bytes is not None:
                value_for_ifd_entry = struct.unpack('<L', packed_direct_value_bytes)[0]
            else: # Should not happen if all cases covered
                raise RuntimeError(f"Tag {tag_id} was not processed into direct value or aux data.")

            processed_tags_list.append((tag_id, tag_type, tag_count, value_for_ifd_entry))
        return processed_tags_list

    processed_ifd0_tags = process_tags(ifd0_tags_definitions, is_exif_ifd=False)
    if exif_ifd_tags_definitions:
        processed_exif_tags = process_tags(exif_ifd_tags_definitions, is_exif_ifd=True)
    
    # Update entry counts based on actual processed tags (e.g. if Exif IFD pointer was skipped)
    num_ifd0_entries = len(processed_ifd0_tags)
    num_exif_ifd_entries = len(processed_exif_tags)


    with open(filename, 'wb') as f:
        f.write(dng_header)

        # Write IFD0
        f.write(struct.pack('<H', num_ifd0_entries))
        for tag_id, tag_type, tag_count, value_or_offset in processed_ifd0_tags:
            f.write(struct.pack('<HHL', tag_id, tag_type, tag_count))
            f.write(struct.pack('<L', value_or_offset)) # This is now always an integer
        
        next_ifd0_offset = exif_ifd_start_offset if exif_ifd_tags_definitions and num_exif_ifd_entries > 0 else 0
        f.write(struct.pack('<L', next_ifd0_offset)) # Offset to next IFD (Exif IFD or 0)

        # Write Exif IFD (if it exists)
        if exif_ifd_tags_definitions and num_exif_ifd_entries > 0:
            f.write(struct.pack('<H', num_exif_ifd_entries))
            for tag_id, tag_type, tag_count, value_or_offset in processed_exif_tags:
                f.write(struct.pack('<HHL', tag_id, tag_type, tag_count))
                f.write(struct.pack('<L', value_or_offset))
            f.write(struct.pack('<L', 0))  # Next IFD offset for Exif (0)

        # Write the image data
        f.write(image_data.tobytes())

        # Write all accumulated auxiliary data
        f.write(aux_data_buffer)

if __name__ == '__main__':
    # Example usage:
    width_main = 640
    height_main = 480
    # Create a simple test pattern (checkerboard)
    image_data_main = np.zeros((height_main, width_main), dtype=np.uint16)
    for y_main in range(height_main):
        for x_main in range(width_main):
            if (x_main // 32 + y_main // 32) % 2 == 0:
                image_data_main[y_main, x_main] = 4000
            else:
                image_data_main[y_main, x_main] = 100

    # Write the data to a DNG file
    try:
        write_dng('test_pattern.dng', image_data_main, bayer_pattern='RGGB', white_level=4095, black_level=64)
        print("DNG file 'test_pattern.dng' created successfully.")
    except Exception as e:
        print(f"Error creating DNG file: {e}")
        import traceback
        traceback.print_exc()