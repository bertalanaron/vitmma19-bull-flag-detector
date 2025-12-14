# Data preprocessing script
# This script handles data loading, cleaning, and transformation.
from utils import setup_logger
import pandas as pd
import json
import os
import re
import shutil
import zipfile
import tempfile
from datetime import datetime
import requests

logger = setup_logger()

N_BARS_LOOKBACK = 50  # How many candles to look back for pole search

WINDOW = 64  # Window size for training samples
STRIDE = 8  # Stride between windows
NEG_POS_RATIO = 2  # Ratio of negative to positive samples
MIN_OVERLAP = 0.6  # Minimum overlap to assign a label to a window
TEST_SPLIT = 0.15  # Fraction of data to hold out for test set

MIN_LABEL_LENGTH = 16
MAX_LABEL_LENGTH = WINDOW  # Maximum label length in bars

raw_data_dir = "/data/raw"
merged_data_dir = "/data/merged"
processed_dir = "/data/processed"

def download_and_extract_zip(zip_url):
    logger.info(f"Downloading data from {zip_url}...")
    
    # Create temporary file for download
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
        temp_zip_path = tmp_file.name
    
    try:
        response = requests.get(zip_url)

        # Check if the request was successful
        if response.status_code == 200:
            # Save the content as a zip file
            with open(temp_zip_path, 'wb') as zip_file:
                zip_file.write(response.content)
            logger.info(f"Downloaded to {temp_zip_path}")
            
            # Create raw data directory if it doesn't exist
            os.makedirs(raw_data_dir, exist_ok=True)
            
            # Extract the zip file
            logger.info(f"Extracting to {raw_data_dir}...")
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                zip_ref.extractall(raw_data_dir)
            
            logger.info("Extraction complete")
        else:
            print(f"Failed to download file. Status code: {response.status_code}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)
            logger.info("Cleaned up temporary files")

def convert_to_datetime_format(input):
    input_string = str(input)
    try:
        # If the input is a valid timestamp in milliseconds (10+ digits), convert it
        if len(input_string) > 10:  # it's in milliseconds
            timestamp = int(input_string) / 1000
        else:  # It's in seconds
            timestamp = int(input_string)

        # Convert the timestamp to a datetime object
        dt_object = datetime.utcfromtimestamp(timestamp)

        # Return the formatted string
        return dt_object.strftime('%Y-%m-%d %H:%M')
    
    except ValueError:
        # If input is in "YYYY-MM-DD HH:mm" format already, return it unchanged
        try:
            datetime.strptime(input_string, "%Y-%m-%d %H:%M")
            return input_string
        except ValueError:
            raise ValueError("Invalid input format")

def normalize_csv_timestamps(csv_path):
    """
    Normalize the first column of a CSV file to Unix timestamps with second precision.
    
    Args:
        csv_path: Path to the CSV file to normalize
    
    Returns:
        DataFrame with normalized timestamps and mapping from original to normalized
    """
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Get the first column name
    time_col = df.columns[0]
    
    # Try to parse the time column
    try:
        # Try parsing as datetime first (handles formats like "2025-07-27 21:02")
        df[time_col] = df[time_col].apply(convert_to_datetime_format)
    except:
        logger.warning(f"Could not parse time column in {csv_path}")
        return df, {}
        
    # Save the normalized CSV
    df.to_csv(csv_path, index=False)
    
    # Create mapping: row index -> timestamp
    timestamp_map = {idx: ts for idx, ts in enumerate(df[time_col])}
    
    return df, timestamp_map

def handle_labels_in_single_file(labels_in_file_json: json, labels_json: json, current_input_dir, next_file_id):
    # Ensure output structure
    if 'files' not in labels_json:
        labels_json['files'] = []
    if 'labels' not in labels_json:
        labels_json['labels'] = []

    # Skip file if it has 0 annotations
    annotations = labels_in_file_json.get('annotations', []) if isinstance(labels_in_file_json, dict) else []
    if not annotations or len(annotations) == 0:
        return next_file_id
    label_count = sum(len(a.get('result', [])) for a in annotations)
    if label_count < 1: 
        return next_file_id

    # determine file path
    file_path = None
    if isinstance(labels_in_file_json, dict):
        file_path = labels_in_file_json.get('file_upload') or labels_in_file_json.get('data', {}).get('csv')
    if not file_path:
        raise ValueError("Cannot determine file path for input file JSON")

    # Parse filename to match pattern: SYMBOL_Xmin_INDEX.csv or SYMBOL_XH_INDEX.csv
    # May have optional prefix like: 351f4d2a-EURUSD_1H_005.csv
    filename = os.path.basename(file_path)
    
    # Pattern: (optional_prefix-)SYMBOL_(number)(unit)_(index).csv
    # Examples: EURUSD_1min_001.csv, XAU_1H_001.csv, EURUSD_15min_003.csv, 351f4d2a-EURUSD_1H_005.csv
    pattern = r'^(?:[a-zA-Z0-9]+-)?([A-Z]+)_(\d+)(min|H)_(\d+)\.csv$'
    match = re.match(pattern, filename)
    
    if not match:
        logger.warning(f"File path doesn't match expected pattern, discarding: {file_path}")
        return next_file_id
    
    symbol = match.group(1)
    timestep_count = int(match.group(2))
    unit_raw = match.group(3)
    original_index = int(match.group(4))
    
    # Normalize unit
    if unit_raw == 'min':
        timestep_unit = 'min'
    elif unit_raw == 'H':
        timestep_unit = 'H'
    else:
        timestep_unit = unit_raw

    normalized_filename = f"{symbol}_{timestep_count}{timestep_unit}_{original_index:03d}.csv"
    
    # Check if file exists in current_input_dir
    full_file_path = os.path.join(current_input_dir, normalized_filename)
    if not os.path.isfile(full_file_path):
        logger.warning(f"File not found in {current_input_dir}, discarding: {full_file_path}")
        return next_file_id
    
    # Find lowest available 3-digit index for normalized path
    normalized_index = 1
    while True:
        normalized_filename = f"{symbol}_{timestep_count}{timestep_unit}_{normalized_index:03d}.csv"
        normalized_path = os.path.join(merged_data_dir, normalized_filename)
        
        # Check if this normalized path already exists in our output structure
        existing = next((f for f in labels_json['files'] if f.get('normalized_file') == normalized_filename), None)
        
        # Also check if file physically exists in merged_data_dir
        if existing is None and not os.path.exists(normalized_path):
            break
        
        normalized_index += 1
        if normalized_index > 999:
            raise ValueError(f"Cannot find available index for {symbol}_{timestep_count}{timestep_unit}")
        
    # check if original file path already present
    existing = next((f for f in labels_json['files'] if f.get('normalized_file') == normalized_filename), None)
    if existing is None:
        # count labels available in this file (sum of result lengths)
        label_count = sum(len(a.get('result', [])) for a in annotations)

        # Copy original file from current_input_dir to merged_data_dir with normalized filename
        os.makedirs(merged_data_dir, exist_ok=True)
        shutil.copy2(full_file_path, normalized_path)
        logger.info(f"Copied {full_file_path} -> {normalized_path}")
        
        # Normalize timestamps in the CSV file
        df, timestamp_map = normalize_csv_timestamps(normalized_path)

        file_entry = {
            'id': next_file_id,
            'file': full_file_path,
            'normalized_file': normalized_filename,
            'symbol': symbol,
            'timestep_count': timestep_count,
            'timestep_unit': timestep_unit,
            'label_count': label_count
        }
        labels_json['files'].append(file_entry)
        file_id = next_file_id
        next_file_id += 1
    else:
        file_id = existing['id']
        # Load timestamp map for existing file
        normalized_path = os.path.join(merged_data_dir, normalized_filename)
        df = pd.read_csv(normalized_path)
        time_col = df.columns[0]
        timestamp_map = {idx: ts for idx, ts in enumerate(df[time_col])}

    # helper to normalize type/subtype
    def split_label(lbl: str):
        if not lbl:
            return ('unknown', '')
        parts = lbl.strip().split()
        t = parts[0].lower()
        subtype = ' '.join(parts[1:]).lower() if len(parts) > 1 else ''
        # normalize common tokens
        if t.startswith('bull'):
            t = 'bullish'
        elif t.startswith('bear'):
            t = 'bearish'
        return (t, subtype)

    # extract individual timeserieslabels
    for ann in annotations:
        for res in ann.get('result', []):
            if res.get('type') != 'timeserieslabels':
                continue
            val = res.get('value', {})
            start = val.get('start')
            end = val.get('end')
            instant = val.get('instant', False)
            ts_labels = val.get('timeserieslabels', []) or []

            # Normalize start and end using the timestamp map
            # start and end are row indices from Label Studio
            normalized_start = timestamp_map.get(start, start) if start is not None else None
            normalized_end = timestamp_map.get(end, end) if end is not None else None

            # create one output label per timeseries label in the result
            for ts_label in ts_labels:
                t, subtype = split_label(ts_label)
                label_entry = {
                    'file': file_id,
                    'start': convert_to_datetime_format(start),
                    'end': convert_to_datetime_format(end),
                    'instant': instant,
                    'type': t,
                    'subtype': subtype
                }
                labels_json['labels'].append(label_entry)

    return next_file_id

def convert_labels(out_labels_json: json, labels_json: json, current_input_dir: str, next_file_id = 0):
    """
    Convert labels from Label Studio export format to the custom output format.
    
    Args:
        out_labels_json: Dictionary to populate with converted labels (modified in-place)
        labels_json: List of Label Studio task objects from the export JSON
        next_file_id: Starting file ID (default 0)
    
    Returns:
        The next available file_id after processing
    """
    # Ensure labels_json is a list
    if not isinstance(labels_json, list):
        raise ValueError("labels_json must be a list of Label Studio task objects")
    
    # Process each task in the input
    for task in labels_json:
        if not isinstance(task, dict):
            continue
        next_file_id = handle_labels_in_single_file(task, out_labels_json, current_input_dir, next_file_id)
    
    return next_file_id

def merge_and_convert_labels(input_dir: str):
    """
    Merge and convert all labels.json files from subdirectories of input_dir.
    
    Args:
        input_dir: Root directory containing subdirectories with labels.json files
    
    Returns:
        Dictionary with merged labels in the custom format
    """
    import os
    
    next_file_id = 0
    out_labels_json = {}
    
    # Check if input_dir exists
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Iterate through subdirectories
    for subdir_name in os.listdir(input_dir):
        subdir_path = os.path.join(input_dir, subdir_name)
        
        # Skip if not a directory
        if not os.path.isdir(subdir_path):
            continue
        
        # Look for labels.json in this subdirectory
        labels_file = os.path.join(subdir_path, 'labels.json')
        if not os.path.isfile(labels_file):
            logger.warning(f"No labels.json found in {subdir_path}")
            continue
        
        # Load and convert labels from this file
        try:
            with open(labels_file, 'r', encoding='utf-8') as f:
                input_labels = json.load(f)
            
            next_file_id = convert_labels(out_labels_json, input_labels, subdir_path, next_file_id)
            logger.info(f"Processed {labels_file}: {len(out_labels_json.get('labels', []))} total labels")
        except Exception as e:
            logger.error(f"Error processing {labels_file}: {e}")
    
    return out_labels_json

def standardize_pole_starts(labels_json: dict):
    """
    Recalculate pole_start for all flag pattern labels using slope maximization.
    Adapted to work with the custom label structure.
    
    Args:
        labels_json: Dictionary with 'labels' and 'files' keys
    
    Returns:
        List of standardized labels with 'pole_start' field added
    """
    logger.info("Standardizing pole starts for flag patterns...")
    labels = labels_json.get("labels", [])
    files = labels_json.get("files", [])
    
    standardized_labels = []
    
    for label in labels:
        # Only process flag patterns
        label_type = label.get('type', '')
        label_subtype = label.get('subtype', '')
        
        # Determine if this is a bull or bear flag
        pattern_type = None
        if label_type == 'bullish':
            pattern_type = "BULL_FLAG"
        elif label_type == 'bearish':
            pattern_type = "BEAR_FLAG"
        else:
            # Not a flag pattern, skip
            continue
        
        try:
            # Load the CSV file for this label
            file_id = label['file']
            file_entry = next((f for f in files if f['id'] == file_id), None)
            
            if not file_entry:
                continue
            
            csv_path = os.path.join(merged_data_dir, file_entry['normalized_file'])
            if not os.path.exists(csv_path):
                continue
            
            # Load OHLCV data with timestamp as index
            df = pd.read_csv(csv_path)
            time_col = df.columns[0]
            df[time_col] = pd.to_datetime(df[time_col])
            df.set_index(time_col, inplace=True)
            
            # Get flag start timestamp
            flag_start_ts = pd.to_datetime(label['start'])
            
            # Find nearest index position for flag start
            flag_start_idx_pos = df.index.get_indexer([flag_start_ts], method='nearest')[0]
            flag_start_bar = df.iloc[flag_start_idx_pos]
            
            # Variables to track best candidate
            best_pole_start_ts = None
            max_slope = -float('inf')
            
            # Search window (look back)
            for i in range(1, N_BARS_LOOKBACK + 1):
                candidate_idx_pos = flag_start_idx_pos - i
                
                # Check bounds
                if candidate_idx_pos < 0:
                    break
                
                candidate_bar = df.iloc[candidate_idx_pos]
                time_bars_elapsed = i
                
                # Calculate slope based on pattern type
                price_change = 0.0
                
                if pattern_type == "BULL_FLAG":
                    # Pole: Rising. Starts at flag top (high), pole begins at low
                    anchor_price = flag_start_bar['high']
                    candidate_price = candidate_bar['low']
                    price_change = anchor_price - candidate_price
                    
                elif pattern_type == "BEAR_FLAG":
                    # Pole: Falling. Starts at flag bottom (low), pole begins at high
                    anchor_price = flag_start_bar['low']
                    candidate_price = candidate_bar['high']
                    price_change = candidate_price - anchor_price
                
                # Update best candidate if slope is better
                if price_change > 0:
                    current_slope = price_change / time_bars_elapsed
                    
                    if current_slope > max_slope:
                        max_slope = current_slope
                        best_pole_start_ts = candidate_bar.name  # Get timestamp from index
            
            # Add standardized label if valid pole was found
            if best_pole_start_ts is not None:
                new_label = {
                    **label,  # Copy all original fields
                    'pole_start': best_pole_start_ts.strftime('%Y-%m-%d %H:%M'),
                    'calculated_slope': max_slope,
                    'pattern_type': pattern_type
                }
                standardized_labels.append(new_label)
            
        except Exception as e:
            file_id = label['file']
            file_entry = next((f for f in files if f['id'] == file_id), None)
            
            if not file_entry:
                continue
            
            csv_path = os.path.join(merged_data_dir, file_entry['normalized_file'])
            print(f"Warning: Could not process label in {csv_path}: {e}")
            continue
    
    logger.info(f"Processed {len(standardized_labels)} flag pattern labels")
    return standardized_labels

def filter_labels(labels_json: dict, standardized_labels: list):
    """
    Filter labels by length (number of bars from pole_start to end).
    
    Args:
        labels_json: Original labels dictionary with 'files' list
        standardized_labels: List of standardized labels with 'pole_start' field
    
    Returns:
        Dictionary with filtered labels and updated file list
    """
    logger.info(f"Filtering labels (min: {MIN_LABEL_LENGTH}, max: {MAX_LABEL_LENGTH} bars)...")
    
    files = labels_json.get('files', [])
    filtered_labels = []
    files_with_labels = set()
    
    for label in standardized_labels:
        try:
            # Get file info
            file_id = label['file']
            file_entry = next((f for f in files if f['id'] == file_id), None)
            
            if not file_entry:
                continue
            
            # Load CSV to calculate length
            csv_path = os.path.join(merged_data_dir, file_entry['normalized_file'])
            if not os.path.exists(csv_path):
                continue
            
            df = pd.read_csv(csv_path)
            time_col = df.columns[0]
            
            # Get pole_start and end timestamps
            pole_start_ts = label.get('pole_start')
            end_ts = label.get('end')
            
            if not pole_start_ts or not end_ts:
                continue
            
            # Find row indices
            start_mask = df[time_col] == pole_start_ts
            end_mask = df[time_col] == end_ts
            
            start_indices = df[start_mask].index
            end_indices = df[end_mask].index
            
            if len(start_indices) == 0 or len(end_indices) == 0:
                continue
            
            start_idx = start_indices[0]
            end_idx = end_indices[0]
            
            # Calculate length in bars
            label_length = int(end_idx - start_idx + 1)
            
            # Filter by length
            if MIN_LABEL_LENGTH <= label_length <= MAX_LABEL_LENGTH:
                label['length'] = label_length
                filtered_labels.append(label)
                files_with_labels.add(file_id)
            
        except Exception as e:
            logger.warning(f"Error filtering label: {e}")
            continue
    
    # Filter files list to only include files with valid labels
    filtered_files = [f for f in files if f['id'] in files_with_labels]
    
    # Update label counts
    for file_entry in filtered_files:
        file_id = file_entry['id']
        label_count = sum(1 for label in filtered_labels if label['file'] == file_id)
        file_entry['label_count'] = label_count
    
    logger.info(f"Filtered {len(filtered_labels)}/{len(standardized_labels)} labels")
    logger.info(f"Kept {len(filtered_files)}/{len(files)} files")
    
    return {
        'files': filtered_files,
        'labels': filtered_labels
    }

def generate_dataframes(labels_json: dict, out_dir: str):
    """
    Generate training data windows from labeled CSV files.
    
    Args:
        labels_json: Dictionary with 'files' and 'labels' lists
        out_dir: Output directory for processed data
    """
    import numpy as np
    import random
    
    logger.info(f"Generating training windows (window={WINDOW}, stride={STRIDE})...")
    
    files = labels_json.get('files', [])
    all_labels = labels_json.get('labels', [])
    
    # Store samples by file to enable proper train/test splitting
    samples_by_file = {}  # file_id -> list of (features, class_id)
    
    # Pattern type+subtype to class ID mapping
    # Class 0: no_flag
    # Class 1-3: bullish (normal, pennant, wedge)
    # Class 4-6: bearish (normal, pennant, wedge)
    pattern_to_class = {
        ('bullish', 'normal'): 1,
        ('bullish', 'pennant'): 2,
        ('bullish', 'wedge'): 3,
        ('bearish', 'normal'): 4,
        ('bearish', 'pennant'): 5,
        ('bearish', 'wedge'): 6
    }
    
    def extract_features(ohlc_window):
        """
        Extract features from OHLC window.
        Input: (T, 4) - O, H, L, C
        Output: (T, 5) - returns, body, upper_wick, lower_wick, range
        """
        O, H, L, C = ohlc_window.T
        
        returns = np.diff(np.log(C + 1e-8), prepend=0)
        body = C - O
        upper_wick = H - np.maximum(O, C)
        lower_wick = np.minimum(O, C) - L
        range_ = H - L
        
        feats = np.stack([returns, body, upper_wick, lower_wick, range_], axis=1)
        
        # Per-window normalization
        feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-6)
        return feats.astype(np.float32)
    
    def window_label(start_idx, end_idx, file_labels):
        """
        Assign label to window based on overlap with labeled regions.
        Returns class ID (0 for no-flag, 1 for bull, 2 for bear)
        """
        window_len = end_idx - start_idx
        best_overlap = 0
        best_class = 0  # no-flag
        
        for label in file_labels:
            l_start = label['start_idx']
            l_end = label['end_idx']
            class_id = label['class_id']
            
            # Calculate overlap
            overlap = max(0, min(end_idx, l_end) - max(start_idx, l_start))
            frac = overlap / window_len
            
            if frac > best_overlap:
                best_overlap = frac
                best_class = class_id
        
        if best_overlap >= MIN_OVERLAP:
            return best_class
        return 0
    
    # Process each file
    for file_entry in files:
        file_id = file_entry['id']
        csv_path = os.path.join(merged_data_dir, file_entry['normalized_file'])
        
        if not os.path.exists(csv_path):
            logger.warning(f"CSV file not found: {csv_path}")
            continue
        
        logger.info(f"Processing {file_entry['normalized_file']}...")
        
        # Load OHLC data
        df = pd.read_csv(csv_path)
        time_col = df.columns[0]
        ohlc = df[['open', 'high', 'low', 'close']].values.astype(np.float32)
        
        # Get labels for this file with indices
        file_labels = []
        for label in all_labels:
            if label['file'] != file_id:
                continue
            
            pole_start_ts = label.get('pole_start')
            end_ts = label.get('end')
            label_type = label.get('type')  # bullish or bearish
            label_subtype = label.get('subtype')  # normal, pennant, or wedge
            
            if not pole_start_ts or not end_ts or not label_type or not label_subtype:
                continue
            
            # Find indices
            start_mask = df[time_col] == pole_start_ts
            end_mask = df[time_col] == end_ts
            
            start_indices = df[start_mask].index
            end_indices = df[end_mask].index
            
            if len(start_indices) > 0 and len(end_indices) > 0:
                file_labels.append({
                    'start_idx': start_indices[0],
                    'end_idx': end_indices[0],
                    'class_id': pattern_to_class.get((label_type, label_subtype), 0)
                })
        
        # Generate windows
        N = len(ohlc)
        for start in range(0, N - WINDOW, STRIDE):
            end = start + WINDOW
            
            # Get label for this window
            cls = window_label(start, end, file_labels)
            
            # Extract features
            window_ohlc = ohlc[start:end]
            features = extract_features(window_ohlc)
            
            # Store sample by file
            sample = (features, cls)
            if file_id not in samples_by_file:
                samples_by_file[file_id] = []
            samples_by_file[file_id].append(sample)
    
    # Count samples
    total_samples = sum(len(samples) for samples in samples_by_file.values())
    logger.info(f"Generated {total_samples} samples from {len(samples_by_file)} files")
    
    # Split files into train_val and test (file-level split to prevent data leakage)
    file_ids = list(samples_by_file.keys())
    random.shuffle(file_ids)
    
    n_test_files = max(1, int(len(file_ids) * TEST_SPLIT))
    test_file_ids = set(file_ids[:n_test_files])
    train_val_file_ids = set(file_ids[n_test_files:])
    
    logger.info(f"Split files: {len(train_val_file_ids)} train_val files, {len(test_file_ids)} test files")
    
    # Separate samples by split
    train_val_samples = []
    test_samples = []
    
    for file_id, samples in samples_by_file.items():
        if file_id in test_file_ids:
            test_samples.extend(samples)
        else:
            train_val_samples.extend(samples)
    
    logger.info(f"Samples per split: train_val={len(train_val_samples)}, test={len(test_samples)}")
    
    # Balance train_val dataset by subsampling negatives
    train_val_positives = [s for s in train_val_samples if s[1] != 0]
    train_val_negatives = [s for s in train_val_samples if s[1] == 0]
    
    k = min(len(train_val_negatives), NEG_POS_RATIO * len(train_val_positives))
    if len(train_val_negatives) > k:
        train_val_negatives = random.sample(train_val_negatives, k)
        logger.info(f"Subsampled train_val negatives to {k} samples")
    
    train_val_samples = train_val_positives + train_val_negatives
    random.shuffle(train_val_samples)
    
    # Note: We don't balance test set - keep natural distribution for evaluation
    random.shuffle(test_samples)
    
    # Convert to arrays
    X_train_val = np.stack([s[0] for s in train_val_samples])
    y_train_val = np.array([s[1] for s in train_val_samples], dtype=np.int64)
    X_test = np.stack([s[0] for s in test_samples])
    y_test = np.array([s[1] for s in test_samples], dtype=np.int64)
    
    logger.info(f"Final dataset: train_val={X_train_val.shape}, test={X_test.shape}")
    logger.info(f"Train_val class distribution: {np.bincount(y_train_val)}")
    logger.info(f"Test class distribution: {np.bincount(y_test)}")
    
    # Save to disk
    os.makedirs(out_dir, exist_ok=True)
    
    # Save train_validation set
    X_train_val_path = os.path.join(out_dir, 'X_train_val.npy')
    y_train_val_path = os.path.join(out_dir, 'y_train_val.npy')
    np.save(X_train_val_path, X_train_val)
    np.save(y_train_val_path, y_train_val)
    logger.info(f"Saved train_validation data to {X_train_val_path} and {y_train_val_path}")
    
    # Save test set
    X_test_path = os.path.join(out_dir, 'X_test.npy')
    y_test_path = os.path.join(out_dir, 'y_test.npy')
    np.save(X_test_path, X_test)
    np.save(y_test_path, y_test)
    logger.info(f"Saved test data to {X_test_path} and {y_test_path}")
    
    # Also save metadata
    y_combined = np.concatenate([y_train_val, y_test])
    metadata = {
        'window_size': WINDOW,
        'stride': STRIDE,
        'num_features': X_train_val.shape[2],
        'num_classes': len(np.unique(y_combined)),
        'total_samples': len(X_train_val) + len(X_test),
        'train_val_samples': len(X_train_val),
        'test_samples': len(X_test),
        'test_split': TEST_SPLIT,
        'class_distribution': np.bincount(y_combined).tolist(),
        'train_val_class_distribution': np.bincount(y_train_val).tolist(),
        'test_class_distribution': np.bincount(y_test).tolist(),
        'train_val_files': len(train_val_file_ids),
        'test_files': len(test_file_ids),
        'class_names': [
            'no_flag',
            'bullish_normal',
            'bullish_pennant',
            'bullish_wedge',
            'bearish_normal',
            'bearish_pennant',
            'bearish_wedge'
        ]
    }
    
    metadata_path = os.path.join(out_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved metadata to {metadata_path}")

def preprocess():
    logger.info("Preprocessing data...")

    # Download and extract zip archive
    zip_url = "https://bmeedu-my.sharepoint.com/:u:/g/personal/gyires-toth_balint_vik_bme_hu/IQAtwZfO4KZURYZ7QTq838KlATzWm7x-GtnoYpe6WfSMnCg?e=WwCXrH&download=1"
    download_and_extract_zip(zip_url)

    # Merge labels, copy and unify csv files
    labels = merge_and_convert_labels(raw_data_dir)
    # Standardize pole starts
    standardized_labels = standardize_pole_starts(labels)
    # Filter based on label length
    filtered_data = filter_labels(labels, standardized_labels)

    # Write labels.json to merged dir
    with open(merged_data_dir + "/labels.json", "w", encoding="utf-8") as f:
        json.dump(filtered_data, f)
    
    # Generate training data
    generate_dataframes(filtered_data, processed_dir)

if __name__ == "__main__":
    preprocess()
