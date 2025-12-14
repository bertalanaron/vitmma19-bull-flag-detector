# Inference script
# This script runs the model on new, unseen data to detect flag patterns.
import config
from utils import setup_logger
import torch
import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import torch.nn as nn
from matplotlib.patches import Rectangle
from model import FlagCNN

logger = setup_logger()

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

def find_unannotated_file(max_bars=None):
    """
    Find a CSV file in data/merged that has no labels.
    
    Args:
        max_bars: Maximum number of bars/candles in the file (optional)
    """
    merged_dir = config.DATA_DIR + "/merged"
    labels_path = os.path.join(merged_dir, "labels.json")
    
    if not os.path.exists(labels_path):
        logger.error(f"Labels file not found: {labels_path}")
        return None
    
    # Load labels to find which files have annotations
    with open(labels_path, 'r') as f:
        labels_data = json.load(f)
    
    annotated_files = {f['normalized_file'] for f in labels_data.get('files', [])}
    
    # Find all CSV files
    all_files = [f for f in os.listdir(merged_dir) if f.endswith('.csv')]
    
    # Find unannotated files
    unannotated = [f for f in all_files if f not in annotated_files]
    
    if not unannotated:
        logger.warning("No unannotated files found. Using a file with annotations for demonstration.")
        unannotated = all_files
    
    if not unannotated:
        logger.error("No CSV files found")
        return None
    
    # Filter by size if max_bars is specified
    if max_bars is not None:
        logger.info(f"Looking for files with less than {max_bars} bars...")
        suitable_files = []
        for filename in unannotated:
            filepath = os.path.join(merged_dir, filename)
            try:
                df = pd.read_csv(filepath)
                num_bars = len(df)
                if num_bars < max_bars:
                    suitable_files.append((filename, num_bars))
                    logger.info(f"  {filename}: {num_bars} bars")
            except Exception as e:
                logger.warning(f"  Could not read {filename}: {e}")
                continue
        
        if not suitable_files:
            logger.warning(f"No files found with less than {max_bars} bars. Using smallest available file.")
            # Find smallest file
            file_sizes = []
            for filename in unannotated:
                filepath = os.path.join(merged_dir, filename)
                try:
                    df = pd.read_csv(filepath)
                    file_sizes.append((filename, len(df)))
                except:
                    continue
            if file_sizes:
                suitable_files = sorted(file_sizes, key=lambda x: x[1])[:1]
        
        if suitable_files:
            # Sort by size and pick the largest within the limit
            suitable_files.sort(key=lambda x: x[1], reverse=True)
            selected_filename, num_bars = suitable_files[0]
            selected_file = os.path.join(merged_dir, selected_filename)
            logger.info(f"Selected file: {selected_filename} ({num_bars} bars)")
            return selected_file
    
    # Return the first unannotated file
    selected_file = os.path.join(merged_dir, unannotated[0])
    logger.info(f"Found unannotated file: {unannotated[0]}")
    return selected_file

def plot_predictions(df, predictions, probabilities, class_names, save_path):
    """
    Plot candlestick chart with predicted flag patterns highlighted.
    
    Args:
        df: DataFrame with OHLC data
        predictions: Array of (start_idx, end_idx, class_id, confidence) - top patterns to display
        probabilities: Array of probabilities for all windows
        class_names: List of class names
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1])
    
    # Plot candlesticks
    for i in range(len(df)):
        o, h, l, c = df.iloc[i][['open', 'high', 'low', 'close']]
        color = 'green' if c >= o else 'red'
        ax1.plot([i, i], [l, h], color='black', linewidth=0.5)
        ax1.plot([i, i], [o, c], color=color, linewidth=2)
    
    # Overlay predictions
    colors = {
        1: 'blue',      # bullish_normal
        2: 'cyan',      # bullish_pennant
        3: 'lightblue', # bullish_wedge
        4: 'orange',    # bearish_normal
        5: 'red',       # bearish_pennant
        6: 'darkred'    # bearish_wedge
    }
    
    for start_idx, end_idx, class_id, conf in predictions:
        if class_id == 0:  # Skip no_flag
            continue
        
        color = colors.get(class_id, 'gray')
        rect = Rectangle((start_idx, df.iloc[start_idx:end_idx+1]['low'].min()),
                        end_idx - start_idx + 1,
                        df.iloc[start_idx:end_idx+1]['high'].max() - df.iloc[start_idx:end_idx+1]['low'].min(),
                        alpha=0.3, facecolor=color, edgecolor=color, linewidth=2)
        ax1.add_patch(rect)
        
        # Add label
        mid_idx = (start_idx + end_idx) // 2
        mid_price = df.iloc[mid_idx]['high']
        ax1.text(mid_idx, mid_price, f"{class_names[class_id]}\n{conf:.2f}",
                ha='center', va='bottom', fontsize=8, bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))
    
    ax1.set_xlabel('Bar Index')
    ax1.set_ylabel('Price')
    ax1.set_title(f'Detected Flag Patterns (Top {len(predictions)} by Confidence)')
    ax1.grid(True, alpha=0.3)
    
    # Plot confidence heatmap
    window_size = 64
    confidence_map = np.zeros(len(df))
    counts = np.zeros(len(df))
    
    for i, probs in enumerate(probabilities):
        start_idx = i * 8  # STRIDE = 8
        end_idx = start_idx + window_size
        if end_idx <= len(df):
            # Max probability excluding no_flag class
            max_prob = np.max(probs[1:])
            confidence_map[start_idx:end_idx] += max_prob
            counts[start_idx:end_idx] += 1
    
    # Average confidence
    confidence_map = np.divide(confidence_map, counts, where=counts>0)
    
    ax2.plot(confidence_map, color='purple', linewidth=1)
    ax2.fill_between(range(len(confidence_map)), confidence_map, alpha=0.3, color='purple')
    ax2.set_xlabel('Bar Index')
    ax2.set_ylabel('Confidence')
    ax2.set_title('Pattern Detection Confidence (excluding no_flag)')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved prediction plot to {save_path}")

def predict(csv_path=None, confidence_threshold=0.5, min_window_confidence=0.3, max_bars=20000):
    """
    Run inference on an unannotated CSV file.
    
    Args:
        csv_path: Path to CSV file. If None, finds an unannotated file automatically.
        confidence_threshold: Minimum confidence to report a pattern
        min_window_confidence: Minimum per-window confidence to consider
        max_bars: Maximum number of bars to process (for visualization clarity)
    """
    logger.info("="*60)
    logger.info("INFERENCE CONFIGURATION")
    logger.info("="*60)
    logger.info(f"Confidence threshold: {confidence_threshold}")
    logger.info(f"Min window confidence: {min_window_confidence}")
    logger.info(f"Max bars to process: {max_bars if max_bars else 'unlimited'}")
    logger.info(f"Max patterns to plot: 10")
    logger.info("")
    
    # Find file if not provided
    if csv_path is None:
        csv_path = find_unannotated_file(max_bars=max_bars)
        if csv_path is None:
            logger.error("No CSV file available for inference")
            return
    
    if not os.path.exists(csv_path):
        logger.error(f"File not found: {csv_path}")
        return
    
    logger.info(f"Running inference on: {csv_path}")
    
    # Load metadata
    data_dir = config.DATA_DIR + "/processed"
    metadata_path = os.path.join(data_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    num_classes = metadata['num_classes']
    num_features = metadata['num_features']
    class_names = metadata.get('class_names', [f'Class_{i}' for i in range(num_classes)])
    window_size = metadata['window_size']
    
    logger.info(f"Model configuration: {num_classes} classes, {num_features} features, window={window_size}")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model_path = os.path.join(config.DATA_DIR, 'best_model.pth')
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return
    
    model = FlagCNN(num_features=num_features, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    logger.info(f"Loaded model from {model_path}")
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    logger.info("")
    
    # Load CSV data
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} bars from CSV")
    
    if len(df) < window_size:
        logger.error(f"File too short: {len(df)} bars < {window_size} required")
        return
    
    ohlc = df[['open', 'high', 'low', 'close']].values.astype(np.float32)
    
    # Generate sliding windows
    stride = 8
    windows = []
    window_positions = []
    
    for start in range(0, len(ohlc) - window_size, stride):
        end = start + window_size
        window_ohlc = ohlc[start:end]
        features = extract_features(window_ohlc)
        windows.append(features)
        window_positions.append((start, end))
    
    logger.info(f"Generated {len(windows)} windows with stride={stride}")
    
    # Run inference
    X = np.stack(windows)
    X_tensor = torch.FloatTensor(X).to(device)
    
    all_predictions = []
    all_probabilities = []
    
    batch_size = config.BATCH_SIZE
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch_X = X_tensor[i:i+batch_size]
            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)
            all_probabilities.extend(probs.cpu().numpy())
    
    all_probabilities = np.array(all_probabilities)
    
    # Find patterns with high confidence
    detected_patterns = []
    
    for i, (start, end) in enumerate(window_positions):
        probs = all_probabilities[i]
        pred_class = np.argmax(probs)
        confidence = probs[pred_class]
        
        # Skip no_flag class and low confidence predictions
        if pred_class > 0 and confidence >= min_window_confidence:
            detected_patterns.append((start, end, pred_class, confidence))
    
    logger.info(f"Found {len(detected_patterns)} potential patterns")
    
    # Merge overlapping detections of same class
    merged_patterns = []
    if detected_patterns:
        detected_patterns.sort(key=lambda x: (x[2], x[0]))  # Sort by class, then start
        
        current_class = detected_patterns[0][2]
        current_start = detected_patterns[0][0]
        current_end = detected_patterns[0][1]
        max_confidence = detected_patterns[0][3]
        
        for start, end, cls, conf in detected_patterns[1:]:
            if cls == current_class and start <= current_end:
                # Extend current pattern
                current_end = max(current_end, end)
                max_confidence = max(max_confidence, conf)
            else:
                # Save current and start new
                if max_confidence >= confidence_threshold:
                    merged_patterns.append((current_start, current_end, current_class, max_confidence))
                current_class = cls
                current_start = start
                current_end = end
                max_confidence = conf
        
        # Add last pattern
        if max_confidence >= confidence_threshold:
            merged_patterns.append((current_start, current_end, current_class, max_confidence))
    
    logger.info(f"Merged to {len(merged_patterns)} patterns (confidence >= {confidence_threshold})")
    
    # Report findings
    logger.info(f"\n{'='*60}")
    logger.info("DETECTED PATTERNS")
    logger.info(f"{'='*60}")
    
    if not merged_patterns:
        logger.info("No patterns detected above confidence threshold")
    else:
        for start, end, cls, conf in merged_patterns:
            logger.info(f"\n{class_names[cls]}:")
            logger.info(f"  Position: bars {start} to {end} ({end-start+1} bars)")
            logger.info(f"  Confidence: {conf:.4f}")
            if 'time' in df.columns or df.columns[0].lower() in ['timestamp', 'datetime', 'date']:
                time_col = 'time' if 'time' in df.columns else df.columns[0]
                logger.info(f"  Time range: {df.iloc[start][time_col]} to {df.iloc[end][time_col]}")
    
    # Plot results (only top patterns for clarity)
    output_dir = config.DATA_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    csv_filename = os.path.basename(csv_path).replace('.csv', '')
    plot_path = os.path.join(output_dir, f'inference_{csv_filename}.png')
    
    # Select top N patterns by confidence for plotting
    max_patterns_to_plot = 10
    patterns_to_plot = sorted(merged_patterns, key=lambda x: x[3], reverse=True)[:max_patterns_to_plot]
    
    if len(merged_patterns) > max_patterns_to_plot:
        logger.info(f"Plotting top {max_patterns_to_plot} of {len(merged_patterns)} detected patterns")
    
    plot_predictions(df, patterns_to_plot, all_probabilities, class_names, plot_path)
    
    # Save results to JSON
    results = {
        'file': csv_path,
        'total_bars': len(df),
        'windows_analyzed': len(windows),
        'patterns_detected': len(merged_patterns),
        'confidence_threshold': confidence_threshold,
        'patterns': [
            {
                'type': class_names[cls],
                'start_bar': int(start),
                'end_bar': int(end),
                'length_bars': int(end - start + 1),
                'confidence': float(conf)
            }
            for start, end, cls, conf in merged_patterns
        ]
    }
    
    results_path = os.path.join(output_dir, f'inference_{csv_filename}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")
    
    logger.info("\nInference complete.")

if __name__ == "__main__":
    predict()
