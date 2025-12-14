# Model evaluation script
# This script evaluates the trained model on the test set and generates metrics.
import config
from utils import setup_logger
import torch
import numpy as np
import json
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from model import FlagCNN

logger = setup_logger()

def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Saved confusion matrix to {save_path}")

def evaluate():
    logger.info("Evaluating model on test set...")
    
    # Load test data
    data_dir = config.DATA_DIR + "/processed"
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    logger.info(f"Loaded test data: X={X_test.shape}, y={y_test.shape}")
    logger.info(f"Test class distribution: {np.bincount(y_test)}")
    
    # Load metadata
    with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    num_classes = metadata['num_classes']
    num_features = metadata['num_features']
    class_names = metadata.get('class_names', [f'Class_{i}' for i in range(num_classes)])
    
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Class names: {class_names}")
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X_test)
    y_tensor = torch.LongTensor(y_test)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. PyTorch might not be installed with CUDA support.")
        logger.warning("To enable GPU support, install PyTorch with CUDA:")
        logger.warning("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    else:
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load the best model
    model_path = os.path.join(config.DATA_DIR, 'best_model.pth')
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}. Please train the model first.")
        return
    
    model = FlagCNN(num_features=num_features, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    logger.info(f"Loaded model from {model_path}")
    
    # Make predictions
    all_predictions = []
    all_true = []
    all_probs = []
    
    batch_size = config.BATCH_SIZE
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            batch_X = X_tensor[i:i+batch_size].to(device)
            batch_y = y_tensor[i:i+batch_size]
            
            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_true.extend(batch_y.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_true = np.array(all_true)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_true, all_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_true, all_predictions, average='weighted', zero_division=0
    )
    
    logger.info(f"\n{'='*60}")
    logger.info("TEST SET EVALUATION RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Overall Accuracy: {accuracy*100:.2f}%")
    logger.info(f"Weighted Precision: {precision:.4f}")
    logger.info(f"Weighted Recall: {recall:.4f}")
    logger.info(f"Weighted F1-Score: {f1:.4f}")
    
    # Per-class metrics
    logger.info(f"\n{'='*60}")
    logger.info("PER-CLASS METRICS")
    logger.info(f"{'='*60}")
    
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        all_true, all_predictions, average=None, zero_division=0, labels=list(range(num_classes))
    )
    
    for i, class_name in enumerate(class_names):
        logger.info(f"\n{class_name}:")
        logger.info(f"  Precision: {precision_per_class[i]:.4f}")
        logger.info(f"  Recall: {recall_per_class[i]:.4f}")
        logger.info(f"  F1-Score: {f1_per_class[i]:.4f}")
        logger.info(f"  Support: {support_per_class[i]}")
    
    # Confusion matrix
    cm = confusion_matrix(all_true, all_predictions)
    logger.info(f"\n{'='*60}")
    logger.info("CONFUSION MATRIX")
    logger.info(f"{'='*60}")
    logger.info(f"\n{cm}")
    
    # Save confusion matrix plot
    cm_path = os.path.join(config.DATA_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(cm, class_names, cm_path)
    
    # Classification report
    logger.info(f"\n{'='*60}")
    logger.info("DETAILED CLASSIFICATION REPORT")
    logger.info(f"{'='*60}")
    report = classification_report(all_true, all_predictions, target_names=class_names, 
                                   labels=list(range(num_classes)), zero_division=0)
    logger.info(f"\n{report}")
    
    # Save results to JSON
    results = {
        'test_accuracy': float(accuracy),
        'weighted_precision': float(precision),
        'weighted_recall': float(recall),
        'weighted_f1': float(f1),
        'per_class_metrics': {
            class_names[i]: {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1_score': float(f1_per_class[i]),
                'support': int(support_per_class[i])
            }
            for i in range(len(class_names))
        },
        'confusion_matrix': cm.tolist(),
        'class_names': class_names
    }
    
    results_path = os.path.join(config.DATA_DIR, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved evaluation results to {results_path}")
    
    logger.info("\nEvaluation complete.")

if __name__ == "__main__":
    evaluate()
