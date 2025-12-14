# Model training script
# This script defines the model architecture and runs the training loop.
import config
from utils import setup_logger
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
import numpy as np
import json
import os
from model import FlagCNN

logger = setup_logger()

def train():
    logger.info("Starting training process...")
    logger.info(f"Loaded configuration. Epochs: {config.EPOCHS}, Batch size: {config.BATCH_SIZE}")
    
    # Load training data
    data_dir = config.DATA_DIR + "/processed"
    X_train_val = np.load(os.path.join(data_dir, 'X_train_val.npy'))
    y_train_val = np.load(os.path.join(data_dir, 'y_train_val.npy'))
    
    logger.info(f"Loaded data: X={X_train_val.shape}, y={y_train_val.shape}")
    logger.info(f"Class distribution: {np.bincount(y_train_val)}")
    
    # Load metadata to get number of classes
    with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    num_classes = metadata['num_classes']
    num_features = metadata['num_features']
    logger.info(f"Number of classes: {num_classes}, features: {num_features}")
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X_train_val)
    y_tensor = torch.LongTensor(y_train_val)
    
    # K-Fold Cross Validation
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. PyTorch might not be installed with CUDA support.")
        logger.warning("To enable GPU support, install PyTorch with CUDA:")
        logger.warning("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    else:
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Calculate class weights for imbalanced dataset
    class_counts = np.bincount(y_train_val, minlength=num_classes)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = torch.FloatTensor(class_weights).to(device)
    logger.info(f"Class weights: {class_weights}")
    
    fold_results = []
    
    # K-Fold training
    for fold, (train_ids, val_ids) in enumerate(kfold.split(X_tensor)):
        logger.info(f"\n{'='*50}")
        logger.info(f"FOLD {fold + 1}/{k_folds}")
        logger.info(f"{'='*50}")
        
        # Split data for this fold
        X_train_fold = X_tensor[train_ids]
        y_train_fold = y_tensor[train_ids]
        X_val_fold = X_tensor[val_ids]
        y_val_fold = y_tensor[val_ids]
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_fold, y_train_fold)
        val_dataset = TensorDataset(X_val_fold, y_val_fold)
        
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        
        # Initialize model
        model = FlagCNN(num_features=num_features, num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, config.EPOCHS + 1):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += batch_y.size(0)
                train_correct += predicted.eq(batch_y).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = 100.0 * train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += batch_y.size(0)
                    val_correct += predicted.eq(batch_y).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = 100.0 * val_correct / val_total
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Log progress every 10 epochs
            if epoch % 10 == 0 or epoch == 1:
                logger.info(f"Epoch {epoch}/{config.EPOCHS} - "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model for this fold
                model_path = os.path.join(config.DATA_DIR, f'best_model_fold{fold+1}.pth')
                torch.save(model.state_dict(), model_path)
            else:
                patience_counter += 1
                if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
        
        # Load best model and evaluate on validation set
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()
        
        fold_acc = 100.0 * val_correct / val_total
        fold_results.append({
            'fold': fold + 1,
            'best_val_loss': best_val_loss,
            'val_accuracy': fold_acc
        })
        logger.info(f"Fold {fold + 1} final validation accuracy: {fold_acc:.2f}%")
    
    # Summary of all folds
    logger.info(f"\n{'='*50}")
    logger.info("K-FOLD CROSS VALIDATION RESULTS")
    logger.info(f"{'='*50}")
    for result in fold_results:
        logger.info(f"Fold {result['fold']}: Val Loss = {result['best_val_loss']:.4f}, Val Acc = {result['val_accuracy']:.2f}%")
    
    avg_acc = np.mean([r['val_accuracy'] for r in fold_results])
    std_acc = np.std([r['val_accuracy'] for r in fold_results])
    logger.info(f"\nAverage Validation Accuracy: {avg_acc:.2f}% (+/- {std_acc:.2f}%)")
    
    # Save results
    results_path = os.path.join(config.DATA_DIR, 'kfold_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'fold_results': fold_results,
            'average_accuracy': avg_acc,
            'std_accuracy': std_acc
        }, f, indent=2)
    logger.info(f"Saved results to {results_path}")
    
    # Find the best fold model based on validation accuracy
    best_fold = max(fold_results, key=lambda x: x['val_accuracy'])
    best_fold_num = best_fold['fold']
    best_fold_acc = best_fold['val_accuracy']
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Best fold: Fold {best_fold_num} with {best_fold_acc:.2f}% validation accuracy")
    
    # Copy the best fold model to best_model.pth
    import shutil
    best_fold_path = os.path.join(config.DATA_DIR, f'best_model_fold{best_fold_num}.pth')
    best_model_path = os.path.join(config.DATA_DIR, 'best_model.pth')
    shutil.copy(best_fold_path, best_model_path)
    logger.info(f"Copied {best_fold_path} to {best_model_path}")
    
    logger.info("Training complete.")

if __name__ == "__main__":
    train()
