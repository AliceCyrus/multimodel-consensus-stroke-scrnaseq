"""
OPTIMIZED: Transformer-Inspired Foundation Models with Progress Tracking
=========================================================================
NOTE: These models are INSPIRED BY foundation model architectures (scGPT, scBERT,
scFormer, Geneformer) but are NOT the original pretrained models. They use
transformer architectures trained from scratch on PCA-reduced features.

Quality: PRESERVED (all original architectures)
Change: Added progress indicators during training
"""

import os
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# Import your framework
from phase1_framework import ModelValidator, PublicationFigures

print("="*70)
print("FOUNDATION MODELS - WITH PROGRESS TRACKING")
print("="*70)

# ============================================================================
# GPU CONFIGURATION
# ============================================================================

def setup_gpu():
    """Configure GPU settings"""
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA Available: {cuda_available}")
    
    if cuda_available:
        device = torch.device('cuda:0')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        
        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.empty_cache()
        
        print("✓ GPU optimizations enabled (TF32, cuDNN benchmark)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device

DEVICE = setup_gpu()

CONFIG = {
    'device': DEVICE,
    'batch_size': 128,
    'epochs': 30,
    'learning_rate': 2e-4,
    'early_stopping_patience': 5,
    'random_seed': 42,
    'num_workers': 2,
    'pin_memory': DEVICE.type == 'cuda',
    'mixed_precision': True,
    'checkpoint_dir': 'model_checkpoints'
}

# Create checkpoint directory
os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
print(f"\nConfiguration:")
print(f"  Batch Size: {CONFIG['batch_size']}")
print(f"  Max Epochs: {CONFIG['epochs']}")
print(f"  Learning Rate: {CONFIG['learning_rate']}")
print(f"  Early Stop Patience: {CONFIG['early_stopping_patience']}")
print(f"  Mixed Precision: {CONFIG['mixed_precision']}")
print(f"  Checkpoint Dir: {CONFIG['checkpoint_dir']}/")

# ============================================================================
# DATA LOADING WITH DRIVE PROTECTION
# ============================================================================

def setup_local_data_if_needed():
    """Copy data from Drive to local storage (prevents disconnection)"""
    import shutil
    from pathlib import Path
    
    try:
        from google.colab import drive
        in_colab = True
    except:
        in_colab = False
        # Not on Colab: data lives in the repository itself.
        from repo_paths import data_dir
        return data_dir()
    
    print("\n" + "="*70)
    print("DATA SETUP (One-time copy to local storage)")
    print("="*70)
    
    drive_data_dir = '/content/drive/MyDrive/stroke/Phase4_RevisedModels'
    local_data_dir = '/content/Phase4_RevisedModels'
    
    Path(local_data_dir).mkdir(parents=True, exist_ok=True)
    
    files = [
        'stroke_pca_train.csv',
        'stroke_pca_test.csv',
        'stroke_labels_train.csv',
        'stroke_labels_test.csv'
    ]
    
    all_exist = all(Path(f"{local_data_dir}/{f}").exists() for f in files)
    
    if all_exist:
        print("✓ Data already in local storage")
        return local_data_dir
    
    print("Copying data from Drive to local storage...")
    for filename in files:
        src = f"{drive_data_dir}/{filename}"
        dst = f"{local_data_dir}/{filename}"
        
        if not Path(dst).exists():
            print(f"  {filename}... ", end='', flush=True)
            try:
                shutil.copy2(src, dst)
                print(f"✓")
            except Exception as e:
                print(f"❌ {e}")
                return drive_data_dir
        else:
            print(f"  {filename}... ✓ (cached)")
    
    print(f"\n✓ Data ready in: {local_data_dir}")
    return local_data_dir


def load_data():
    """Load preprocessed stroke data"""
    print("\nLoading data...")
    
    data_dir = setup_local_data_if_needed()
    
    train_pca = pd.read_csv(f'{data_dir}/stroke_pca_train.csv')
    test_pca = pd.read_csv(f'{data_dir}/stroke_pca_test.csv')
    train_labels = pd.read_csv(f'{data_dir}/stroke_labels_train.csv')
    test_labels = pd.read_csv(f'{data_dir}/stroke_labels_test.csv')
    
    X_train = train_pca.drop(columns=['cell_id']).values
    X_test = test_pca.drop(columns=['cell_id']).values
    y_train = (train_labels['condition'] == 'Stroke').astype(int).values
    y_test = (test_labels['condition'] == 'Stroke').astype(int).values
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

class SingleCellDataset(Dataset):
    """PyTorch Dataset for single-cell data"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================================
# MODEL ARCHITECTURES (ORIGINAL SIZES - NO CHANGES!)
# ============================================================================

class scGPTModel(nn.Module):
    """scGPT-inspired architecture"""
    
    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        x = self.input_projection(x).unsqueeze(1)
        x = x + self.pos_encoding
        x = self.transformer(x)
        x = x.squeeze(1)
        logits = self.classifier(x)
        return logits


class scBERTModel(nn.Module):
    """scBERT-inspired architecture"""
    
    def __init__(self, input_dim, d_model=512, nhead=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.gene_embedding = nn.Linear(input_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.pooler = nn.Linear(d_model, d_model)
        self.pooler_activation = nn.Tanh()
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        x = self.gene_embedding(x).unsqueeze(1)
        x = self.layer_norm(x)
        encoded = self.encoder(x)
        pooled = self.pooler(encoded[:, 0, :])
        pooled = self.pooler_activation(pooled)
        logits = self.classifier(pooled)
        return logits


class scFormerModel(nn.Module):
    """scFormer-inspired architecture"""
    
    def __init__(self, input_dim, d_model=384, nhead=6, num_layers=3, dropout=0.1):
        super().__init__()
        
        self.gene_encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.feed_forwards = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers * 2)
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 192),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(192, 2)
        )
    
    def forward(self, x):
        x = self.gene_encoder(x).unsqueeze(1)
        
        for i, (attn, ff) in enumerate(zip(self.attention_layers, self.feed_forwards)):
            x_norm = self.layer_norms[i*2](x)
            attn_out, _ = attn(x_norm, x_norm, x_norm)
            x = x + attn_out
            
            x_norm = self.layer_norms[i*2 + 1](x)
            ff_out = ff(x_norm)
            x = x + ff_out
        
        x = x.squeeze(1)
        logits = self.classifier(x)
        return logits


class GeneformerModel(nn.Module):
    """Geneformer-inspired architecture"""
    
    def __init__(self, input_dim, d_model=512, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        
        self.value_embedding = nn.Linear(1, d_model // 2)
        self.rank_embedding = nn.Linear(1, d_model // 2)
        self.gene_positions = nn.Parameter(torch.randn(1, input_dim, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        ranks = torch.argsort(torch.argsort(x, dim=1), dim=1).float().unsqueeze(-1)
        ranks = ranks / x.size(1)
        
        value_emb = self.value_embedding(x.unsqueeze(-1))
        rank_emb = self.rank_embedding(ranks)
        
        x = torch.cat([value_emb, rank_emb], dim=-1)
        x = x + self.gene_positions
        
        x = self.transformer(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        return logits

# ============================================================================
# SKLEARN-COMPATIBLE WRAPPER WITH PROGRESS TRACKING
# ============================================================================

class FoundationModelClassifier(BaseEstimator, ClassifierMixin):
    """
    Foundation Model Classifier with Progress Tracking
    """
    
    def __init__(self, model_class, model_name, input_dim=50, 
                 d_model=256, nhead=8, num_layers=4, dropout=0.1,
                 epochs=30, batch_size=128, learning_rate=2e-4,
                 early_stopping_patience=5, device='cpu', verbose=1,
                 checkpoint_dir='model_checkpoints'):
        
        self.model_class = model_class
        self.model_name = model_name
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        
        if isinstance(device, torch.device):
            self.device = str(device)
        else:
            self.device = device
        
        self.verbose = verbose
        self.checkpoint_dir = checkpoint_dir
        
        self.model_ = None
        self.is_fitted_ = False
        self.scaler_ = None
        self.checkpoint_path_ = None
        self.device_ = None
    
    def _get_device(self):
        """Convert device string to torch.device"""
        if self.device_ is None:
            self.device_ = torch.device(self.device)
        return self.device_
    
    def get_params(self, deep=True):
        """Get parameters for sklearn cloning"""
        return {
            'model_class': self.model_class,
            'model_name': self.model_name,
            'input_dim': self.input_dim,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'early_stopping_patience': self.early_stopping_patience,
            'device': self.device,
            'verbose': self.verbose,
            'checkpoint_dir': self.checkpoint_dir
        }
    
    def set_params(self, **params):
        """Set parameters for sklearn cloning"""
        for key, value in params.items():
            setattr(self, key, value)
        if 'device' in params:
            self.device_ = None
        return self
    
    def fit(self, X, y, X_val=None, y_val=None):
        """Training with progress tracking"""
        X = np.array(X)
        y = np.array(y)
        
        device = self._get_device()
        
        if X.shape[1] != self.input_dim:
            self.input_dim = X.shape[1]
        
        # ✅ SHOW START TIME
        start_time = time.time()
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        # Initialize model
        self.model_ = self.model_class(
            input_dim=self.input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(device)
        
        # DataLoader
        train_dataset = SingleCellDataset(X, y)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True if device.type == 'cuda' else False,
            prefetch_factor=2 if device.type == 'cuda' else None
        )
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), 
            lr=self.learning_rate, 
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.epochs,
            eta_min=self.learning_rate * 0.1
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # Mixed precision
        use_amp = device.type == 'cuda'
        if use_amp:
            self.scaler_ = torch.cuda.amp.GradScaler()
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        # ✅ SHOW TRAINING START
        if self.verbose:
            print(f"\n[{timestamp}] Training {self.model_name}...")
            print(f"  Config: d_model={self.d_model}, layers={self.num_layers}, "
                  f"batch={self.batch_size}, epochs={self.epochs}")
        else:
            # ✅ MINIMAL OUTPUT EVEN WHEN verbose=0
            print(f"\n[{timestamp}] {self.model_name} training started "
                  f"(d_model={self.d_model}, layers={self.num_layers})...", flush=True)
        
        for epoch in range(self.epochs):
            self.model_.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        logits = self.model_(batch_X)
                        loss = criterion(logits, batch_y)
                    
                    self.scaler_.scale(loss).backward()
                    self.scaler_.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                    self.scaler_.step(optimizer)
                    self.scaler_.update()
                else:
                    logits = self.model_(batch_X)
                    loss = criterion(logits, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                    optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            scheduler.step()
            
            # ✅ SHOW PROGRESS EVERY 5 EPOCHS (EVEN IF verbose=0)
            if epoch % 5 == 0 or epoch == self.epochs - 1:
                elapsed = time.time() - start_time
                timestamp = datetime.now().strftime('%H:%M:%S')
                
                if self.verbose:
                    current_lr = optimizer.param_groups[0]['lr']
                    if X_val is not None:
                        val_loss = self._evaluate(X_val, y_val, criterion)
                        print(f"  [{timestamp}] Epoch {epoch+1:2d}/{self.epochs} - "
                              f"Train: {train_loss:.4f}, Val: {val_loss:.4f}, "
                              f"LR: {current_lr:.6f} ({elapsed/60:.1f}min)")
                    else:
                        print(f"  [{timestamp}] Epoch {epoch+1:2d}/{self.epochs} - "
                              f"Loss: {train_loss:.4f}, LR: {current_lr:.6f} "
                              f"({elapsed/60:.1f}min)")
                else:
                    # ✅ MINIMAL PROGRESS FOR verbose=0
                    progress_pct = (epoch + 1) / self.epochs * 100
                    print(f"  [{timestamp}] {self.model_name}: "
                          f"Epoch {epoch+1}/{self.epochs} ({progress_pct:.0f}%) - "
                          f"Loss: {train_loss:.4f} [{elapsed/60:.1f}min]", 
                          end='\r', flush=True)
            
            # Validation & Early Stopping
            if X_val is not None and y_val is not None:
                val_loss = self._evaluate(X_val, y_val, criterion)
                
                if val_loss < best_loss - 0.001:
                    best_loss = val_loss
                    patience_counter = 0
                    self._save_checkpoint(epoch, best_loss)
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        elapsed = time.time() - start_time
                        timestamp = datetime.now().strftime('%H:%M:%S')
                        print(f"\n  [{timestamp}] {self.model_name}: "
                              f"Early stopping at epoch {epoch+1} ({elapsed/60:.1f}min)")
                        break
            else:
                if train_loss < best_loss - 0.001:
                    best_loss = train_loss
                    self._save_checkpoint(epoch, best_loss)
        
        # Load best checkpoint
        if self.checkpoint_path_ and os.path.exists(self.checkpoint_path_):
            self._load_checkpoint()
        
        self.is_fitted_ = True
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # ✅ SHOW COMPLETION
        total_time = time.time() - start_time
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"\n  [{timestamp}] ✓ {self.model_name} training complete! "
              f"Total time: {total_time/60:.1f} minutes, Best loss: {best_loss:.4f}")
        
        return self
    
    def _save_checkpoint(self, epoch, loss):
        """Save model checkpoint"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.checkpoint_path_ = os.path.join(
            self.checkpoint_dir,
            f"{self.model_name.lower().replace(' ', '_')}_best.pt"
        )
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model_.state_dict(),
            'loss': loss,
            'model_config': {
                'input_dim': self.input_dim,
                'd_model': self.d_model,
                'nhead': self.nhead,
                'num_layers': self.num_layers,
                'dropout': self.dropout
            }
        }
        
        torch.save(checkpoint, self.checkpoint_path_)
    
    def _load_checkpoint(self):
        """Load model checkpoint"""
        if not self.checkpoint_path_ or not os.path.exists(self.checkpoint_path_):
            return False
        
        device = self._get_device()
        checkpoint = torch.load(self.checkpoint_path_, map_location=device)
        self.model_.load_state_dict(checkpoint['model_state_dict'])
        
        return True
    
    def load_from_checkpoint(self, checkpoint_path):
        """Load model from saved checkpoint"""
        self.checkpoint_path_ = checkpoint_path
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        device = self._get_device()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        config = checkpoint['model_config']
        self.model_ = self.model_class(
            input_dim=config['input_dim'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        ).to(device)
        
        self.model_.load_state_dict(checkpoint['model_state_dict'])
        self.is_fitted_ = True
        
        if self.verbose:
            print(f"✓ Model loaded from: {checkpoint_path}")
        
        return self
    
    def _evaluate(self, X, y, criterion):
        """Evaluate on validation set"""
        device = self._get_device()
        
        self.model_.eval()
        dataset = SingleCellDataset(X, y)
        loader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        total_loss = 0
        use_amp = device.type == 'cuda'
        
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        logits = self.model_(batch_X)
                        loss = criterion(logits, batch_y)
                else:
                    logits = self.model_(batch_X)
                    loss = criterion(logits, batch_y)
                
                total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def predict(self, X):
        """Predict class labels"""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        device = self._get_device()
        
        self.model_.eval()
        X = np.array(X)
        dataset = SingleCellDataset(X, np.zeros(len(X)))
        loader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        all_probs = []
        use_amp = device.type == 'cuda'
        
        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(device, non_blocking=True)
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        logits = self.model_(batch_X)
                else:
                    logits = self.model_(batch_X)
                
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
        
        return np.vstack(all_probs)

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Two-stage pipeline with progress tracking"""
    
    print("\n" + "="*70)
    print("TWO-STAGE TRAINING PIPELINE")
    print("="*70)
    print("Stage 1: 10-Fold CV for performance metrics")
    print("Stage 2: Final training for interpretability checkpoints")
    print("\n⏱️ ESTIMATED TIMES (with A100 GPU):")
    print("  scGPT:      ~20-25 minutes")
    print("  scBERT:     ~40-50 minutes (larger model)")
    print("  scFormer:   ~15-20 minutes")
    print("  Geneformer: ~25-30 minutes")
    print("  TOTAL:      ~2-2.5 hours")
    print("="*70)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Initialize validator
    validator = ModelValidator(X_train, y_train, X_test, y_test)
    
    # ✅ ORIGINAL MODEL SIZES (NO CHANGES!)
    models_config = [
        {
            'class': scGPTModel,
            'name': 'scGPT',
            'params': {'d_model': 256, 'nhead': 8, 'num_layers': 4}
        },
        {
            'class': scBERTModel,
            'name': 'scBERT',
            'params': {'d_model': 512, 'nhead': 8, 'num_layers': 6}  # ✅ ORIGINAL SIZE
        },
        {
            'class': scFormerModel,
            'name': 'scFormer',
            'params': {'d_model': 384, 'nhead': 6, 'num_layers': 3}
        },
        {
            'class': GeneformerModel,
            'name': 'Geneformer',
            'params': {'d_model': 512, 'nhead': 8, 'num_layers': 4}
        }
    ]
    
    # Process each model
    for idx, config in enumerate(models_config, 1):
        print(f"\n{'='*70}")
        print(f"[{idx}/4] Processing {config['name']}")
        print(f"{'='*70}")
        
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        
        checkpoint_path = os.path.join(
            CONFIG['checkpoint_dir'],
            f"{config['name'].lower()}_best.pt"
        )
        
        # STAGE 1: CROSS-VALIDATION
        if not os.path.exists(f"results_{config['name'].lower()}.json"):
            print(f"\n📊 STAGE 1: 10-Fold Cross-Validation")
            
            model = FoundationModelClassifier(
                model_class=config['class'],
                model_name=config['name'],
                input_dim=X_train.shape[1],
                **config['params'],
                epochs=CONFIG['epochs'],
                batch_size=CONFIG['batch_size'],
                learning_rate=CONFIG['learning_rate'],
                early_stopping_patience=CONFIG['early_stopping_patience'],
                device=str(CONFIG['device']),
                verbose=0,  # Progress shown by fit() method
                checkpoint_dir=CONFIG['checkpoint_dir']
            )
            
            results = validator.evaluate_model(
                model,
                model_name=config['name'],
                n_folds=10,
                n_bootstrap=1000
            )
            
            filename = f"results_{config['name'].lower()}.json"
            validator.save_results(results, filename)
            print(f"\n✓ Results saved: {filename}")
        else:
            print(f"\n✓ Stage 1 already complete")
        
        # STAGE 2: FINAL TRAINING
        if not os.path.exists(checkpoint_path):
            print(f"\n💾 STAGE 2: Final Training for Checkpoint")
            
            model = FoundationModelClassifier(
                model_class=config['class'],
                model_name=config['name'],
                input_dim=X_train.shape[1],
                **config['params'],
                epochs=CONFIG['epochs'],
                batch_size=CONFIG['batch_size'],
                learning_rate=CONFIG['learning_rate'],
                early_stopping_patience=CONFIG['early_stopping_patience'],
                device=str(CONFIG['device']),
                verbose=1,
                checkpoint_dir=CONFIG['checkpoint_dir']
            )
            
            model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
            print(f"✓ Checkpoint: {checkpoint_path}")
        else:
            print(f"\n✓ Stage 2 already complete")
        
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
    
    # FINAL COMPARISON
    print(f"\n{'='*70}")
    print("GENERATING FINAL COMPARISON")
    print(f"{'='*70}")
    
    perf_table = validator.create_performance_table()
    print("\n", perf_table.to_string(index=False))
    perf_table.to_csv("performance_table_foundation_models.csv", index=False)
    
    comparison_df = validator.compare_models()
    if comparison_df is not None:
        comparison_df.to_csv("statistical_comparison_foundation.csv", index=False)
    
    figures = PublicationFigures(validator)
    figures.create_all_figures(output_dir='figures_foundation_models')
    
    # FINAL SUMMARY
    print(f"\n{'='*70}")
    print("✓ TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"\n📁 Checkpoints: {CONFIG['checkpoint_dir']}/")
    
    for config in models_config:
        checkpoint = f"{config['name'].lower()}_best.pt"
        checkpoint_path = os.path.join(CONFIG['checkpoint_dir'], checkpoint)
        if os.path.exists(checkpoint_path):
            size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
            print(f"  ✅ {checkpoint} ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ {checkpoint} (MISSING!)")
    
    print(f"\n🔬 Ready for Phase 2 (Interpretability)!")

if __name__ == "__main__":
    main()

"""
---

## ✅ **WHAT THIS DOES**

### **NO COMPROMISES:**
- ✅ All model sizes EXACTLY as designed
- ✅ scBERT: 512-dim, 6 layers (full size)
- ✅ All other models: original specs

### **ADDED VISIBILITY:**
- ✅ Timestamps on every message
- ✅ Progress shown even when verbose=0
- ✅ Time elapsed displayed
- ✅ Shows "scBERT training started..." so you know it's working
- ✅ Updates every 5 epochs with progress %
- ✅ Shows total time when complete

### **EXAMPLE OUTPUT:**
```
[14:32:15] scBERT training started (d_model=512, layers=6)...
  [14:32:45] scBERT: Epoch 5/30 (17%) - Loss: 0.2134 [0.5min]
  [14:33:20] scBERT: Epoch 10/30 (33%) - Loss: 0.1876 [1.1min]
  [14:34:00] scBERT: Epoch 15/30 (50%) - Loss: 0.1654 [1.7min]
  ...
  [14:42:30] ✓ scBERT training complete! Total time: 10.2 minutes
  """