import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class AttentionLayer(nn.Module):
    """Self-attention layer for feature importance weighting."""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, input_dim]
        attention_weights = self.attention(x)  # [batch_size, seq_len, 1]
        attended = x * attention_weights  # [batch_size, seq_len, input_dim]
        return attended, attention_weights

class ResidualBlock(nn.Module):
    """Residual block with batch normalization and dropout."""
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout_rate: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim),
            nn.BatchNorm1d(input_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.block(x))

class EnhancedDysarthriaClassifier(nn.Module):
    """Enhanced classifier with attention and residual connections."""
    
    def __init__(self, input_dim: int, num_classes: int = 4, 
                 hidden_dims: list = [256, 128, 64], dropout_rate: float = 0.3):
        super().__init__()
        
        # Feature processing layers
        self.feature_processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dims[0], hidden_dims[0], dropout_rate)
            for _ in range(2)
        ])
        
        # Attention layer
        self.attention = AttentionLayer(hidden_dims[0])
        
        # Additional processing layers
        self.processing_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.BatchNorm1d(hidden_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            for i in range(len(hidden_dims)-1)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], num_classes)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Initial feature processing
        x = self.feature_processor(x)
        
        # Apply residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Apply attention
        x = x.unsqueeze(1)  # Add sequence dimension
        attended, attention_weights = self.attention(x)
        x = attended.squeeze(1)  # Remove sequence dimension
        
        # Additional processing
        for layer in self.processing_layers:
            x = layer(x)
        
        # Output
        logits = self.output_layer(x)
        
        return logits, attention_weights

def train_enhanced_model(model: nn.Module,
                      train_loader: torch.utils.data.DataLoader,
                      val_loader: torch.utils.data.DataLoader,
                      device: torch.device,
                      num_epochs: int = 50,
                      lr: float = 0.001,
                      weight_decay: float = 1e-4) -> Tuple[nn.Module, list, list]:
    """Train the enhanced model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                         factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            # Handle both single output and tuple output models
            if isinstance(outputs, tuple):
                outputs = outputs[0]
                
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                # Handle both single output and tuple output models
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                    
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
        
        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
    
    return model, train_losses, val_losses 