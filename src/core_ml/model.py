import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class DeepfakeHybridModel(nn.Module):
    def __init__(self, hidden_dim=256, lstm_layers=1, dropout_rate=0.3, freeze_cnn=True):
        super(DeepfakeHybridModel, self).__init__()
        
        weights = MobileNet_V2_Weights.DEFAULT
        self.cnn = mobilenet_v2(weights=weights).features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        if freeze_cnn:
            for i, layer in enumerate(self.cnn.children()):
                if i < 14: 
                    for param in layer.parameters():
                        param.requires_grad = False

        lstm_input_size = 1280 + 8 

        self.lstm = nn.LSTM(
            input_size=lstm_input_size, 
            hidden_size=hidden_dim, 
            num_layers=lstm_layers, 
            batch_first=True,
            dropout=dropout_rate if lstm_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, fft_scores):
        batch_size, seq_len, c, h, w = x.size()

        x = x.view(batch_size * seq_len, c, h, w)
        
        cnn_out = self.cnn(x)               
        cnn_out = self.pool(cnn_out)        
        
        cnn_out = cnn_out.view(batch_size, seq_len, -1) 
        
        combined_features = torch.cat((cnn_out, fft_scores), dim=2) 
        
        lstm_out, _ = self.lstm(combined_features)
        
        final_timestep_out = lstm_out[:, -1, :] 
        
        out = self.dropout(final_timestep_out)
        logits = self.fc(out)
        
        return logits 