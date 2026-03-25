import torch
import torch.nn as nn
import torch.nn.functional as F

# Energy Network (bounded)
class EnergyNet(nn.Module):

    def __init__(self, feature_dim, hidden_dim=256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.interaction = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def compute_pairwise_energy(self, f1, f2):

        B1, B2 = f1.size(0), f2.size(0)

        e1 = self.encoder(f1)
        e2 = self.encoder(f2)

        e1 = e1.unsqueeze(1).expand(-1, B2, -1)
        e2 = e2.unsqueeze(0).expand(B1, -1, -1)

        interaction = e1 * e2

        combined = torch.cat([e1, e2, interaction], dim=-1)

        energy = self.interaction(combined)

        energy = torch.tanh(energy)

        return energy.squeeze(-1)

    def forward(self, f_orig, f_trans):

        return self.compute_pairwise_energy(f_orig, f_trans)
    

    
class UAVEncoder(nn.Module):

    def __init__(self,  bottleneck_dim=64):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64,128,3,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128,256,3,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256,256,3,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()

        )
        # Compress spatial features
        self.compress = nn.Conv2d(256, 128, 1)

        # Bottleneck
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(128, bottleneck_dim)

    def forward(self,x):

        x = self.features(x)
        #print(x.shape)
        
        x = self.compress(x)

        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x



# Channel Model
class Channel(nn.Module):

    def __init__(self, snr_db=20, tx_ratio=1.0):

        super().__init__()

        self.snr_db = snr_db
        self.tx_ratio = tx_ratio

    def set_snr(self, snr):
        self.snr_db = snr

    def set_tx_ratio(self, ratio):
        self.tx_ratio = ratio

    def forward(self, features):

        # normalize (important for SNR control)
        features = F.normalize(features, p=2, dim=1)

        if self.tx_ratio < 1.0:

            mask = (torch.rand_like(features) < self.tx_ratio).float()

            signal_power = features.pow(2).mean(dim=1, keepdim=True)

            noise_power = signal_power / (10 ** (self.snr_db / 10))

            noise = torch.randn_like(features) * torch.sqrt(noise_power)

            features = features * mask + noise * mask

        else:

            signal_power = features.pow(2).mean(dim=1, keepdim=True)

            noise_power = signal_power / (10 ** (self.snr_db / 10))

            noise = torch.randn_like(features) * torch.sqrt(noise_power)

            features = features + noise

        return torch.clamp(features, -3.0, 3.0)




# Edge Server
class EdgeServer(nn.Module):

    def __init__(self, bottleneck_dim=64, num_classes=21):

        super().__init__()

        self.channel = Channel()

        self.fusion = nn.Sequential(

            nn.Linear(bottleneck_dim*2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256,128),
            nn.ReLU()
        )

        self.classifier = nn.Linear(128, num_classes)

        self.clip_projection = nn.Linear(128,512)

    def forward(self, f1, f2):

        f1 = self.channel(f1)
        f2 = self.channel(f2)

        fused = torch.cat([f1, f2], dim=1)

        fused = self.fusion(fused)

        logits = self.classifier(fused)

        fused_clip = self.clip_projection(fused)

        return logits, fused_clip
















