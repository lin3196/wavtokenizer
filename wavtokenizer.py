import torch
import torch.nn as nn
from decoder.feature_extractors import EncodecFeatures
from decoder.models import VocosBackbone
from decoder.heads import ISTFTHead


class WavTokenizer(nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()
        feature_extractor = EncodecFeatures(**config['network_config']['feature_extractor'])
        backbone = VocosBackbone(**config['network_config']['backbone'])
        head = ISTFTHead(**config['network_config']['head'])
        
        self.feature_extractor = feature_extractor
        self.backbone = backbone
        self.head = head
    
    def forward(self, audio_input, **kwargs):
        features, _, commit_loss = self.feature_extractor(audio_input, **kwargs)
        x = self.backbone(features, **kwargs)
        audio_output = self.head(x)
        
        return audio_output, commit_loss

    def encode_infer(self, audio_input: torch.Tensor, **kwargs) -> torch.Tensor:
        features, discrete_codes, _ = self.feature_extractor.infer(audio_input, **kwargs)
        return features,discrete_codes
    
    def decode(self, features_input: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.backbone(features_input, **kwargs)
        audio_output = self.head(x)
        return audio_output  # (B, T)
    
    def infer(self, audio_input):
        device = audio_input.device
        
        bandwidth_id = torch.tensor([0]).to(device)
        features, _ = self.encode_infer(audio_input, bandwidth_id=bandwidth_id)
        audio_output = self.decode(features, bandwidth_id=bandwidth_id)
        
        return audio_output


if __name__ == "__main__":
    from omegaconf import OmegaConf
    
    config = OmegaConf.load("configs/cfg_train.yaml")
    
    model = WavTokenizer(config)
    
    x = torch.randn(1, 16000*3)
    bandwidth_id = torch.tensor([0])
    y, loss = model(x, bandwidth_id=bandwidth_id)
    
    print(y.shape)
    print(loss)