import torch
import torch.nn as nn
import timm

class FSLPoseModel(nn.Module):
    def __init__(self):
        super(FSLPoseModel, self).__init__()
        # Load HRNet. features_only=True returns a list of feature maps from different stages
        self.backbone = timm.create_model('hrnet_w32', pretrained=True, features_only=True)
        
        # Based on your error, the last feature map has 1024 channels
        # We will use this to predict 16 joint heatmaps
        self.matching_layer = nn.Conv2d(1024, 16, kernel_size=3, padding=1)

    def forward(self, support_imgs, query_img):
        """
        support_imgs: [Batch, 5, 3, 256, 256]
        query_img: [Batch, 3, 256, 256]
        """
        b, n, c, h, w = support_imgs.shape
        
        # 1. Extract Support Features
        support_flat = support_imgs.view(-1, c, h, w)
        s_feats_list = self.backbone(support_flat)
        s_feats = s_feats_list[-1] # Shape: [B*5, 1024, 8, 8]
        
        # 2. Average Support Features (Prototype)
        # Reshape to separate Batch and Shot: [B, 5, 1024, 8, 8]
        feat_dim = s_feats.shape[1]
        feat_h = s_feats.shape[2]
        feat_w = s_feats.shape[3]
        
        s_feats = s_feats.view(b, n, feat_dim, feat_h, feat_w)
        support_proto = s_feats.mean(dim=1) # [B, 1024, 8, 8]
        
        # 3. Extract Query Features
        q_feats_list = self.backbone(query_img)
        q_feats = q_feats_list[-1] # [B, 1024, 8, 8]
        
        # 4. Fusion / Matching
        # We add the query features to the averaged support template
        combined = q_feats + support_proto 
        heatmaps = self.matching_layer(combined) # Predict 16 heatmaps
        
        return heatmaps

if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FSLPoseModel().to(device)
    
    # Mock data: 1 Batch, 5 Shots, 3 Channels, 256x256 Image
    s_in = torch.randn(1, 5, 3, 256, 256).to(device)
    q_in = torch.randn(1, 3, 256, 256).to(device)
    
    out = model(s_in, q_in)
    print(f"✅ Model Check Passed!")
    print(f"Input: 5 Support + 1 Query")
    print(f"Output Heatmap Shape: {out.shape}") # Should be [1, 16, 8, 8]
