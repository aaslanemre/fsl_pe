import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import numpy as np

from dataset import MPIIFewShotDataset
from model import FSLPoseModel

# --- PCK-O METRIC LOGIC ---
def calculate_pck_split(pred_heatmaps, target_heatmaps, vis_flags, threshold=1.0):
    # Get max-indices (coordinates) from 8x8 heatmaps
    pred_coords = torch.argmax(pred_heatmaps.view(pred_heatmaps.size(0), 16, -1), dim=-1)
    target_coords = torch.argmax(target_heatmaps.view(target_heatmaps.size(0), 16, -1), dim=-1)
    
    px, py = pred_coords % 8, pred_coords // 8
    tx, ty = target_coords % 8, target_coords // 8
    
    dist = torch.sqrt((px - tx).float()**2 + (py - ty).float()**2)
    correct = (dist < threshold).float()

    # Split: Vis (1.0) and Occ (0.0)
    vis_mask = (vis_flags == 1).float()
    occ_mask = (vis_flags == 0).float()

    # Calculate mean only for existing joints in that category
    pck_v = (correct * vis_mask).sum() / (vis_mask.sum() + 1e-6)
    pck_o = (correct * occ_mask).sum() / (occ_mask.sum() + 1e-6)

    return pck_v.item(), pck_o.item()

def train():
    wandb.init(project="fsl-pose-mpii-research")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = MPIIFewShotDataset('data/mpii/annotations/train.json', 'data/mpii/images')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    model = FSLPoseModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/10")

        for batch in pbar:
            s_imgs = batch['support'].to(device)
            q_img = batch['query'].to(device)
            target = batch['target'].to(device)
            vis = batch['vis'].to(device)

            optimizer.zero_grad()
            output = model(s_imgs, q_img)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Calculate Split PCK
            pck_v, pck_o = calculate_pck_split(output, target, vis)
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'PCK-O': f"{pck_o:.2%}"})
            
            wandb.log({
                "batch_loss": loss.item(),
                "PCK_Visible": pck_v,
                "PCK_Occluded": pck_o
            })

        torch.save(model.state_dict(), f"fsl_research_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()
