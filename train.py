import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import os

from dataset import MPIIFewShotDataset
from model import FSLPoseModel

# --- CONFIG ---
EPOCHS = 10
BATCH_SIZE = 8 # Increased slightly for stability if GPU allows
LR = 1e-4

def train():
    wandb.init(project="fsl-pose-mpii", config={"lr": LR, "epochs": EPOCHS, "batch": BATCH_SIZE})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = MPIIFewShotDataset('data/mpii/annotations/train.json', 'data/mpii/images')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = FSLPoseModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in pbar:
            s_imgs = batch['support'].to(device)
            q_img = batch['query'].to(device)
            target = batch['target'].to(device)

            optimizer.zero_grad()
            output = model(s_imgs, q_img)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.5f}"})
            wandb.log({"batch_loss": loss.item()})

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.5f}")
        wandb.log({"avg_epoch_loss": avg_loss})
        
        torch.save(model.state_dict(), "best_fsl_model.pth")

    wandb.finish()

if __name__ == "__main__":
    train()
