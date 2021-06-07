import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import load_checkpoint
import config
from dataset import HorseZebraDataset
from generator import Generator
from torchvision.utils import save_image
from tqdm import tqdm

val_dataset = HorseZebraDataset(root_zebra=config.VAL_DIR+"\\zebra",root_horse=config.VAL_DIR+"\\horse",transform=config.transforms)
val_loader = DataLoader(val_dataset,batch_size=1,shuffle=False)
gen_H = Generator(img_channels=3,num_residuals=9).to(config.DEVICE)
gen_Z = Generator(img_channels=3,num_residuals=9).to(config.DEVICE)
opt_gen = optim.Adam(
        list(gen_H.parameters()) + list(gen_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas= (0.5,0.999)
    )
load_checkpoint(config.CHECKPOINT_GEN_H,gen_H,opt_gen,config.LEARNING_RATE)
# load_checkpoint(config.CHECKPOINT_GEN_Z,gen_Z,opt_gen,config.LEARNING_RATE)

def save_some_examples(gen_H, val_loader, folder):
    loop = tqdm(val_loader,leave=True)
    for idx, (zebra,horse) in enumerate(loop):
        zebra = zebra.to(config.DEVICE)
        # horse = horse.to(config.DEVICE)
        gen_H.eval()
        with torch.no_grad():
            y_fake = gen_H(zebra)
            y_fake = y_fake * 0.5 + 0.5  # remove normalization#
            save_image(y_fake, folder + f"/fake_horse{idx}.png")
            save_image(zebra * 0.5 + 0.5, folder + f"/zebra{idx}.png")
        gen_H.train()


if __name__=="__main__":
    save_some_examples(gen_H,val_loader,folder='evaluation')