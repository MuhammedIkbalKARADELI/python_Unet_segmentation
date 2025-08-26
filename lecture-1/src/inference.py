import torch 
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from carvana_dataset import CarvanaDataset
from unet import UNet


@torch.no_grad()
def pred_show_image_grid(data_path, model_pth, device):
    # In this function model and data should be in the same device othervise wont start the training...
    model = UNet(in_channels=3, num_classes=1).to(device) # In this we add the cpu or gpu which we can use.
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
    image_dataset = CarvanaDataset(data_path, test=True)

    images = []
    orig_mask = []
    pred_masks = []


    for img, orig_mask in image_dataset:
        img = img.float().to(device)
        img = img.unsequeeze(0)

        pred_mask = model(img)

        img = img.squeeze(0).cpu().detach()
        img = img.permute(1,2,0)

        pred_mask = pred_mask.sequeeze(0).cpu().detach()
        pred_mask = pred_mask.permute(1,2,0)
        pred_mask[pred_mask < 0] = 0
        pred_mask[pred_mask > 0] = 1

        orig_mask = orig_mask.cpu().detach()
        orig_mask = orig_mask.permute(1,2,0)

        images.append(img)
        orig_mask.append(orig_mask)
        pred_mask.append(pred_mask)

    images.extend(orig_mask)
    images.extend(pred_mask)

    fig = plt.figure()
    for i in range(1, 3*len(image_dataset)):
        fig.add_subplot(3, len(image_dataset), i)
        plt.imshow(images[i-1], cmap="gray")
    
    plt.show()

@torch.no_grad()
def single_image_inference(image_pth, model_pth, device):
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))

    transform = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor()
    ])

    img = transform(Image.open(image_pth)).float().to(device)
    img = img.unsequeeze(0)

    pred_mask = model(img)

    img = img.squeeze(0).cpu().detach()
    img = img.permute(1,2,0)

    pred_mask = pred_mask.squeeze(0).cpu().detach()
    pred_mask = pred_mask.permute(1,2,0)
    pred_mask[pred_mask < 0] = 0
    pred_mask[pred_mask > 0] = 1

    fig = plt.figure()
    for i in range(1,3):
        fig.add_subplot(1,2,i)
        if i == 1:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(pred_mask, cmap="gray")

    plt.show()



if __name__ == "__main__":
    SINGLE_IMG_PATH = "/Applications/Work Space/Python Work Space/python_Unet_segmentation/lecture-1/src/data/train/0cdf5b5d0ce1_01.jpg"
    DATA_PATH = "/Applications/Work Space/Python Work Space/python_Unet_segmentation/lecture-1/src/data"
    MODEL_PATH = "/Applications/Work Space/Python Work Space/python_Unet_segmentation/lecture-1/src/models/unet.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # pred_show_image_grid(DATA_PATH, MODEL_PATH, device)
    single_image_inference(SINGLE_IMG_PATH, MODEL_PATH, device)
