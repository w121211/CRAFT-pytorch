import argparse
import glob

from PIL import Image
import torch
import torchvision.transforms as transforms

from algo.kmeans import lloyd
from model.vgg import VGGFeatures


def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imsize", type=int, default=128)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--save_to", type=str, default="../../my-dataset")
    return parser.parse_args()


if __name__ == "__main__":
    opt = get_parameters()
    opt.root = "../tmp"
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VGGFeatures().to(opt.device)
    model.eval()
    trans = transforms.Compose(
        [
            transforms.Resize((opt.imsize, opt.imsize / 2)),
            transforms.ToTensor(),
        ]
    )

    feats = []
    for f in glob.glob(opt.root + "/*"):
        x = trans(Image.open(f))
        feats.append(model(x))
    feats = torch.cat(feats, dim=0).numpy()

    idx, ctr = lloyd(feats, 2, opt.device)

