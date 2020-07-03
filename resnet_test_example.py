import torch
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from model import Network

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main():

    classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

    PATH_ex = 'networks/network.pt'
    netx = Network().cuda()
    netx.load_state_dict(torch.load(PATH_ex))
    netx.eval()


    imsize = 32
    loader_ex = transforms.Compose([
        transforms.Resize(imsize),  # scale imported image
        transforms.ToTensor()])

    image_dir = 'images/image4.jpg'  ## image directory goes here
    image = Image.open(image_dir)   
    x = loader_ex(image)
    idx = netx(x.unsqueeze(0).cuda()).argmax(dim = 1)[0]
    print(classes[idx])

    img = mpimg.imread(image_dir)
    imgplot = plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    main()