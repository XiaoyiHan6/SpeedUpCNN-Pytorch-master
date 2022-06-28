import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
# if you run train_XXX.py, you will use this
from alexnet import alexnet
# if you tun train.py, you will use this
# from torchvision.models.alexnet import alexnet
import argparse


# eval
def parse_args():
    parser = argparse.ArgumentParser(description='CV Evaluation')
    parser.add_mutually_exclusive_group()
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='CIFAR10')
    parser.add_argument('--dataset_root', type=str, default='../data', help='Dataset root directory path')
    parser.add_argument('--img_size', type=int, default=227, help='image size')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--checkpoint', type=str, default='../checkpoint', help='Save .pth fold')
    return parser.parse_args()


args = parse_args()
# 1.Create model
model = alexnet()
model = torch.nn.DataParallel(model)

# 2.Ready Dataset
if args.dataset == 'CIFAR10':
    test_dataset = torchvision.datasets.CIFAR10(root=args.dataset_root, train=False,
                                                transform=transforms.Compose(
                                                    [transforms.Resize(args.img_size),
                                                     transforms.ToTensor()]),
                                                download=True)
else:
    raise ValueError("Dataset is not CIFAR10")
# 3.Length
test_dataset_size = len(test_dataset)
print("the test dataset size is {}".format(test_dataset_size))

# 4.DataLoader
test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size)

# 5. Set some parameters for testing the network
total_accuracy = 0

# test
model.eval()
with torch.no_grad():
    for data in test_dataloader:
        imgs, targets = data
        device = torch.device('cpu')
        imgs, targets = imgs.to(device), targets.to(device)
        model_load = torch.load("{}/AlexNet.pth".format(args.checkpoint), map_location=device)
        model.load_state_dict(model_load)
        outputs = model(imgs)
        outputs = outputs.to(device)
        accuracy = (outputs.argmax(1) == targets).sum()
        total_accuracy = total_accuracy + accuracy
        accuracy = total_accuracy / test_dataset_size
    print("the total accuracy is {}".format(accuracy))
