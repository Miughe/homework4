import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .models import Detector
from .utils import DetectionSuperTuxDataset, load_detection_data
from .dense_transforms import Compose, ToHeatmap, ToTensor

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = Compose([ToTensor(), ToHeatmap()])

    train_data = load_detection_data('dense_data/train', num_workers=4, transform=transform)
    valid_data = load_detection_data('dense_data/valid', num_workers=4)

    model = Detector(num_classes=3)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        model.train()
        for batch_idx, (images, heatmaps, _) in enumerate(train_data):
            images = images.to(device)
            heatmaps = heatmaps.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, heatmaps)

            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(images)}/{len(train_data.dataset)} '
                      f'({100. * batch_idx / len(valid_data):.0f}%)]\tLoss: {loss.item():.6f}')

    torch.save(model.state_dict(), 'det.th')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Your Detector Training')
    parser.add_argument('-g', '--gamma', type=float, default=0, help="class dependent weight for cross entropy")
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status (default: 10)')
    parser.add_argument('-t', '--transform',
                        default='Compose([ColorJitter(0.9, 0.9, 0.9, 0.1), RandomHorizontalFlip(), ToTensor()])')
    args = parser.parse_args()
    train(args)