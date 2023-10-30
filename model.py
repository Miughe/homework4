import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import DetectionSuperTuxDataset
from torchvision import transforms

def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
    max_pooled = F.max_pool2d(heatmap[None, None], max_pool_ks, stride=1, padding=max_pool_ks // 2).squeeze()

    is_peak = (heatmap == max_pooled) & (heatmap > min_score)

    peak_indices = torch.nonzero(is_peak, as_tuple=False)
    scores = heatmap[is_peak]

    sorted_indices = torch.argsort(scores, descending=True)
    top_indices = sorted_indices[:max_det]

    valid_peaks = [(scores[i].item(), peak_indices[i, 1].item(), peak_indices[i, 0].item()) for i in top_indices]

    return valid_peaks

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.stride != 1:
            identity = F.avg_pool2d(identity, 2)
            identity = torch.cat((identity, torch.zeros(identity.shape).to(x.device)), dim=1)

        out += identity
        out = self.relu(out)

        return out

class Detector(nn.Module):
    def __init__(self, num_classes=3, output_channels=3, num_residual_blocks=3):
        super(Detector, self).__init__()
        self.num_classes = num_classes
        self.output_channels = output_channels
        self.num_residual_blocks = num_residual_blocks
        self._build_model()

    def _build_model(self):
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.res_blocks = self._make_residual_blocks()
        self.fc = nn.Conv2d(64, self.output_channels, kernel_size=1)

    def _make_residual_blocks(self):
        layers = []
        for _ in range(self.num_residual_blocks):
            layers.append(ResidualBlock(64, 64))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.res_blocks(x)
        heatmaps = self.fc(x)
        return heatmaps

    def detect(self, image):
        heatmaps = self.forward(image[None])
        class_detections = [[] for _ in range(self.num_classes)]
        for class_id in range(self.num_classes):
            heatmap = heatmaps[0][class_id]
            peaks = extract_peak(heatmap)
            peaks = peaks[:30]
            for score, cx, cy in peaks:
                class_detections[class_id].append((score, cx, cy, 0, 0))
        return class_detections



def save_model(model):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'det.th'))

def load_model():
    from torch import load
    from os import path
    r = Detector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'det.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    """
    Shows detections of your detector
    """
    from .utils import DetectionSuperTuxDataset
    dataset = DetectionSuperTuxDataset('dense_data/valid', min_size=0)
    import torchvision.transforms.functional as TF
    from pylab import show, subplots
    import matplotlib.patches as patches

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    fig, axs = subplots(3, 4)
    model = load_model().eval().to(device)
    for i, ax in enumerate(axs.flat):
        im, kart, bomb, pickup = dataset[i]
        ax.imshow(TF.to_pil_image(im), interpolation=None)
        for k in kart:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='r'))
        for k in bomb:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='g'))
        for k in pickup:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='b'))
        detections = model.detect(im.to(device))
        for c in range(3):
            for s, cx, cy, w, h in detections[c]:
                ax.add_patch(patches.Circle((cx, cy), radius=max(2 + s / 2, 0.1), color='rgb'[c]))
        ax.axis('off')
    show()
