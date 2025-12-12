import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader


class ResNet18_CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = resnet18(num_classes=num_classes)
        self.model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.model.maxpool = nn.Identity()

    def forward(self, x):
        return self.model(x)


def evaluate_clean(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def evaluate_pgd(model, loader, device,
                 eps=8/255, alpha=2/255, steps=20):
    model.eval()
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # random start
        x_adv = x + torch.empty_like(x).uniform_(-eps, eps)
        x_adv = torch.clamp(x_adv, 0, 1)

        for _ in range(steps):
            x_adv.requires_grad = True
            logits = model(x_adv)
            loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, x_adv)[0]

            # PGD update
            x_adv = x_adv + alpha * torch.sign(grad)
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
            x_adv = torch.clamp(x_adv.detach(), 0, 1)

        pred = model(x_adv).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return correct / total


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # No data augmentation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_set = datasets.CIFAR10("./cifar", train=True, download=True,
                                 transform=transform)
    test_set = datasets.CIFAR10("./cifar", train=False, download=True,
                                transform=transform)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    model = ResNet18_CIFAR10().to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150], gamma=0.1
    )

    # Clean Training
    for epoch in range(200):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        if (epoch + 1) % 20 == 0:
            clean_acc = evaluate_clean(model, test_loader, device)
            print(f"Epoch {epoch+1}: clean accuracy = {clean_acc:.4f}")

    # Final evaluation
    clean_acc = evaluate_clean(model, test_loader, device)
    pgd_acc = evaluate_pgd(model, test_loader, device)

    print("\nFinal clean accuracy:", clean_acc)
    print("Final PGD accuracy:", pgd_acc)


if __name__ == "__main__":
    main()
