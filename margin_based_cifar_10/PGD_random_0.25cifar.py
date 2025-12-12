import time
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


def pgd_attack(model, x, y, eps=8.0 / 255.0, alpha=2.0 / 255.0, steps=10):
    x_adv = x.clone().detach()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0, 1)

    for _ in range(steps):
        x_adv.requires_grad = True
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv + alpha * torch.sign(grad)
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
        x_adv = torch.clamp(x_adv.detach(), 0, 1)

    return x_adv


def evaluate(model, test_loader, device):
    model.eval()

    clean_correct = 0
    pgd_correct = 0
    total = 0

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        total += y.size(0)

        # clean
        pred = model(x).argmax(1)
        clean_correct += (pred == y).sum().item()

        # PGD
        x_adv = pgd_attack(model, x, y)
        pred_adv = model(x_adv).argmax(1)
        pgd_correct += (pred_adv == y).sum().item()

    clean_acc = clean_correct / total
    pgd_acc = pgd_correct / total
    return clean_acc, pgd_acc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.ToTensor()

    train_set = datasets.CIFAR10("./cifar", train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10("./cifar", train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    model = ResNet18_CIFAR10().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    adv_ratio = 0.25      # always select random 25%
    lambda_clean = 0.2    # same as your main experiment

    total_start = time.time()

    # =============================
    # RANDOM 25% SAMPLING AT
    # =============================
    for epoch in range(200):

        epoch_start = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)

            # -------- Random selection only  -------- #
            k = int(batch_size * adv_ratio)
            idx = torch.randperm(batch_size)[:k]

            x_sel = x[idx]
            y_sel = y[idx]

            # PGD
            x_adv = pgd_attack(model, x_sel, y_sel)

            # loss
            logits_adv = model(x_adv)
            loss_adv = F.cross_entropy(logits_adv, y_sel)

            logits_clean = model(x)
            loss_clean = F.cross_entropy(logits_clean, y)

            loss = loss_adv + lambda_clean * loss_clean

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_end = time.time()
        print(f"Epoch {epoch} finished. Time: {epoch_end - epoch_start:.2f} sec")

        # eval every 20 epochs
        if (epoch + 1) % 20 == 0:
            clean_acc, pgd_acc = evaluate(model, test_loader, device)
            print(f"Test at epoch {epoch}")
            print(f"Clean accuracy: {clean_acc:.4f}")
            print(f"PGD accuracy:   {pgd_acc:.4f}")

    total_end = time.time()
    print(f"Total training time: {(total_end - total_start) / 60:.2f} minutes")

    clean_acc, pgd_acc = evaluate(model, test_loader, device)
    print("Final clean accuracy:", clean_acc)
    print("Final PGD accuracy:", pgd_acc)


if __name__ == "__main__":
    main()
