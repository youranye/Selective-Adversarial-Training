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


def compute_margin(model, x, y):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        correct_logit = logits[range(len(y)), y]
        mask = torch.ones_like(logits, dtype=bool)
        mask[range(len(y)), y] = False
        second_logit, _ = logits[mask].reshape(len(y), -1).max(dim=1)
        margin = correct_logit - second_logit
        return margin


def margin_to_prob(margin, eps=1e-6):
    weight = 1.0 / (margin.abs() + eps)
    prob = weight / weight.sum()
    return prob


def evaluate(model, test_loader, device):
    model.eval()

    clean_correct = 0
    pgd_correct = 0
    total = 0

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        total += y.size(0)

        pred = model(x).argmax(1)
        clean_correct += (pred == y).sum().item()

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

    warmup_epochs = 20
    adv_ratio = 0.25
    lambda_clean = 0.2

    total_start = time.time()  # global timer

    model.train()
    for epoch in range(200):

        epoch_start = time.time()  # epoch timer

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)
            k = int(batch_size * adv_ratio)

            if epoch < warmup_epochs:
                idx = torch.randperm(batch_size)[:k]
            else:
                margin = compute_margin(model, x, y)
                prob = margin_to_prob(margin)
                idx = torch.multinomial(prob, num_samples=k, replacement=False)

            x_sel = x[idx]
            y_sel = y[idx]

            x_adv = pgd_attack(model, x_sel, y_sel)

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
