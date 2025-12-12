import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ---------------------------
# MNIST network
# ---------------------------
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ---------------------------
# PGD attack
# ---------------------------
def pgd_attack(model, x, y, eps=0.3, alpha=0.01, steps=40):
    x_adv = x.clone().detach()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0, 1)

    for _ in range(steps):
        x_adv.requires_grad = True
        logits = model(x_adv)
        loss = nn.CrossEntropyLoss()(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv + alpha * torch.sign(grad)
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
        x_adv = torch.clamp(x_adv.detach(), 0, 1)

    return x_adv


# ---------------------------
# Main
# ---------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.ToTensor()
    train_set = datasets.MNIST("./mnist", train=True, download=True, transform=transform)
    test_set = datasets.MNIST("./mnist", train=False, download=True, transform=transform)

    print(len(train_set))
    print(len(test_set))

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
    print(len(test_loader))
    print(len(train_loader))

    model = MNISTNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ---------------------------
    # PGD adversarial training
    # ---------------------------
    model.train()
    for epoch in range(5):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # generate PGD adversarial samples
            x_adv = pgd_attack(model, x, y)

            optimizer.zero_grad()
            logits = model(x_adv)
            loss = nn.CrossEntropyLoss()(logits, y)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} done")

    # ---------------------------
    # Evaluate clean accuracy
    # ---------------------------
    model.eval()
    clean_correct = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        clean_correct += (pred == y).sum().item()

    # ---------------------------
    # Evaluate PGD accuracy
    # ---------------------------
    pgd_correct = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        x_adv = pgd_attack(model, x, y)
        pred_adv = model(x_adv).argmax(1)
        pgd_correct += (pred_adv == y).sum().item()

    print("Clean accuracy:", clean_correct / len(test_set))
    print("PGD accuracy:", pgd_correct / len(test_set))


if __name__ == "__main__":
    main()
