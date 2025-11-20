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
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv + alpha * torch.sign(grad)
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
        x_adv = torch.clamp(x_adv.detach(), 0, 1)

    return x_adv


# ---------------------------
# Compute margin = (logit_y - second_max_logit)
# ---------------------------
def compute_margin(model, x, y):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        correct_logit = logits[range(len(y)), y]

        # second largest logit
        mask = torch.ones_like(logits, dtype=bool)
        mask[range(len(y)), y] = False
        second_logit, _ = logits[mask].reshape(len(y), -1).max(dim=1)

        margin = correct_logit - second_logit
        return margin  # smaller margin = harder sample


# ---------------------------
# Convert margin → sampling probability
# smaller margin → higher weight
# ---------------------------
def margin_to_prob(margin, eps=1e-6):
    weight = 1.0 / (margin.abs() + eps)  # boundary → large weight
    prob = weight / weight.sum()
    return prob


# ---------------------------
# Main
# ---------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.ToTensor()
    train_set = datasets.MNIST("./mnist", train=True, download=True, transform=transform)
    test_set = datasets.MNIST("./mnist", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    model = MNISTNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    warmup_epochs = 2        # important for stabilization
    adv_ratio =0.25     # use 1/4 samples for PGD
    lambda_clean = 0.2       # weight for clean loss

    # ---------------------------
    # Selective PGD adversarial training
    # ---------------------------
    model.train()
    for epoch in range(5):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            batch_size = x.size(0)
            k = int(batch_size * adv_ratio)

            # ---------------------------
            # Warmup: random selection for first few epochs
            # ---------------------------
            if epoch < warmup_epochs:
                idx = torch.randperm(batch_size)[:k]
            else:
                # true selective sampling: choose small-margin samples
                margin = compute_margin(model, x, y)
                prob = margin_to_prob(margin)
                idx = torch.multinomial(prob, num_samples=k, replacement=False)

            x_sel = x[idx]
            y_sel = y[idx]

            # ---------------------------
            # PGD only on selected samples
            # ---------------------------
            x_adv = pgd_attack(model, x_sel, y_sel)

            # ---------------------------
            # Clean + adversarial mixed loss
            # ---------------------------
            logits_adv = model(x_adv)
            loss_adv = F.cross_entropy(logits_adv, y_sel)

            logits_clean = model(x)
            loss_clean = F.cross_entropy(logits_clean, y)

            loss = loss_adv + lambda_clean * loss_clean

            optimizer.zero_grad()
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
