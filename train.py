import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import PrunableNet, PrunableLinear, PrunableConv2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


#  Data transforms  (defined at module level — safe on Windows)
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])




def sparsity_loss(model: nn.Module) -> torch.Tensor:
    loss = torch.tensor(0.0, device=device)
    eps  = 1e-6
    for module in model.modules():
        if isinstance(module, (PrunableLinear, PrunableConv2d)):
            g       = torch.sigmoid(module.gate_scores)
            entropy = -(g * torch.log(g + eps) + (1 - g) * torch.log(1 - g + eps))
            loss   += entropy.mean()
    return loss


#  Hard pruning
def apply_hard_pruning(model: nn.Module) -> dict:
    masks = {}
    for name, module in model.named_modules():
        if isinstance(module, (PrunableLinear, PrunableConv2d)):
            gates = torch.sigmoid(module.gate_scores.detach())
            mask  = (gates > 0.5).float()

            if module.weight.dim() == 4:
                expanded = mask.view(-1, 1, 1, 1).expand_as(module.weight)
            else:
                expanded = mask.view(-1, 1).expand_as(module.weight)

            module.weight.data *= expanded
            masks[name] = expanded.clone()
    return masks

#  Evaluation
def evaluate(model: nn.Module, loader) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            correct += model(data).argmax(dim=1).eq(target).sum().item()
    return 100.0 * correct / len(loader.dataset)


#  Weight sparsity
def compute_weight_sparsity(model: nn.Module) -> float:
    total, zero = 0, 0
    for module in model.modules():
        if isinstance(module, (PrunableLinear, PrunableConv2d)):
            w      = module.weight.data
            total += w.numel()
            zero  += (w.abs() < 1e-6).sum().item()
    return 100.0 * zero / total if total > 0 else 0.0


#  Fine-tune
def finetune(model: nn.Module, masks: dict, loader, test_loader,
             epochs: int = 3, lr: float = 3e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                for name, module in model.named_modules():
                    if name in masks:
                        module.weight.data *= masks[name]

            total_loss += loss.item()

        scheduler.step()
        acc = evaluate(model, test_loader)
        print(f"    Finetune epoch {epoch+1}/{epochs} | "
              f"loss: {total_loss:.2f} | acc: {acc:.2f}%")



#everything that spawns workers must be inside here
def main():
    print("Using device:", device)

    # --- dataloaders (created inside main) ---
    train_dataset = datasets.CIFAR10(root='./data', train=True,  download=True, transform=train_transform)
    test_dataset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    train_subset  = torch.utils.data.Subset(train_dataset, range(20000))

    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=128, shuffle=True,
        num_workers=2,          
        pin_memory=True,
        persistent_workers=True 
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    lambda_values = [0.01, 0.05, 0.1]
    results       = []

    for lambda_sparse in lambda_values:
        print(f"\n{'='*55}")
        print(f"  Soft training  —  lambda = {lambda_sparse}")
        print(f"{'='*55}")

        model     = PrunableNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        epochs    = 5
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        # ---------- soft training ----------
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0

            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                loss  = criterion(model(data), target)
                loss += lambda_sparse * sparsity_loss(model)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            gate_sparsity = model.get_total_sparsity()
            print(f"  Epoch {epoch+1}/{epochs} | loss: {total_loss:.2f} | "
                  f"gate sparsity: {gate_sparsity:.1f}% | "
                  f"lr: {scheduler.get_last_lr()[0]:.6f}")

        # ---------- layer-wise gate breakdown ----------
        print("\n  Gate sparsity per layer:")
        for name, pct in model.get_layer_sparsities().items():
            print(f"    {name:40s}  {pct}")

        # ---------- hard pruning ----------
        print("\n  Applying hard pruning...")
        masks   = apply_hard_pruning(model)
        pre_acc = evaluate(model, test_loader)
        pre_sp  = compute_weight_sparsity(model)
        print(f"  Post-prune (before finetune) | acc: {pre_acc:.2f}% | "
              f"weight sparsity: {pre_sp:.2f}%")

        # ---------- fine-tune ----------
        print("\n  Fine-tuning with frozen masks...")
        finetune(model, masks, train_loader, test_loader, epochs=3, lr=3e-4)

        # ---------- final result ----------
        accuracy = evaluate(model, test_loader)
        sparsity = compute_weight_sparsity(model)
        print(f"\n  RESULT → lambda: {lambda_sparse} | "
              f"accuracy: {accuracy:.2f}% | weight sparsity: {sparsity:.2f}%")

        results.append((lambda_sparse, accuracy, sparsity))

    print(f"\n{'='*55}")
    print("  FINAL RESULTS")
    print(f"{'='*55}")
    print(f"  {'Lambda':<10} {'Accuracy':>10} {'Sparsity':>12}")
    print(f"  {'-'*34}")
    for l, acc, sp in results:
        print(f"  {l:<10} {acc:>9.2f}%  {sp:>10.2f}%")


if __name__ == '__main__':
    main()