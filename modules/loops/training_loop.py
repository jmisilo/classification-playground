import torch
import torch.nn as nn
from tqdm import tqdm

def train_epoch(epoch, model, train_loader, criterion, optimizer, scaler, device):
    model.train()
    
    binary_classification = isinstance(criterion, nn.BCELoss)

    train_loss = 0
    correct = 0
    total = 0

    loop = tqdm(train_loader, total=len(train_loader))
    loop.set_description(f'Epoch: {epoch} | Loss: --- | Acc: ---')
    for batch_idx, (data, target) in enumerate(loop):

        data, target = data.to(device), target.float().to(device) if binary_classification else target.long().to(device)

        with torch.cuda.amp.autocast():
            output = model(data)

            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.3)

        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()

        train_loss += loss.item()

        if binary_classification:
            predicted = (output >= 0.5).float()
        else:
            _, predicted = output.max(1)

        total += target.shape[0]
        correct += (predicted == target).sum().item()

        loop.set_description(f'Epoch: {epoch} | Loss: {train_loss / (batch_idx + 1):.3f} | Acc: {100 * correct / total:.2f}%')
        loop.refresh()

    return {
        'loss': train_loss / len(train_loader),
        'accuracy': correct / total
    }