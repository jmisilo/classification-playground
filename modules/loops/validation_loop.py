import torch
import torch.nn as nn
from tqdm import tqdm

def valid_epoch(model, val_loader, criterion, device):
    if not val_loader:
        return {
            'loss': 0,
            'accuracy': 0
        }

    model.eval()

    binary_classification = isinstance(criterion, nn.BCELoss)

    valid_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            loop = tqdm(val_loader, total=len(val_loader))
            loop.set_description(f'Validation Loss: --- | Validation Acc: ---')
            for batch_idx, (data, target) in enumerate(loop):
                data, target = data.to(device), target.float().to(device) if binary_classification else target.long().to(device)
                
                output = model(data)
                
                loss = criterion(output, target)

                valid_loss += loss.item()

                if binary_classification:
                    predicted = (output >= 0.5).float()
                else:
                    _, predicted = output.max(1)

                total += target.shape[0]
                correct += (predicted == target).sum().item()

                loop.set_description(f'Validation Loss: {valid_loss / (batch_idx + 1):.3f} | Validation Acc: {100 * correct / total:.2f}%')
                loop.refresh()

        return {
            'loss': valid_loss / (batch_idx + 1),
            'accuracy':  correct / total
        }