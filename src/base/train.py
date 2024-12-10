import torch
import torch.nn.functional as F
from torcheval.metrics.functional import multiclass_accuracy
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

class Trainer:
    def __init__(self, device, model, cfg):
        self.device = device
        self.model = model
        self.max_epochs = cfg['max_epochs']
        self.task = cfg['task']
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg['max_epochs'])
    
    def compute_loss(self, pred, tgt):
        if self.task == 'regression':
            return F.mse_loss(pred, tgt, reduction='mean')
        elif self.task == 'classification':
            return F.cross_entropy(pred, tgt, reduction='mean')

    def train_epoch(self, loader):
        self.model.train()
        tot_loss = 0
        for batch in loader:
            ## send all tensors to device (x1, x2, ..., tgt)
            b = [t.to(self.device) for t in batch]
            self.optimizer.zero_grad()
            pred = self.model(b[:-1]) # x1, x2, ...
            loss = self.compute_loss(pred, b[-1])
            loss.backward()
            self.optimizer.step()
            tot_loss += (loss.item() * b[0].size(0))
        return tot_loss / len(loader.dataset)

    def evaluate(self, loader):
        self.model.eval()
        tot_loss = 0
        with torch.no_grad():
            for batch in loader:
                b = [t.to(self.device) for t in batch]
                pred = self.model(b[:-1])
                loss = self.compute_loss(pred, b[-1])
                tot_loss += (loss.item() * b[0].size(0))
        return tot_loss / len(loader.dataset)
    
    def predict(self, loader):
        self.model.eval()
        preds, tgts = [], []
        with torch.no_grad():
            for batch in loader:
                b = [t.to(self.device) for t in batch]
                preds.append(self.model(b[:-1]))
                tgts.append(b[-1])
        if self.task == 'regression':
            return torch.hstack(preds), torch.hstack(tgts)
        elif self.task == 'classification':
            return torch.vstack(preds), torch.hstack(tgts)

    def train(self, tr_loader, va_loader):
        tr_losses = []
        va_losses = []

        best_va_loss = float('inf')
        no_improvement_epochs = 0
        patience = 5

        for epoch in range(self.max_epochs):
            tr_loss = self.train_epoch(tr_loader)
            va_loss = self.evaluate(va_loader)
            #self.scheduler.step()
            tr_losses.append(tr_loss)
            va_losses.append(va_loss)
            print(f'Epoch: {epoch+1}\tTr Loss: {tr_loss:.3f}\tVal Loss: {va_loss:.3f}')

            if va_loss < best_va_loss:
                best_va_loss = va_loss
                no_improvement_epochs = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                no_improvement_epochs += 1  # Increment counter for no improvement

            if no_improvement_epochs >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs.')
                break

        return tr_losses, va_losses

def eval_metrics(preds, tgts, figures=False, mapping=None, type='regression', classes=None):
    if type == 'regression':
        eval_metrics_regression(preds, tgts, figures, mapping)
    elif type == 'classification':
        eval_metrics_classification(preds, tgts, figures, mapping, classes)

def eval_metrics_classification(preds, tgts, figures=False, mapping=None, classes=None):
    preds, tgts = preds.cpu(), tgts.cpu()
    print(f'Number of unique targets: {tgts.unique().numel()}')
    acc1 = multiclass_accuracy(preds, tgts)
    print(f'Accuracy ({len(classes)}-way): {acc1:.4f}')
    acc2 = torch.eq(preds.argmax(dim=1)>2, tgts>2).float().mean() # NEED more elegant code here!!!
    print(f'Accuracy (speedup/slowdown): {acc2:.4f}')
    ConfusionMatrixDisplay.from_predictions(tgts, torch.argmax(preds,dim=1), display_labels=classes, normalize='true')
    plt.show()

def eval_metrics_regression(preds, tgts, figures=False, mapping=None):
    # calculate the accuracy of the model on each datapoint
    preds, tgts = preds.cpu(), tgts.cpu()
    acc = torch.eq(preds>1, tgts>1).float().mean()
    print(f'Accuracy (speedup/slowdown): {acc:.4f}')
    # loop level analysis (requires loop groups to have at least 3 transformations)
    loop_acc = torch.zeros(3)
    if mapping != None:
        for loop in mapping:
            loop_preds = preds[loop['start']:loop['end']]
            loop_tgts = tgts[loop['start']:loop['end']]            
            _, idx_tgt = torch.topk(loop_tgts, 1)
            for i, k in enumerate([1,2,3]):
                _, idx_pred = torch.topk(loop_preds, k)
                loop_acc[i] += (1 if len(set(idx_tgt.numpy()).intersection(set(idx_pred.numpy()))) > 0 else 0)
        print(f'acc top-1: {loop_acc[0]/len(mapping):.4f} top-2: {loop_acc[1]/len(mapping):.4f} top-3: {loop_acc[2]/len(mapping):.4f}')
    # plot the predicted vs true speedups
    if figures:
        plt.figure()
        plt.scatter(tgts, preds, marker='.', color='gray', alpha=0.5)
        plt.xlabel('True Speedup')
        plt.ylabel('Predicted Speedup')
        plt.plot([0,preds.max()],[0,preds.max()], color='lightgreen', linestyle='dashed')
        plt.plot([1,1],[0,preds.max()], color='orange', linestyle='dashed')
        plt.plot([0,tgts.max()],[1,1], color='orange', linestyle='dashed')
        plt.show()