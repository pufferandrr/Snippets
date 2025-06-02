import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.dalta = delta
    def __call__(self,val_loss,model,path):## 调用函数
        print("val_loss={}".format(val_loss))
        score = -val_loss## 损失值越小越好
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss,model,path)
        elif score < self.best_score + self.delta:## 如果当前损失值小于最佳损失值，则更新最佳损失值
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:## 如果当前损失值大于最佳损失值，则更新最佳损失值
            self.best_score = score
            self.save_checkpoint(val_loss,model,path)
            self.counter = 0
    
    def save_checkpoint(self,val_loss,model,path): ## 保存模型断点  
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        torch.save(model.state_dict(),path)
        self.val_loss_min = val_loss