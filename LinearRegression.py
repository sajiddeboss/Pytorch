##importing libraries
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics

##Custom datasetclass
class CustomDataset:
    def __init__(self,data,targets):
        self.data=data
        self.targets=targets

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        current_sample=self.data[idx,:]
        current_target=self.targets[idx]
        return {
            "x":torch.tensor(current_sample,dtype=torch.float),
            "y":torch.tensor(current_target,dtype=torch.long)
        }

##Creating the dataset
data,targets=make_classification(n_samples=10000)

train_data,test_data,train_targets,test_targets=train_test_split(data,targets,stratify=targets)

train_dataset=CustomDataset(train_data,train_targets)
test_dataset=CustomDataset(test_data,test_targets)

##creating train and test loader
train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=8,num_workers=2)
test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=8,num_workers=2)

##defining our model
model= lambda x,w,b: torch.matmul(x,w)+b

##initializing random weights
W=torch.randn(20,1,requires_grad=True)
b=torch.randn(1,requires_grad=True)
learning_rate=0.0001

if __name__ == '__main__':
    #train_loop
    for epoch in range(20):
        epoch_loss=0
        for data in train_loader:
                xtrain=data["x"]
                ytrain=data["y"]

                if W.grad is not None:
                    W.grad_zero()

                output=model(xtrain,W,b)
                loss=torch.mean((ytrain.view(-1)-output.view(-1))**2)
                epoch_loss=epoch_loss+loss.item()
                loss.backward()

                with torch.no_grad():
                    W = W - learning_rate*W.grad
                    b = b - learning_rate*b.grad
                
                W.requires_grad_(True)
                b.requires_grad_(True)
        print(f"{epoch} epoch : {epoch_loss}")

    #test loop
    labels=[]
    outputs=[]

    with torch.no_grad():
        for data in test_loader:
            xtrain=data["x"]
            ytrain=data["y"]
            
            output=model(xtrain,W,b)
            
            labels.append(ytrain)
            outputs.append(output)

    #evaluating the performance of our model
    print("-"*25)
    print("roc_score : ", metrics.roc_auc_score(torch.cat(labels).view(-1),torch.cat(outputs).view(-1)))


