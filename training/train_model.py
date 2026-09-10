"""CPU-friendly MobileNetV3 transfer-learning trainer.
Expected folders: dataset/train/class_name and dataset/validation/class_name.
"""
import argparse, json, copy
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

def loader(folder, transform, batch, shuffle):
    return DataLoader(datasets.ImageFolder(folder, transform), batch_size=batch, shuffle=shuffle, num_workers=0)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='dataset'); p.add_argument('--epochs',type=int,default=8); p.add_argument('--batch-size',type=int,default=8); p.add_argument('--patience',type=int,default=3); args=p.parse_args()
    root=Path(args.data); out=Path('model'); out.mkdir(exist_ok=True)
    train_t=transforms.Compose([transforms.Resize(256),transforms.RandomResizedCrop(224,scale=(.8,1)),transforms.RandomHorizontalFlip(),transforms.RandomRotation(8),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    val_t=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    train=loader(root/'train',train_t,args.batch_size,True); val=loader(root/'validation',val_t,args.batch_size,False)
    classes=train.dataset.classes; (out/'class_names.json').write_text(json.dumps(classes,indent=2))
    device='cuda' if torch.cuda.is_available() else 'cpu'; model=models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT); model.classifier[3]=nn.Linear(model.classifier[3].in_features,len(classes)); model.to(device)
    loss_fn=nn.CrossEntropyLoss(); opt=optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); scheduler=optim.lr_scheduler.ReduceLROnPlateau(opt,patience=1,factor=.3)
    best,stale,best_state=0,0,None
    for epoch in range(1,args.epochs+1):
        model.train(); total=correct=0; running=0
        for x,y in train:
            x,y=x.to(device),y.to(device); opt.zero_grad(); logits=model(x); loss=loss_fn(logits,y); loss.backward(); opt.step(); running+=loss.item()*len(y); correct+=(logits.argmax(1)==y).sum().item(); total+=len(y)
        model.eval(); vtotal=vcorrect=0; vloss=0
        with torch.no_grad():
            for x,y in val:
                x,y=x.to(device),y.to(device); logits=model(x); vloss+=loss_fn(logits,y).item()*len(y); vcorrect+=(logits.argmax(1)==y).sum().item(); vtotal+=len(y)
        acc=vcorrect/max(vtotal,1); scheduler.step(vloss/max(vtotal,1)); print(f'Epoch {epoch}: train loss {running/total:.3f}, train acc {correct/total:.3f}, val loss {vloss/vtotal:.3f}, val acc {acc:.3f}')
        if acc>best: best,stale,best_state=acc,0,copy.deepcopy(model.state_dict()); torch.save({'model_state':best_state,'classes':classes},out/'plant_disease_model.pth')
        else: stale+=1
        if stale>=args.patience: print('Early stopping.'); break
    print(f'Saved best checkpoint; validation accuracy: {best:.3f}. This is not a real-world guarantee.')
if __name__=='__main__': main()
