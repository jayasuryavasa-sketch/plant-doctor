import argparse, json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',default='dataset/test');p.add_argument('--model',default='model/plant_disease_model.pth');args=p.parse_args(); classes=json.loads(Path('model/class_names.json').read_text());t=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]);ds=datasets.ImageFolder(args.data,t);dl=DataLoader(ds,batch_size=8,num_workers=0);m=models.mobilenet_v3_small(weights=None);m.classifier[3]=nn.Linear(m.classifier[3].in_features,len(classes));ck=torch.load(args.model,map_location='cpu',weights_only=False);m.load_state_dict(ck['model_state']);m.eval();actual=[];pred=[]
 with torch.no_grad():
  for x,y in dl: actual+=y.tolist();pred+=m(x).argmax(1).tolist()
 print(classification_report(actual,pred,target_names=classes,zero_division=0));print('Confusion matrix:\n',confusion_matrix(actual,pred))
if __name__=='__main__':main()
