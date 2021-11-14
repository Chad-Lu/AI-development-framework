from model import cnn_model
from model.resnet import resnet18,resnet152,resnet101
from config import cfg
from datasets import Mango_data, Mango_data_text
import torch
from torch.utils.data import DataLoader
from datasets.transforms import build_transform
from torchvision import datasets

model = resnet101(pretrained=False, num_classes=3, input_channels=3)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 3)
# parpamters setting
weight_path = "C:\\Users\\user\\Desktop\\ML_context\\weight_model.pth"
use_cuda = True
gpu_id = 0

# using gpu
if use_cuda:
    torch.cuda.set_device(gpu_id)
    model.cuda()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load weight
weight = torch.load(weight_path)  # load weight path
model.load_state_dict(weight)  # load weight

# prepare test data
# print('laod data start.....')
train_data_path = 'C:\\Users\\user\\Desktop\\ML_context\\test_data\\'
# train_file = 'dev.csv'
transforms = build_transform(cfg)
# test_loader  = Mango_data.Mango_data_text(root_dir=train_data_path,transform=transforms)
test_loader = datasets.ImageFolder(train_data_path, transform=transforms)
print('test_loader:',test_loader)
test_set = DataLoader(test_loader, batch_size=1, num_workers=0, shuffle=False)
print('test_set:',test_set)
# print('laod data done.....')

# evaluation mode
model.eval()

test_loss = 0.
correct = 0
# our csv file list
f_2 = []

# start evaluation
with torch.no_grad():
    for data ,label in test_set:
        # if use_cuda:
        #   data, target = data.cuda(), target.cuda()
        #print('data:',data)
        #print('label:',label)

        data = data.to(device)
        #labels = labels.to(device, dtype=torch.long)
        output = model(data)
        print(output.max(1)[1])

        #loss = torch.nn.functional.cross_entropy(output, labels)
        #test_loss += loss.item() * data.size(0)
        #correct += (output.max(1)[1] == labels).sum()

        if output.max(1)[1] == 0:
            f_2.append('A')
        elif output.max(1)[1] == 1:
            f_2.append('B')
        else:
            f_2.append('C')

        # csvfile.close()

    #test_loss /= len(test_set.dataset)
    #accuracy = 100. * correct / len(test_set.dataset)

    #print('Test Loss: {:.6f}, Test Accuracy: {:.2f}% ({} correct/{} total correct)'.format(test_loss, accuracy, correct,len(test_set.dataset)))

print('f_2 range:',len(f_2))
######################################################33

import pandas as pd

dataframe_1 = pd.DataFrame({'label':f_2})
print(dataframe_1)
dataframe_1.to_csv('dataframe_1.csv', mode='w', header=True, index=False)

dataframe_2 = pd.read_csv('dataframe_1.csv')
dataframe_3 = pd.read_csv('test_submission.csv')
dataframe_3 = dataframe_3.drop(['label'], axis=1)
print(dataframe_3)
pd_final = pd.concat([dataframe_3,dataframe_2],axis=1)#pd.concat(dataframe_2,dataframe_3)
print(pd_final)
pd_final.to_csv('final_submission_v11.csv',header=True,index=False)




'''
import csv

##輸出csv file
filename = 'submission_v1.csv'
with open(filename, 'a', newline='') as csvfile: #非複寫
    files = ['image_id','label']
    outwriter = csv.DictWriter(csvfile, fieldnames=files)
    outwriter.writeheader()
    #outwriter.writerow(['Needed', 'Predicted'])  # 抬頭

    for i in range(1600):
        info = str(f_2[i])
        outwriter.writerow({'label':info})
'''