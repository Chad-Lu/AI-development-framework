from model import cnn_model
from model.resnet import resnet18
from config import cfg
from datasets import Mango_data
import torch
from torch.utils.data import DataLoader
from datasets.transforms import build_transform

model = resnet18(pretrained=False, num_classes=3, input_channels=3)
#parpamters setting
weight_path = "C:\\Users\\user\\Desktop\\ML_context\\weight_model.pth"
use_cuda    = True
gpu_id      =  0

#using gpu
if use_cuda:
    torch.cuda.set_device(gpu_id)
    model.cuda()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#load weight
weight = torch.load(weight_path) #load weight path
model.load_state_dict(weight) #load weight

#prepare test data
#print('laod data start.....')
train_data_path = 'C:\\Users\\user\\Desktop\\ML_context\\ML_HW2_data\\C1-P1_Dev\\'
train_file = 'dev.csv'
transforms = build_transform(cfg)
test_loader  = Mango_data.Mango_data_df(root_dir=train_data_path,csv_file=train_file,transform=transforms)
test_set = DataLoader(test_loader, batch_size=1, num_workers=0,shuffle=False)
#print('laod data done.....')

#evaluation mode
model.eval()

test_loss = 0.
correct = 0
#our csv file list
f_1 =[]
f_2 =[]

#start evaluation
with torch.no_grad():
    for data, labels in test_set:
        #if use_cuda:
        #   data, target = data.cuda(), target.cuda()

        data = data.to(device)
        labels = labels.to(device, dtype=torch.long)
        output = model(data)

        loss = torch.nn.functional.cross_entropy(output, labels)
        test_loss += loss.item() * data.size(0)
        correct += (output.max(1)[1] == labels).sum()

        if output.max(1)[1] == 0:
            f_1.append(1)
            f_2.append('A')
        elif output.max(1)[1] == 1:
            f_1.append(1)
            f_2.append('B')
        else:
            f_1.append(1)
            f_2.append('C')

        #csvfile.close()

        
    test_loss /= len(test_set.dataset)
    accuracy = 100. * correct / len(test_set.dataset)

    print('Test Loss: {:.6f}, Test Accuracy: {:.2f}% ({} correct/{} total correct)'.format(test_loss, accuracy, correct, len(test_set.dataset)))

print(f_1)
print(f_2)

######################################################33
import csv
##輸出csv file
filename='v_submission.csv'
with open(filename,'w', newline='') as csvfile:    
    outwriter = csv.writer(csvfile)
    outwriter.writerow(['Needed','Predicted']) #抬頭
    
    for i in range(800):
        outwriter.writerow([f_1[i],f_2[i]])

