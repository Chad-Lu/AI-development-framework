from torch.utils.data import DataLoader
from model import cnn_model
from model.resnet import resnet18,resnet34,resnet152,resnet101
from config import cfg
from datasets import Mango_data
from datasets.transforms import build_transform
from torch.utils.data.sampler import SubsetRandomSampler
from torchsummary import summary
import torch
import numpy as np
import time
import torch.nn as nn

#parpamters setting
valid_size  = 0.1#cfg.DATA.VALIDATION_SIZE
#epochs      = cfg.MODEL.EPOCH
lr          = 0.001#cfg.MODEL.LR
weight_path = cfg.MODEL.OUTPUT_PATH
use_cuda    = cfg.DEVICE.CUDA
gpu_id      = cfg.DEVICE.GPU
epochs = 1000
valid_loss_min = np.Inf # track change in validation loss

#gpu device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('chose ',device)
transforms = build_transform(cfg)

#model = cnn_model()
#model = ResNet18()
model = resnet101(pretrained=True, num_classes=1000, input_channels=3)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 512)
# model.fc = nn.Linear(512, 128)
# model.fc = nn.Linear(128, 3)


if use_cuda:
    torch.cuda.set_device(0)
    model = model.cuda()
print('cuda is available: ',torch.cuda.is_available())
print('GPU: ',torch.cuda.get_device_name(0))
print()

#prepare our data (0.8 for train 0.2 for train_val)
train_data_path = 'C:\\Users\\user\\Desktop\\ML_context\\train_all\\C1-P1_Train\\'  #'C://Users//user//Desktop//hw2//AIMango_sample//sample_image//'
train_file = 'train.csv'

print('laod data start.....')
train_loader  = Mango_data.Mango_data_df(root_dir=train_data_path,csv_file=train_file,transform=transforms)
num_train = len(train_loader)
print('total_images: ',num_train)
print('spilte to 0.9, 0.1 ....')
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))

np.random.shuffle(indices)
train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_set = DataLoader(train_loader, batch_size=8, num_workers=0,sampler=train_sampler)
train_val_set = DataLoader(train_loader, batch_size=8, num_workers=0, sampler=valid_sampler)
print('laod data done.....')

#show our data transforms
for t_imgs, t_label in train_set:
    print ('Size of image(train):', t_imgs.size())  # batch_size*3*224*224
    print ('Type of image(train):', t_imgs.dtype)   # float32
    print ('Size of label(train):', t_label.size())  # batch_size
    print ('Type of label(train):', t_label.dtype)   # int64(long)
    break
''' 
for imgs, label in train_val_set:
    print ('Size of image(train_val):', imgs.size())  # batch_size*3*224*224
    print ('Type of image(train_val):', imgs.dtype)   # float32
    print ('Size of label(train_val):', label.size())  # batch_size
    print ('Type of label(train_val):', label.dtype)   # int64(long)
'''
#print model summary
summary(model.cuda(), (3, 224, 224))
#optimizer setting
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
print('optimizer lr: {}'.format(optimizer.param_groups[0]['lr']))
print('----------------------------------------------------------------')

#start our training
print('total epoch: ',epochs)
for epoch in range(1, epochs+1):
    epoch_start_time = time.time()
    model.train(True)
    train_loss = 0.
    valid_loss = 0.
    accuracy = 0.
    count = 0.
    correct = 0.



    for data, labels in train_set:
        #if use_cuda:
        #    data, target = data.cuda(), labels.cuda()
        #using gpu
        data = data.to(device)
        labels = labels.to(device, dtype = torch.long)

        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, labels)
        #_, predicted = torch.max(output.data, 1) #To Calculation predicted
        #count += len(data)
        #accuracy += (predicted == labels).sum().item()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)

    model.eval()
    for data_v, target in train_val_set:
        #if use_cuda:
        #    data, target = data.cuda(), target.cuda()
        data_v = data_v.to(device)
        target = target.to(device, dtype=torch.long)
        output = model(data_v)
        loss = torch.nn.functional.cross_entropy(output, target)
        _, predicted = torch.max(output.data, 1)  # To Calculation predicted
        count += len(data_v)
        correct += (predicted == target).sum()
        valid_loss += loss.item() * data_v.size(0)

    train_loss /= int(np.floor(len(train_set.dataset) * (1 - valid_size)))
    valid_loss /= int(np.floor(len(train_val_set.dataset) * valid_size))
    accuracy = ((100. * correct) / count)
    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time

    if (epoch >=100)and(train_loss - valid_loss) > 0.1:
        optimizer.param_groups[0]['lr'] /= 10
        print('lreaning rate change to ',optimizer.param_groups[0]['lr'])

    print('Epoch: [{}/{}],ep_time: {:.2f} ,Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch,epochs,per_epoch_ptime, train_loss, valid_loss))
    print("Training_vailation Accuracy: {:.2f}% ({} correct/{} total correct)".format(accuracy, correct, count))

    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
        torch.save(model.state_dict(), 'weight_model.pth') #model name : weight_model.pth
        valid_loss_min = valid_loss


'''
output_dir = "/".join(weight_path.split("/")[:-1])
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
torch.save(model.state_dict(), weight_path)
'''
