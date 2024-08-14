import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import tqdm
import cv2
import numpy as np

# 加载数据集
transform = transforms.Compose([
    transforms.Resize((120, 80)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(root='/mnt/liying.cibr.ac.cn_Data_Temp/time_tag_frame', transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)


transform2 = transforms.Compose([
    transforms.Resize((60, 40)),
    transforms.ToTensor()
])

train_dataset2 = datasets.ImageFolder(root='/mnt/liying.cibr.ac.cn_Data_Temp/time_tag_frame_120', transform=transform2)
train_loader2 = DataLoader(dataset=train_dataset2, batch_size=64, shuffle=True)


# 初始化模型
if True:
    model_name= 'resnet18'
    model = models.resnet18(pretrained=False, num_classes=10)
else:
    model_name = 'shufflenet'
    model = models.shufflenet_v2_x0_5(pretrained=False, num_classes=10)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


num_epochs = 20
for epoch in tqdm.trange(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')


    for i, (images, labels) in enumerate(train_loader2):
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

#%%
correct = 0
total = 0
with torch.no_grad():
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    for images, labels in train_loader2:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the train images: {100 * correct / total}%')

#%%
from torch.nn.parameter import Parameter
class NormalizedModel(torch.nn.Module):
    def __init__(self, patchInnerHW, module):
        super().__init__()
        self.module = module
        self.patchH, self.patchW = patchInnerHW
        self.unfold = nn.Unfold(kernel_size=patchInnerHW, stride=patchInnerHW, padding=0)
        self.f = torch.nn.Softmax(dim=1)
        self.std_value = Parameter(torch.Tensor([255.0]).to(device), requires_grad=False)
        
    def forward(self, x): #x: NCHW rgb 0-255
        x = x/self.std_value
        patches = self.unfold(x[None,None])
        patches2 = patches.view(1, self.patchH, self.patchW, -1)
        patches3 = torch.permute(patches2, (3,0,1,2))

        imgNCHW = patches3.repeat(1, 3, 1, 1)
        outputs = self.module(imgNCHW)
        outputs = self.f(outputs)
        max_values, max_indices = torch.max(outputs, dim=1)
        outcode = max_indices[0]*100 + max_indices[1]*10 + max_indices[2]
        return outcode, max_values

#%%
if False:
    model_new = NormalizedModel((120, 80), model)

    imgs = ['/mnt/liying.cibr.ac.cn_Data_Temp/time_tag_frame/1/80.jpg',
            '/mnt/liying.cibr.ac.cn_Data_Temp/time_tag_frame/6/80.jpg',
            '/mnt/liying.cibr.ac.cn_Data_Temp/time_tag_frame/7/86.jpg']
else:
    model_new = NormalizedModel((60, 40), model)
    imgs = ['/mnt/liying.cibr.ac.cn_Data_Temp/time_tag_frame_120/1/a86.jpg',
            '/mnt/liying.cibr.ac.cn_Data_Temp/time_tag_frame_120/6/a86.jpg',
            '/mnt/liying.cibr.ac.cn_Data_Temp/time_tag_frame_120/7/a82.jpg']
    model_name = model_name + '_120'
img_mats = [ cv2.imread(img) for img in imgs]
img_concat = np.concatenate(img_mats, axis=1)[:,:,0]

img_tensor = torch.from_numpy(img_concat).float().to(device) #/ 255.0
out_class, out_value = model_new(img_tensor)
assert out_class.item() == 167 and torch.all(out_value > 0.9)

# 导出模型为ONNX格式
model_new.eval()
# dummy_input_new = torch.randn(120, 240).to(device)
torch.onnx.export(model_new, img_tensor, f"{model_name}_norm.onnx", 
                  input_names = ["input_tensor"], 
                  output_names=["a_timecode", "b_confidence"], opset_version=11)

print(f'trtexec --onnx={model_name}_norm.onnx --fp16 --saveEngine={model_name}_norm.engine')
