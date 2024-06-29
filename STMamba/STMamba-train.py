import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv

import time
import STMamba_1 #STMamba_1
import spectral
import matplotlib
matplotlib.use('TkAgg')  # 使用Agg后端，这个后端适用于生成图像文件但不显示它们
import matplotlib.pyplot as plt

#import matplotlib.pyplot as plt


# 对高光谱数据 X 应用 PCA 变换
def applyPCA(X, numComponents):

    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX

# 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def padWithZeros(X, margin=2):

    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X

    return newX

# 在每个像素周围提取 patch ，然后创建成符合 keras 处理的格式
def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):

    # 给 X 做 padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))#  精度小降低OA  ,dtype=np.float16
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1

    return patchesData, patchesLabels


def splitTrainTestSet(X, y, testRatio, randomState=345):
    #X:[10249,]
    #print('X: ',X.shape),print('y:  ',y.shape)

    X_test, X_train, y_test, y_train = train_test_split(X,
                                                        y,
                                                        test_size=testRatio,
                                                        random_state=randomState,
                                                        stratify=y)

    return X_train, X_test, y_train, y_test

BATCH_SIZE_TRAIN = 64

def create_data_loader():
    # 地物类别
    # class_num = 16
    # 读入数据
    #X = sio.loadmat('D:\wodewenjian\HSI_dada\Indian_pines_corrected')['indian_pines_corrected']  # X.shape=(145,145,200)
    #y = sio.loadmat('D:\wodewenjian\HSI_dada\Indian_pines_gt')['indian_pines_gt']#[145,145]

    #X = sio.loadmat('D:\wodewenjian\HSI_dada\PaviaU')['paviaU']
    #y = sio.loadmat('D:\wodewenjian\HSI_dada\PaviaU_gt')['paviaU_gt'] #[610,340,103]

    X = sio.loadmat('D:\wodewenjian\HSI_dada\Huanghekou_data')['huanghekou_data']  # (1185, 1342, 285)--> #(6471, 15, 15, 30)
    y = sio.loadmat('D:\wodewenjian\HSI_dada\Huanghekou_gt')['huanghekou_gt']  #

    # 用于测试样本的比例
    test_ratio = 0.95
    # 每个像素周围提取 patch 的尺寸13
    patch_size = 15
    # 使用 PCA 降维，得到主成分的数量
    pca_components = 30

    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)

    print('\n... ... PCA tranformation ... ...')
    X_pca = applyPCA(X, numComponents=pca_components)
    print('Data shape after PCA: ', X_pca.shape)

    print('\n... ... create data cubes ... ...')
    X_pca, y_all = createImageCubes(X_pca, y, windowSize=patch_size)
    print('Data cube X shape: ', X_pca.shape)

    print('\n... ... create train & test data ... ...')
    Xtest, Xtrain, ytest, ytrain = splitTrainTestSet(X_pca, y_all, test_ratio)
    print('Xtrain shape: ', Xtrain.shape)
    print('Xtest  shape: ', Xtest.shape)

    # 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求
    X = X_pca.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtest = Xtest.reshape(-1, patch_size, patch_size, pca_components, 1)
    print('before transpose: Xtrain shape: ', Xtrain.shape)
    print('before transpose: Xtest  shape: ', Xtest.shape)

    # 为了适应 pytorch 结构，数据要做 transpose
    X = X.transpose(0, 4, 3, 1, 2)
    Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
    Xtest = Xtest.transpose(0, 4, 3, 1, 2)
    print('after transpose: Xtrain shape: ', Xtrain.shape)
    print('after transpose: Xtest  shape: ', Xtest.shape)

    # 创建train_loader和 test_loader
    X = TestDS(X, y_all)
    trainset = TrainDS(Xtrain, ytrain)
    testset = TestDS(Xtest, ytest)
    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=True,
                                               num_workers=0,
                                               )
    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=False,
                                               num_workers=0,
                                              )
    all_data_loader = torch.utils.data.DataLoader(dataset=X,
                                                batch_size=BATCH_SIZE_TRAIN,
                                                shuffle=False,
                                                num_workers=0,
                                              )

    return train_loader, test_loader, all_data_loader, y

""" Training dataset"""

class TrainDS(torch.utils.data.Dataset):

    def __init__(self, Xtrain, ytrain):

        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):

        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self):

        # 返回文件数据的数目
        return self.len

""" Testing dataset"""

class TestDS(torch.utils.data.Dataset):

    def __init__(self, Xtest, ytest):

        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):

        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):

        # 返回文件数据的数目
        return self.len

def train(train_loader, epochs):

    # 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    # 网络放到GPU上
    net = STMamba_1.STMamba().to(device)
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 初始化优化器
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # 开始训练
    total_loss = 0
    for epoch in range(epochs):#
        net.train()
        for i, (data, target) in enumerate(train_loader):#[128, 1, 30, 9, 9]) target: torch.Size([128]
            data, target = data.to(device), target.to(device)
            # 正向传播 +　反向传播 + 优化
            # 通过输入得到预测的输出
            outputs = net(data)
            # 计算损失函数
            loss = criterion(outputs, target)
            # 优化器梯度归零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' % (epoch + 1,
                                                                         total_loss / (epoch + 1),
                                                                         loss.item()))

        if ((epoch + 1) % 5 == 0 and (epoch + 1) >= 60):
            y_pred_test, y_test = test(device, net, test_loader)
            classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)
            #print('oa: ', oa), print('aa: ', aa), print('kappa: ', kappa)
            if(oa>95.5):
               print('classification: ', classification)

    print('Finished Training')

    return net, device

def test(device, net, test_loader):
    count = 0
    # 模型测试
    net.eval()
    y_pred_test = 0
    y_test = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))

    return y_pred_test, y_test

def AA_andEachClassAccuracy(confusion_matrix):

    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def acc_reports(y_test, y_pred_test):
    '''
    target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
        , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                    'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                    'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                    'Stone-Steel-Towers']
    '''
    '''
    target_names = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets',
                    'Bare Soil', 'Bitumen', 'Self-Blocking Bricks', 'Shadows']
    '''

    target_names = ['Salt marsh', 'Acquaculture', 'Mud flat', 'Rice'
        , 'Aquatic vegetation', 'Seep sea', 'Freshwater herbaceous marsh',
                    'Shallow sea', 'Reed', 'Pond', 'Build up',
                    'Suaeda salsa', 'flood plain', 'River', 'Soybean',
                    'Broomcorn', 'Maize', 'Locust', 'Spartina',
                    'Tamarix', 'Intertidal saltwater']
    ''''''
    classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return classification, oa*100, confusion, each_acc*100, aa*100, kappa*100

if __name__ == '__main__':

    train_loader, test_loader, all_data_loader, y_all= create_data_loader()
    tic1 = time.perf_counter()
    net, device = train(train_loader, epochs=100)

    # 只保存模型参数
    #torch.save(net.state_dict(), 'cls_params/SSFTTnet_params.pth')
    toc1 = time.perf_counter()
    tic2 = time.perf_counter()
    y_pred_test, y_test = test(device, net, test_loader)
    toc2 = time.perf_counter()
    # 评价指标
    classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)
    classification = str(classification)
    Training_Time = toc1 - tic1
    Test_time = toc2 - tic2
    print(classification)
    #print('y_pred_test: ',y_pred_test),print('y_pred_test_size: ',y_pred_test.size)

    #print('y_test: ',y_test),print('y_test_size: ',y_test.size)
    print('oa: ', oa), print('aa: ', aa), print('kappa: ', kappa)

    #get_cls_map.get_cls_map(net, device, all_data_loader, y_all)



# 显示结果
# load the original image
# 导入数据

#X=sio.loadmat('D:\wodewenjian\HSI_dada\Indian_pines_corrected')['indian_pines_corrected']                                 #X.shape=(145,145,200)
# y=sio.loadmat('D:\wodewenjian\HSI_dada\Indian_pines_gt')['indian_pines_gt']

#X = sio.loadmat('D:\wodewenjian\HSI_dada\PaviaU')['paviaU']
#y = sio.loadmat('D:\wodewenjian\HSI_dada\PaviaU_gt')['paviaU_gt'] #[610,340,103]

X = sio.loadmat('D:\wodewenjian\HSI_dada\Huanghekou_data')['huanghekou_data']#(1185, 1342, 285)--> #(6471, 15, 15, 30)
y = sio.loadmat('D:\wodewenjian\HSI_dada\Huanghekou_gt')['huanghekou_gt']  #

height = y.shape[0]
width = y.shape[1]
patch_size = 15
pca_components = 30
X = applyPCA(X, numComponents=pca_components)
X = padWithZeros(X, patch_size // 2)

# 逐像素预测类别
outputs = np.zeros((height, width))
for i in range(height):
    for j in range(width):
        if int(y[i, j]) == 0:
            continue
        else:
            image_patch = X[i:i + patch_size, j:j + patch_size, :]
            image_patch = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1], image_patch.shape[2], 1)
            X_test_image = torch.FloatTensor(image_patch.transpose(0, 4, 3, 1, 2)).to(device)
            #X_test_image.shape:[1, 1, 30, 25, 25]
            prediction = net(X_test_image)   #将真实数据X代入模型net得到预测值，然后进行画图 ！
            #print('prediction.shape: ',prediction.shape)
            prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
            outputs[i][j] = prediction + 1
    if i % 20 == 0:
        print('... ... row ', i, ' handling ... ...')


##显示分类结果，阴间

#outputs.shape=145*145
predict_image = spectral.imshow(classes=outputs.astype(int), figsize=(9, 9))
plt.pause(60)



