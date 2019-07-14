import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
import pandas as pd
import pickle
import glob
from skimage import io,transform
# 创建模型输出文件夹
output_dir='output'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
# 数据准备 - 读取图片并贴标签
img_dir = './flower_photos/'
images=[]
labels=[]
isFirst=0
w=100
h=100
cate=[img_dir+x for x in os.listdir(img_dir) if os.path.isdir(img_dir+x)]
imgs=[]
labels=[]
for idx,folder in enumerate(cate):
    for im in glob.glob(folder+'/*.jpg'):
        #print('reading the images:%s'%(im))
        img=cv2.imread(str(im),cv2.IMREAD_COLOR)
        #img=transform.resize(img,(w,h,3))
        imgs.append(img)
        if isFirst<1:
            print(img)
            isFirst = 1
        label = ((str(im).split('/')[2]).split('\\')[0])
        labels.append(label)
 

print(labels)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
print(y)

train_idx,test_idx = train_test_split(range(len(y)),test_size=0.2,stratify = y, random_state = 1234)    # 返回拆分后的索引
train_y = y[train_idx]
test_y = y[test_idx]
#print(train_y)
#print(test_y)

# 计算RGB颜色直方图

def transform(img):
    hist = cv2.calcHist([img],[0,1,2],None,[8]*3,[0,256]*3)
    return hist.ravel()

x_rgb = np.row_stack([transform(img) for img in imgs])

train_x = x_rgb[train_idx,:]
test_x = x_rgb[test_idx,:]
print(train_x)

model_rgb_rf = RandomForestClassifier(n_estimators =1000, max_depth =30, random_state=1234, oob_score=True) # 1234随机初始化的种子
model_rgb_rf.fit(train_x,train_y)
print("finish")
def save_model(model,label_encoder,output_file):
    try:
        with open(output_file,'wb') as outfile:
            pickle.dump({
                'model':model,
                'label_encoder':label_encoder
            },outfile)
        return True
    except:
        return False
save_model(model_rgb_rf,label_encoder,os.path.join(output_dir,'model_rgb_rf.pkl'))
print("save")

def eval_model(y_true,y_pred,labels):
    
    P,r,f1,s =precision_recall_fscore_support(y_true,y_pred)
    tot_P = np.average(P,weights =s)
    tot_r = np.average(r,weights =s)
    tot_f1 = np.average(f1,weights =s)
    tot_s = np.sum(s)
    res1 = pd.DataFrame({
        'Label':labels,
        'Precision':P,
        'Reacll':r,
        'F1':f1,
        'Support':s
    })
    res2 = pd.DataFrame({
        'Label':['总体'],
        'Precision':[tot_P],
        'Recall':[tot_r],
        'F1':[tot_f1],
        'Support':[tot_s]
    })
    res2.index=[999]
    res = pd.concat([res1,res2])
    conf_mat = pd.DataFrame(confusion_matrix(y_true,y_pred),columns=labels,index=labels)
    return conf_mat,res[['Label','Precision','Recall','F1','Support']]

y_pred_rgb_rf = model_rgb_rf.predict(test_x)
conf_mat_lab_rf,evalues_rf = eval_model(test_y,y_pred_rgb_rf,label_encoder.classes_)
print("evalues")
print(conf_mat_lab_rf)
print(evalues_rf)