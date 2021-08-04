# 使用CNN進行貓臉偵測與裝飾
## 摘要
使用Kaggle上一份有近萬張貓咪圖片與其五官座標的資料集，進行模型訓練，可預測貓咪的五官座標並為貓咪做裝飾
## 資料集介紹
總共有9997張圖片，以及圖片相對應的9個五官特徵點座標(左眼、右眼、鼻子、左耳左根部、左耳頂點、左耳右根部、右耳左根部、右耳頂點、右耳右根部的xy座標)共18個值。
![image](https://i.imgur.com/eMSrAw1.png)  
![image](https://i.imgur.com/e3LyRMG.png)  
## 模型訓練-臉部擷取
由於將整張照片直接輸入模型預測五官座標較為困難，因此我們先訓練一個可裁剪臉部的模型(MobileNet)，使用圖片的左上以及右下的x與y座標表示貓咪臉部的位置(4個輸出)。  
下圖為臉部標示範例。(藍色框為實際位置，白色框為預測位置)  
![image](https://i.imgur.com/pcd9SWt.png)  
## 模型訓練-五官預測
將上階段得到的圖片大小調整為224x224，貓咪五官座標的模型輸出為9個五官特徵點(左眼、右眼、鼻子、左耳左根部、左耳頂點、左耳右根部、右耳左根部、右耳
頂點、右耳右根部的xy座標)，共18個輸出。  
我們針對Resnet50與mobileNet做訓練與各種調整，由於mobileNet有更快的預測速度，準確度也與Resnet50差不多，因此最後選擇mobileNet作為我們的模型、loss function為RMSE、optimizer選用Adam。  
下圖為五官標示範例。(藍色點為實際位置，綠色點為預測位置)  
![image](https://i.imgur.com/n8I6pXd.png)  
在模型的驗證上，我們定義預測偏差小於原圖大小的1.5% 即為正確預測，下圖為各座標的準確度。  
![image](https://i.imgur.com/dgrYCmU.png)  
可以看出來我們的模型在眼睛、鼻子、耳朵頂端有較高的準確率，在耳朵根部的準確率較低。  
我們推測原因是因為眼睛、鼻子、耳朵頂端的特徵較為明顯，耳朵根部沒有明顯的特徵所導致。  
## 加入裝飾
利用五官的座標求出適當的裝飾大小、位置以及角度，並將欲裝飾的物品做適當的旋轉後放到正確座標上。  
我們使用PtQt5實作GUI裝飾貓咪的小程式，以下為影片demo連結與程式畫面截圖。   
影片連結： https://www.youtube.com/watch?v=5YOTvhEWpVo&feature=youtu.be  
![image](https://i.imgur.com/H6s8tUQ.png)  
![image](https://i.imgur.com/E2WbHiK.png)  
![image](https://i.imgur.com/p7NMzYL.png)  
![image](https://i.imgur.com/Z5Yhcyi.png)  
![image](https://i.imgur.com/VCkhh09.png)  
