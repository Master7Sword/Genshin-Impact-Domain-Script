# Genshin-Impact-Domain-Script
## 简介
基于 pyautogui 和 OpenCV 的原神全自动通关秘境的脚本框架

实机演示：https://www.youtube.com/watch?v=7oc3tUT3qsc
## 流程
1. 初始视角回正
2. 自动打开背包吃攻击/防御药
3. 自动行走并开启挑战
4. 自动攻击敌人
5. 自动视角回正
6. 自动领奖并继续挑战
## 使用指南
- 所有需要的函数和方法都在script.py中给出，仅需要根据要打的副本和自身的配队修改输入的参数与操作流程。
- 代码写好后，在对应script.py文件的路径下打开终端，输入pyinstaller --onefile --windowed --distpath=(script.py的目录) script.py，比如我的script.py所在目录为C:\Users\86139\Desktop\code\GenshinScript，那么我需要输入pyinstaller --onefile --windowed --distpath=C:\Users\86139\Desktop\code\GenshinScript\ script.py，**注意**script.py前的空格不能省！！！经过一分钟后会生成一个build文件夹、一个.spec文件以及一个.exe文件，build文件夹和.spec文件在exe文件生成后就没用了，可以删掉。
- 如果需要在原神中运行脚本，**必须**要以管理员身份运行.exe文件
- 游戏内设置中关闭战斗智能镜头与镜头高度自动回正
- 需要自己先点 单人挑战-开始挑战-停在地脉异常界面-启动脚本-回到游戏-开始睡觉（
- ### update
- 模型需要预先训练，API已给出
## 开发日记（2023.11.6）
去年大一刷了一整年的绝缘本，在每天几乎完全重复的刷本流程中萌生出写一个自动化脚本的想法，不过那个时候还不会python。     

开发前想的是：只要我保证每一次的操作是完全一样的，那每次挑战最后的结果也应该是完全一样的。    

然而一测试就发现了巨大的问题，原神中许多东西都是随机的，比如说怪物的攻击顺序、怪物冲脸碰撞角色会产生位移，甚至一开始进入副本时视角会随机发生小范围的转动。这一点点微小的随机对于一个流程完全固定的脚本是致命的，随着时间的推移偏差会迅速累积，最后的结果就是走着走着掉到平台下面去了（悲   

为了处理偏差，我添加了基于OpenCV图像识别的视角纠正模块，核心逻辑为使用cv2.matchTemplate方法找到当前屏幕中与模板匹配度最大的点，并以此找到模板在当前屏幕的中心坐标，然后根据视角中心与模板中心坐标在x轴和y轴上的差值纠正当前的视角。经过多次迭代后屏幕中心会收敛到模板中心，这样保证了我们不会走偏。   

有了OpenCV，我又突发奇想顺手写了个自动吃药函数（其实是怪伤害太高有时候把钟离的盾都干爆了，不得不吃防御药）

## 更新（2023.11.12）
原来使用OpenCV检测的鲁棒性不足，时常会检测到错误的目标。原因在于OpenCV的matchTemplate方法是简单粗暴的像素点匹配，在游戏中遇到视角变化，检测的目标物体发生伸缩变换时会失效。

因此引入ResNet来处理通关后对准领奖位置的图像检测问题。具体而言，先从通关后的副本中截取若干图像，并标注领奖台位置在各个图像中的坐标，打包丢进ResNet网络中训练，当学习率降低到一定值时停止训练，保存Loss最低的网络权重至本地。最后只需要在script.py中加载网络及权重，再将捕获的屏幕输入网络即可得到相应的结果。

经过游戏内测试，网络输出基本符合预期坐标，其偏差在实际运行中完全可以忽略不计。

## 更更新（2023.12.26）  
原神4.3更新了进入副本的位置，角色初始位置直接就在战斗区域前，不需要再走半天楼梯了。这导致这个脚本的代码现在无法直接使用，不过影响不大，自行修改一下刚进入副本后的行走时间即可。

## 一点吐槽
加完ResNet后script.exe文件体积直接增加至2个G，这下成自娱自乐了...以后也许会有优化的方法...

github上传dataset时图片顺序有部分乱了，不知道咋回事

### 更新吐槽（2023.11.26）
yolo多目标检测能力很强，但是在这个脚本里的性能不如脚本，因为有时候目标被（钟离柱子）挡住yolo就没办法检测，但是ResNet还是能够根据全局的像素预测出合理的坐标  
还尝试了一下把模型参数改小，原本的ResNet18一共五层，去掉两层之后就没办法收敛了，loss一直下不来。

### 更更新吐槽（2023.11.30）  
b站演示视频被封了，幽默！！！  
于是又在youtube上传了一个新的  
