import numpy as np
import pyautogui
import pydirectinput
import time
import cv2
import ResNet
import torch
from PIL import ImageGrab
from PIL import Image

#光标移到屏幕边缘默认报错，禁掉
pyautogui.FAILSAFE = False

#读取模板
template = cv2.imread('symbol.png',0) # 读灰白的就够了
template2 = cv2.imread('symbol2.png') # 读取RGB图像
Adeptus_Temptation = cv2.imread('Adeptus_Temptation.png',0)
Moon_pie = cv2.imread('Moon_pie.png',0)

w,_ = template.shape[::-1]
h2,w2,_ = template2.shape
w_Adeptus_Temptation,h_Adeptus_Temptation = Adeptus_Temptation.shape[::-1]
w_Moon_pie,h_Moon_pie = Moon_pie.shape[::-1]

# 初始化模型
net = ResNet.net
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载保存的权重
model_path = 'best_model_weights.pth'
net.load_state_dict(torch.load(model_path, map_location=device))

# 设置为评估模式
net.eval()
net.to(device)

def centralize(duration = 0.1):
    for _ in range(100):  # 限制循环次数以避免无限运行
        # 捕捉屏幕
        screen = np.array(ImageGrab.grab())
        gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

        # 模板匹配
        res = cv2.matchTemplate(gray_screen, template, cv2.TM_CCOEFF_NORMED)

        # 获取最佳匹配的位置
        _, _, _, max_loc = cv2.minMaxLoc(res)

        # 确定中心点
        torch_center_x = max_loc[0] + w // 2
        # 计算水平偏移
        screen_center_x = gray_screen.shape[1] // 2
        offset_x = torch_center_x - screen_center_x
        if offset_x == 0:
            break
        # 根据偏移水平调整镜头
        pydirectinput.moveRel(int(offset_x / 2), 0, duration=duration, relative=True)
        
def centralize_2nd():
    for _ in range(100):
        # 捕获屏幕
        screen = np.array(ImageGrab.grab())
        _,width = screen.shape[:2]
        screen = Image.fromarray(screen)  # 转换为PIL图像
        screen = ResNet.transform(screen).unsqueeze(0)  # 应用转换并添加批量维度
        screen = screen.to(device)

        # 使用模型进行预测
        with torch.no_grad():
            prediction = net(screen)
        
        screen_center_x = width // 2      # 960
        offset_x = int(prediction[0][0].item() - screen_center_x)

        print(prediction[0][0].item(),screen_center_x)
        print(offset_x)

        if offset_x == 0:
            break
        # 根据偏移水平移动
        if(offset_x > 0):
            pyautogui.keyDown('d')
            pyautogui.keyUp('d')
        else:
            pyautogui.keyDown('a')
            pyautogui.keyUp('a')
        pydirectinput.moveRel(int(offset_x / 40), 0, duration=0.05, relative=True)
        time.sleep(0.1)

  
def enhancement(threshold = 0.9):

    Adeptus_Temptation_eaten = False
    Moon_pie_eaten           = False

    #打开背包
    pyautogui.keyDown('b')
    pyautogui.keyUp('b')  
    time.sleep(1)
    pyautogui.moveTo(863,50,duration=0.1)
    pyautogui.click()
    time.sleep(0.2)
    pyautogui.moveTo(835,340)
    while True:
        #捕捉屏幕
        screen = np.array(ImageGrab.grab())
        gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        # 仙跳墙模板匹配
        res = cv2.matchTemplate(gray_screen, Adeptus_Temptation, cv2.TM_CCOEFF_NORMED)

        # 获取最佳匹配的位置
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        print("Adeptus_Temptation match result : ")
        print(max_val,max_loc)

        if not Adeptus_Temptation_eaten and max_val >= threshold:       
            # 确定中心点
            torch_center_x = max_loc[0] + w_Adeptus_Temptation // 2
            torch_center_y = max_loc[1] + h_Adeptus_Temptation // 2

            # 吃药
            pyautogui.click(torch_center_x,torch_center_y)
            time.sleep(0.2)
            pyautogui.click(1700,1020)
            time.sleep(0.2)
            pyautogui.moveTo(835,340)
            Adeptus_Temptation_eaten = True

        # 月亮派模板匹配
        res = cv2.matchTemplate(gray_screen, Moon_pie, cv2.TM_CCOEFF_NORMED)

        # 获取最佳匹配的位置
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        print("Moon pie match result : ")
        print(max_val,max_loc)

        if not Moon_pie_eaten and max_val >= threshold:
            # 确定中心点
            torch_center_x = max_loc[0] + w_Moon_pie // 2
            torch_center_y = max_loc[1] + h_Moon_pie // 2

            # 吃药
            pyautogui.click(torch_center_x,torch_center_y)
            time.sleep(0.2)
            pyautogui.click(1700,1020)
            time.sleep(0.2)
            pyautogui.moveTo(835,340)
            Moon_pie_eaten = True
        
        if Moon_pie_eaten and Adeptus_Temptation_eaten:
            pyautogui.keyDown('esc')
            pyautogui.keyUp('esc')
            time.sleep(1)
            break
        time.sleep(0.5)
        for _ in range(20):
            pyautogui.scroll(-200)

for k in range(8): #160树脂，20一次，共8次

    time.sleep(5)

    x = 960
    y = 870
    pyautogui.moveTo(x,y)
    time.sleep(0.1)

    #关闭地脉异常界面
    pyautogui.click()
    time.sleep(0.5)

    centralize()
    time.sleep(0.5)

    #打两把吃一次药
    if k % 2 == 0:
        enhancement()

    #切到钟离，确保行走速度一定
    pyautogui.keyDown('4')
    pyautogui.keyUp('4')
    time.sleep(0.1)

    #走到钥匙前
    pyautogui.keyDown('w')
    time.sleep(18)
    pyautogui.keyUp('w')

    #视角向脚下
    pydirectinput.moveRel(0, 2000, duration=0.2, relative=True)

    #开始挑战
    pyautogui.keyDown('f')
    pyautogui.keyUp('f')

    #钟离 E CD 12s，香菱 E CD 12s，五郎 E CD 10s
    #钟离柱子存在30s

    for i in range(8):

        #钟离开盾
        pyautogui.keyDown('4')
        pyautogui.keyUp('4')
        time.sleep(0.5)
        pyautogui.keyDown('s')
        time.sleep(0.1)
        pyautogui.keyUp('s')
        pyautogui.keyDown('e')
        time.sleep(1)
        pyautogui.keyUp('e')
        time.sleep(1)
        pyautogui.keyDown('w')
        time.sleep(0.3)
        pyautogui.keyUp('w')
        
        #五郎E
        pyautogui.keyDown('3')
        pyautogui.keyUp('3')
        time.sleep(0.1)
        pyautogui.keyDown('e')
        pyautogui.keyUp('e')
        time.sleep(1)
        if i == 3 or i == 4:
            pydirectinput.moveRel(0, -2100, duration=0.2, relative=True)
            time.sleep(0.8)
            pyautogui.keyDown('q')
            pyautogui.keyUp('q')
            time.sleep(1.5)
            pydirectinput.moveRel(0, 2000, duration=0.2, relative=True)
            time.sleep(0.3)

        if i == 1 or i == 5:
            pyautogui.keyDown('2')
            pyautogui.keyUp('2')
            time.sleep(0.1)
            pyautogui.keyDown('q')
            pyautogui.keyUp('q')
            time.sleep(1)

        #林尼射3箭
        pyautogui.keyDown('1')
        pyautogui.keyUp('1')
        time.sleep(0.1)
        pyautogui.keyDown('r')
        pyautogui.keyUp('r')
        time.sleep(0.9)
        for _ in range(3):
            time.sleep(0.9)
            pyautogui.click()
            time.sleep(1.2)
        pyautogui.keyDown('r')
        pyautogui.keyUp('r')

        if i == 5:
            pyautogui.keyDown('q')
            pyautogui.keyUp('q')
            time.sleep(2)
            pyautogui.keyDown('e')
            pyautogui.keyUp('e')
            time.sleep(0.2)
        #隔一轮丢一个E
        if i % 2 == 1:
            pyautogui.keyDown('e')
            pyautogui.keyUp('e')
            time.sleep(1)
            if i != 7:
                pyautogui.keyDown('2')
                pyautogui.keyUp('2')
            time.sleep(0.5)

        time.sleep(0.5)
        if i == 3 or i == 4:
            time.sleep(0.5)
    
    #抬头以方便后续模板匹配
    time.sleep(2)
    pydirectinput.moveRel(0, -2500, duration=0.2, relative=True)
    pyautogui.keyDown('s')
    time.sleep(0.5)
    pyautogui.keyUp('s')
    pyautogui.keyDown('ctrl')
    pyautogui.keyUp('ctrl')
    time.sleep(0.2)
    centralize_2nd()
    pyautogui.keyDown('ctrl')
    pyautogui.keyUp('ctrl')
    #走至石化古树领奖
    pyautogui.keyDown('w')
    time.sleep(9)
    pyautogui.keyUp('w')
    pyautogui.keyDown('a')
    pyautogui.keyUp('a')
    time.sleep(0.1)
    pyautogui.keyDown('d')
    pyautogui.keyUp('d')
    time.sleep(1)
    pyautogui.keyDown('f')

    time.sleep(10)
    #继续挑战
    pyautogui.click(1180,1000)