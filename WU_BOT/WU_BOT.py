
from PIL import Image
from pyautogui import *
import pyautogui
from matplotlib import pyplot as plt
import time
import keyboard
import random
import win32api, win32con, win32ui, win32gui
import numpy as np
import math
import PIL
import threading, time
from datetime import datetime, timedelta
import cv2
import pytesseract
import sys
import skimage.io
import skimage.color
import torch
from sewar.full_ref import ssim
from windowcapture import WindowCapture

pytesseract.pytesseract.tesseract_cmd = r'D:\\Programs\\Tesseract-OCR\\tesseract.exe'

model = torch.hub.load('ultralytics/yolov5', 'custom', path='path/to/best3.pt')
model.conf = 0.85
model.cuda()

wincap = WindowCapture('WarUniverse')
window_x, window_y, window_w, window_h = wincap.get_screen_dimensions()
window_center = int((window_x+window_w)/2), int((window_y+window_h)/2)

global roaming
global collecting
roaming = True
collecting = False


def ImgToStr(image):
    start = time.time()
    img1 = cv2.imread(image)
    img2 = cv2.imread(image)
    text1 = pytesseract.image_to_string(img1)
    text2 = pytesseract.image_to_string(img2)
    if text1==text2:
        print(text1 + ', ' + text2 +' equals')
    else:
        print(text1 + ', ' + text2 +' differs')
    end = time.time()
    print(end-start)
    input()

def click(x,y):
    win32api.SetCursorPos((x,y))
    time.sleep(0.001)
    win32api.SetCursorPos((x,y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0)
    time.sleep(0.1)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)

def getPositionOnMap(x,y,w,h):
    index=-1
    capture=testArrays.get_screenshot(x,y,w,h)
    capture=cv2.cvtColor(capture, cv2.COLOR_BGR2GRAY)
    capture=np.reshape(capture, -1)
    index=np.where(capture>=240)
    position=index[0][0]
    if position!=-1:
        _y=math.floor(position/w)
        _x=position-(_y*w)
        return _x,_y

def getPositionOnMapCv2(x_map,y_map,width_map,height_map):
    flag = 0
    picRGB = pyautogui.screenshot(region=(x_map, y_map, width_map, height_map))
    imgArr = np.asarray(picRGB)
    image = cv2.cvtColor(imgArr, cv2.COLOR_BGR2GRAY)
    width, height=image.shape
    image[image < 128,128,128] = 0
    for x in range(0, width):
        for y in range(0, height):
            r,g,b=image.getpixel((x,y))
            if r>=220 and g>=220 and b>=220:
                flag=1
                x_map,y_map=x,y
                break
        if flag==1:
            break
    if flag==1:
        return x_map,y_map
    else:
        return -1,-1

def calculateDistance(x,y):
    distance = math.sqrt((x-1280)**2+(y-700)**2)
    return distance/(480*0.5)

def corrigateClick(c,d):
    flag = 0
    pic = pyautogui.screenshot(region=(c-100,d-100,200,200))

    width, height = pic.size

    for x in range(0, width, 25):
        for y in range(0, height, 25):

            r, g, b = pic.getpixel((x, y))

            if b >= 240 and r <= 180 and g >= 240:
                flag = 1
                
                sleepTime=calculateDistance(c-100+x,d-100+y)
                click(c-100+x,d-100+y)
                time.sleep(sleepTime)
                break

        if flag == 1:
                break


def collectBox():
    global collecting
    global roaming
    while keyboard.is_pressed('q')==False:
        if not collecting:
            pass
        else:
            screenshot = wincap.get_screenshot()
            screenshot=cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)

            results=model(screenshot, size=416)
            boxes=results.pandas().xyxy[0].to_dict(orient="records")
            closest = math.inf
            if len(boxes) > 0:
                collecting=True
                roaming=False


                coordinates = 0,0
                for box in boxes:
                    x1 = int(box['xmin'])
                    y1 = int(box['ymin'])
                    x2 = int(box['xmax'])
                    y2 = int(box['ymax'])
                    center = [ window_x + int((x1+x2)/2), window_y + int((y1+y2)/2)]
                    distance = math.dist(window_center, center)
                    if closest > distance:
                        closest = distance
                        coordinates = center
                click(coordinates[0], coordinates[1])
                time.sleep(3)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()

def click_on_map():
    global collecting
    global roaming

    map_x = 20
    map_y = window_h - 55 + 40
    map_w = 91
    map_h = 55

    x1=window_x
    x2=window_x+91
    y1=window_y+window_h-55
    y2=window_y+window_h
    
    scrnshot1=wincap.get_screenshot_minimap(map_x,map_y,map_w,map_h)
    even=False
    sleep_time=0
    if collecting:
        sleep_time=0.2
    elif roaming:
        sleep_time=1
    else:
        clickMapThread.stop()

    while keyboard.is_pressed('q')==False:
        if roaming:
            pass
        elif collecting:
            pass

        sleep(sleep_time)

        if even:
            scrnshot1=wincap.get_screenshot_minimap(map_x,map_y,map_w,map_h)
            even=(not even)
        else:
            scrnshot2=wincap.get_screenshot_minimap(map_x,map_y,map_w,map_h)
            even=(not even)

        similarity = ssim(scrnshot1,scrnshot2)
        print(similarity[0])

        if collecting & similarity[0] > 0.97:
            collecting=False
            collectBoxThread.start()

        if similarity[0] > 0.97:
            x_random = int(np.random.uniform(x1, x2))
            y_random = int(np.random.uniform(y1, y2))
            click(x_random, y_random)
  
def recognizeMap():
    minimap = pyautogui.locateOnScreen('minimap3.png',grayscale=True,confidence=0.4)
    while minimap==None:
        minimap = pyautogui.locateOnScreen('minimap3.png',grayscale=True,confidence=0.4)
    image = PIL.Image.open("minimap3.png")
    width, height = image.size
    x, y = pyautogui.center(minimap)
    return int(x-(width/2)*0.73),int(y-(height/2)*0.8),int(width*0.73),int(height*0.8)

def clickMap():
    x,y,width,height = recognizeMap()
    time.sleep(2)
    loop_time = time.time()
    while keyboard.is_pressed('q')==False:
        x_pos, y_pos = getPositionOnMap(x,y,width,height)
        print('FPS {}'.format(1 / (time.time() - loop_time)))

        time.sleep(0.1)
        if x_pos>=0:
            time.sleep(0.5)
            new_x_pos, new_y_pos = getPositionOnMap(x,y,width,height)
            if 3 >= math.sqrt((x_pos-new_x_pos)**2+(y_pos-new_y_pos)**2):
                x_random = int(np.random.uniform(x, x+width))
                y_random = int(np.random.uniform(y, y+height))
                click(x_random, y_random)
        
        loop_time = time.time()
        if cv2.waitKey(1) == ord('q'):
            break
        '''elif cv2.waitKey(1) == ord('w'):
            cv2.waitKey(0) == ord('w')'''

def countlower1(v, w):
    """Return the number of pairs i, j such that v[i] < w[j].

    >>> countlower1(list(range(0, 200, 2)), list(range(40, 140)))
    4500

    """
    return sum(x < y for x in v for y in w)

def countlower2(v, w):
    """Return the number of pairs i, j such that v[i] < w[j].

    >>> countlower2(np.arange(0, 2000, 2), np.arange(400, 1400))
    450000

    """
    grid = np.meshgrid(v, w, sparse=True)
    return np.sum(grid[0] < grid[1])


'''length, height, depth = capture.shape
capture=capture.reshape((length * height * depth, 1))'''

def collectBox_TEST(m,n):
    flag = 0
    pic = pyautogui.screenshot(region=(m-25,n-25,m+25,n+25))

    width, height = pic.size

    for x in range(0, width):
        for y in range(0, height):

            r, g, b = pic.getpixel((x, y))

            if  (r >= 245 and 226 <= g <= 236 and 198 <= b <= 208) or ( 169 <= r <= 189 and 148 <= g <= 168 and 129 <= b <= 149):
                flag = 1
                click(x+m-25,y+n-25)
                time.sleep(2)
                break

        if flag == 1:
            break
    return flag

def corrigate(x,y):
    win32api.SetCursorPos((x,y))
    box = pyautogui.locateOnScreen('box.png',grayscale=True,confidence=0.6)
    pyautogui.write
    if box != None:
        x, y = pyautogui.center(box)
        click(x,y)

    time.sleep(np.random.uniform(0.8,1.2))

clickMapThread = threading.Thread(target=click_on_map)

collectBoxThread = threading.Thread(target=collectBox)

sleep(2)

def Bot_logic():
    collectBoxThread.start()

collectBoxThread.start()
clickMapThread.start()