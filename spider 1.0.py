#  动态网页:只有不断滑动才能获得下一页 没有把所有图片链接都放到网页中 动态存储
import selenium  # 模拟浏览器登录
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains  # 模拟鼠标操作
from selenium.webdriver.common.keys import Keys  # 模拟键盘操作
import time  # 下载速度过快 防止被抓
import requests
import os
import pinyin
import re

while True:
    name = input("请输入想要下载图片名")
    name_pinyin = pinyin.get(name, format='strip', )
    num = int(input("请输入下载数目"))
    if name == 'q':
        print('进程结束')
        break

    b = webdriver.Chrome()
    b.get("https://image.baidu.com/")
    search_box = b.find_element_by_id('kw')
    search_box.send_keys(name)
    search_box.send_keys(Keys.ENTER)  # 键盘模拟回车键操作
    time.sleep(3)
    # 打开第一张图片 在此页面点击左右切换图片
    ele_1 = b.find_element_by_xpath('//*[@id="imgid"]/div[1]/ul/li[7]/div[1]/div[2]/a/img')  # elements 返回是列表
    ele_1.click()
    time.sleep(2)
    latest_window = b.window_handles[-1]
    b.close()  # 关闭窗口，保证浏览器只有一个窗口
    b.switch_to.window(latest_window)

    x = 1
    for i in range(1, num + 1):
        ele_2 = b.find_element_by_xpath('//*[@id="currentImg"]')  # 在第二个窗口的图片的位置
        url = ele_2.get_attribute('src')  # 获取属性
        r = requests.get(
            url)  # requests是用获得的链接下载图片     url == re.findall('\S*?jpg|\S*?JPG', , re.S)  r.status_code == 200
        if r.status_code == 200:  # 如果网页状态码返回200，表示网页能正常访问，如果网页请求出错就不执行接下来的操作
            path_file = "E:\Chinese Medicine\%s" % name_pinyin  # 只用一个%就可以  后边不用写.jpg 否则会在name下在创建一个文件夹
            if not os.path.exists(path_file):  # 如果不存在路径，则创建这个路径，关键函数就在这两行，其他可以改变
                os.makedirs(path_file)
            path_pic = "E:\Chinese Medicine\%s\%s %d.jpg" % (name_pinyin, name_pinyin, x)
            with open(path_pic, 'wb') as f:
                f.write(r.content)
                time.sleep(1)
                f.close()
                print("%s %d.jpg 爬取成功" % (name, x))
                x += 1
            ele_3 = b.find_element_by_xpath('//*[@id="container"]/span[2]')  # 找到下一张图片的箭头
            ele_3.click()
            time.sleep(0.5)
        else:
            ele_3 = b.find_element_by_xpath('//*[@id="container"]/span[2]')
            ele_3.click()
            time.sleep(1)
            continue
    b.close()
