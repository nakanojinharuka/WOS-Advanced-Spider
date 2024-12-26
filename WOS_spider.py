# -*- coding: utf-8 -*-
"""
File: WOS_spider.py
Original Author: Dramwig
Optimizer: nakanojinharuka
Email: dramwig@outlook.com, ui_wither@163.com
Date: 2024-12-25
Version: 3.0

Description: This script uses Selenium and BeautifulSoup to scrape detailed paper information from Web of Science (WOS) website.
It navigates through each paper's detail page, extracts key information such as title, citation count, country, journal, etc., 
and saves the collected data into a CSV file.

Please note that this script is intended for educational purposes only, and you should abide by the terms of service and usage policies 
of the Web of Science when using it or any derivative work.

"""
import selenium.webdriver.chrome.service as service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import time, keyboard
import winsound
from webdriver_manager.chrome import ChromeDriverManager

# 解析html
def parse_html(html):
    index = 0
    soup = BeautifulSoup(html, 'html.parser')
    # 创建一个空的字典
    data_dict = {}
    try:
        containers = soup.find_all('div', class_='cdx-two-column-grid-container')
        for container in containers:
            # 在这个容器内找到所有的标签和数据
            labels = container.find_all(class_='cdx-grid-label')
            datas = container.find_all(class_='cdx-grid-data')
            class_authors = datas[0].text.strip()
            label = labels[0].text.strip()
            data_texts = [data.text.strip() for data in datas] # 提取数据列表中的文本
            text = '\n'.join(data_texts) # 将文本连接成一个字符串，使用换行符分隔
            # 存储到字典中
            data_dict['author'] = class_authors
            break
    except Exception as error:
        print("解析容器/获取作者失败")
    try:
        class_title = soup.select('html > body > app-wos > main > div > div > div:nth-child(2) > div > div > div:nth-child(2) > app-input-route > app-full-record-home > div:nth-child(2) > div:nth-child(1) > div:nth-child(1) > app-full-record > div > div:nth-child(1) > div:nth-child(1) > div > div > h2')
        data_dict['title'] = class_title[0].text.strip()
        print('\t'+class_title[0].text.strip())
    except Exception as error:
        print("获取标题失败")
    time.sleep(0.3)
    try:
        journal=''
        conference=''
        #                                        'mat-tooltip-trigger mat-menu-trigger category section-label-data identifiers-link ng-star-inserted'
        j_find = soup.find_all('a', class_='mat-menu-trigger font-size-14 summary-source-title-link source-title-link remove-space no-left-padding section-label-data identifiers-link ng-star-inserted')
        #                                        'mat-menu-trigger font-size-14 summary-source-title-link source-title-link remove-space no-left-padding section-label-data identifiers-link ng-star-inserted'
        c_find = soup.find_all('span', class_='summary-source-title noLink ng-star-inserted')
        if len(j_find) != 0:
            journal = j_find[0].contents[0].text.strip()
        elif len(c_find) != 0:
            conference = c_find[0].text.strip()
        print('\t'+journal if bool(journal) else '\t'+conference)
        data_dict['journal/conference/book'] = journal if bool(journal) else conference
    except Exception as error:
        print("获取所在期刊、书目或会议名失败")
    time.sleep(0.3)
    try:
        publishyear = ''
        earlyaccess = ''
        class_year_publishyear = soup.find_all('span', class_='value section-label-data', id='FullRTa-pubdate')
        class_year_earlyaccess = soup.find_all('span', class_='value section-label-data', id='FullRTa-earlyAccess')
        if len(class_year_publishyear) != 0:
            publishyear = class_year_publishyear[0].text.strip()
        elif len(class_year_earlyaccess) != 0:
            earlyaccess = class_year_earlyaccess[0].text.strip()
        data_dict['publish time'] = publishyear if bool(publishyear) else earlyaccess
    except Exception as error:
        print("获取时间失败")
    time.sleep(0.3)
    try:
        class_citation = soup.find_all('div', class_='citation-position citation-count ng-star-inserted')
        data_dict['citation'] = class_citation[0].contents[0].text.strip()
    except Exception as error:
        try:
            class_citation = soup.find_all('div', class_='citation-two-column')
            data_dict['citation'] = class_citation[0].contents[0].contents[0].text.strip()
        except Exception as error:
            data_dict['citation'] = '0'
    time.sleep(0.3)
    try:
        abstract = soup.find_all('div', class_='abstract--instance abstract-size section-label-data', id='FullRTa-abstract-basic')
        data_dict['abstract'] = abstract[0].text.strip()
    except Exception as error:
        print("摘要未给出")
    time.sleep(0.3)
    try:
        keywords = ''
        class_keywords = soup.find_all('app-full-record-keywords', class_='ng-star-inserted')
        if len(class_keywords[0].text.strip()) == 0:
            try:
                class_keywords_controlled = soup.find_all('app-full-record-categories-classification', class_='ng-star-inserted')
                if len(class_keywords_controlled) != 0:
                    try:
                        keywords = class_keywords_controlled[0].contents[0].contents[17].contents[0].contents[1].text.strip()
                        data_dict['keywords'] = keywords
                    except Exception as error:
                        print("关键词未给出")
                        data_dict['keywords'] = ''
            except Exception as error:
                print("关键词未给出")
                data_dict['keywords'] = ''
        else:
            keywords = class_keywords[0].contents[0].contents[1].text.strip()
            data_dict['keywords'] = keywords
    except Exception as error:
        print("获取关键词失败")
        data_dict['keywords'] = ''
    time.sleep(0.3)
    try:
        input_box = soup.find(class_='wos-input-underline page-box')  # 获取包含输入框的标签
        index = int(input_box['aria-label'].split()[-1].replace(",", ""))
    except Exception as error:
        print("获取数量失败")
    time.sleep(0.3)
    return index, data_dict


if __name__ == "__main__":
    url_root = 'https://www.webofscience.com/wos/alldb/basic-search'
    papers_need = 1000
    file_path = 'urban.csv'
    wait_time = 3
    pause_time = 1
    # 变量
    judge_xpath = '//*[@id="FullRRPTa-useInWOS"]'
    xpath_nextpaper = '/html/body/app-wos/main/div/div/div[2]/div/div/div[2]/app-input-route/app-full-record-home/div[1]/app-page-controls/div/form/div/button[2]'
    df = pd.DataFrame()
    index = 0
    duration = 2000  # 提示音时间 millisecond
    freq = 440  # 提示音Hz
    flag = 0
    # 读取df
    if_read = input("是否读取已有的CSV文件？(y/n)")
    if if_read == 'y':
        df = pd.read_csv(file_path, index_col=0)
        index = int(df.index[-1].split('_')[-1])
        print(f"读取已有的CSV文件，当前行索引为{index},即第{index+1}篇论文")
    # 创建ChromeOptions对象
    chrome_options = webdriver.ChromeOptions()
    # 禁止加载图片等资源
    chrome_options.add_argument("--disable-images")
    chrome_options.add_argument("--disable-plugins")
    chrome_options.add_argument("--disable-extensions")
    # 创建浏览器对象
    chrome_service = service.Service(ChromeDriverManager().install())
    # 创建WebDriver对象时传入ChromeOptions
    driver = webdriver.Chrome(options=chrome_options, service=chrome_service)
    driver.get(url_root) # 打开的页面
    # 手动操作，比如切换标签页等
    input("请手动操作至论文详情页面,完成后按Enter键继续...")
    # 获取获取当前所有窗口的句柄
    window_handles = driver.window_handles
    # 假设新窗口是最后被打开的
    new_window_handle = window_handles[-1]
    # 切换到新窗口
    driver.switch_to.window(new_window_handle)
    # 在新窗口上进行操作，例如获取新窗口的标题
    print("新窗口的标题(请确保页面正确):", driver.title)
    while index <= papers_need:
        print("正在处理第", index+1, "篇论文")
        # 等待页面加载
        try:
            # 或者等待直到某个元素可见
            element = WebDriverWait(driver, wait_time).until(EC.visibility_of_element_located((By.XPATH, judge_xpath)))
        except Exception as e:
            print("等待超时，页面不存在该元素，也可能是页面加载失败")
        time.sleep(pause_time)
        # 解析HTML
        try:
            html = driver.page_source
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, xpath_nextpaper))).click() # 切换到下一页
            index, data = parse_html(html)
            row_index = f'Row_{index}'
            if row_index in df.index:
                df.loc[row_index] = pd.Series(data, name=row_index) # 如果行索引存在，则覆盖对应行的数据
            else:
                df = df._append(pd.Series(data, name=row_index)) # 如果行索引不存在，则追加新的行
            df.to_csv(file_path, index=True)  # 将DataFrame保存为CSV文件,保留行索引作为第一列
            if keyboard.is_pressed('down') or keyboard.is_pressed('0'):  # 按下键盘上的向下键或者数字0 进入中断检查
                t = input("程序中断，输入Enter键继续，其他用于调试...")
                while t != '':
                    html = driver.page_source
                    index,data = parse_html(html)
                    t = input("程序中断，输入Enter键继续...")
            flag = flag - 1
        except Exception as e:
            print("An error occurred:", e)
            if flag <= 0:
                print("尝试重新加载页面...")
                flag = 2
                driver.back()
                time.sleep(pause_time)
            else:
                winsound.Beep(freq, duration)  # 提示音
                input("网页出现问题等待手动解决...")
    # 关闭浏览器
    driver.quit()
