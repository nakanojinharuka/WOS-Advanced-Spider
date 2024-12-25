# WOS-Advanced-Spider
该爬虫是[WOS爬虫](https://github.com/Dramwig/WOS-spider/tree/main)的二次开发，新版Web of Science其结构之复杂程度需要长时间的调试和测试。
# 使用
## 环境
本地环境：
   - `Python 3.12.7`
   - `selenium 4.15.3`
   - `pandas 2.0.3`
   - `beautifulsoup4 4.12.2`
   - `webdriver (any)`
## 可变参数
```python
url_root = 'https://webofscience...'
papers_need = 100000
file_path = 'result.csv'    
wait_time = 10
pause_time = 3
```
   - `url_root`：自动打开网页（可以手动筛选后进行操作）
   - `papers_need`：自动停止爬取的页数
   - `file_path`：爬取结果表格文件存储路径
   - `wait_time`：等待网页某元素加载的时间，可缩短
   - `pause_time`：每次翻页后的等待时间，平衡速度和稳定性，可缩短
## 步骤
1. 运行程序：自动打开浏览器
2. 手动操作：
   1. 进入Web of Science（新版界面），使用自己的方式进入（没有可以淘宝买账号）
   2. 通过自己想要的方式，选择想要的数据库，输入关键词，点击检索
   3. 点击第一篇文章，进入文章页面
3. 自动爬取：在程序终端输入任意键继续，程序会自动爬取页面信息，并存储在**CSV**文本文件中。
