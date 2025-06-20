from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
from openpyxl import Workbook
import sys
import pandas as pd
import os
import wget
import urllib.request



# print("enter")

# ml_small = pd.read_csv('/Users/caoziyi/Desktop/短视频推荐/MovieLens/ml-latest-small/links.csv')
# movieid = ml_small['movieId'].tolist()
# not_done = []
# for i in movieid:
#     if '{}.mp4'.format(i) not in os.listdir('/Users/caoziyi/Desktop/短视频推荐/MovieLens/trailer'):
#         not_done.append(i)

# print('done')
# youtube_file = '/Users/caoziyi/Desktop/短视频推荐/MovieLens/ml-youtube.csv'
# y_f = pd.read_csv(youtube_file)
ml_small = pd.read_csv('/Users/caoziyi/Desktop/短视频推荐/MovieLens/ml-latest-small/movies.csv')

not_done = [579,722,791,824,889,1107,1856,2156,2503,2897,3281,3303,3429,3626,4278,4883,5239,5736,5922,6020,6402,6553,6853,6965,7264,7566,8987,
25887,26183,26195,26236,26631,26712,26797,26900,27416,27716,31692,33779,40578,40597,44777,48649,49917,52767,53883,59143,62208,62834,65225,68536,
69685,72921,73501,74727,78160,78544,79333,79536,80124,83969,84512,84553,84844,86504,86668,87834,91261,92535,96518,96520,97194,97285,97757,98697,
102602,102742,107159,116169,120761,120827,121374,122260,123200,123545,127052,127098,128366,129514,130978,131749,133712,135198,136445,136469,
140265,140561,141810,141818,141820,141830,141836,141844,142961,144478,145745,145994,146028,147282,147286,147300,147326,147328,147330,147382,
147936,148671,151763,151769,152173,153070,153386,153408,155589,157369,158398,159069,161032,163072,165959,166183,167296,168144,170777,171695,
171917,172577,172583,172585,172587,172589,172637,172793,172825,172875,172909,173355,173535,174403,174551,175397,175401,175431,175781,176621,
179427,180777,182293,183301,184245,193579]





# option = webdriver.ChromeOptions()
# option.add_experimental_option('excludeSwitches', ['enable-automation'])
# option.add_argument('user-agent="Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36"')
# driver = webdriver.Chrome(options=option)
# driver.maximize_window()


# driver.switch_to.default_content()

for nd in not_done:
	search = '+'.join(ml_small[ml_small['movieId'] == nd]['title'].values[0].split()+['trailer'])
	print(str(nd) + '     ' + ml_small[ml_small['movieId'] == nd]['title'].values[0])
	# driver.get('https://www.google.com/search?q={}'.format(search))
	# 在搜索框中输入搜索关键词
	# time.sleep(10)
	# print(nd)
	# print(driver.current_url+'\n')
	# print(nd)
	# url = driver.find_elements(By.XPATH,'//*[@id="rso"]/div[1]/div/div/div[2]/div[2]/div[1]/div/div/div/div')
	# if len(url) > 0:
	# 	print(url[0].get_attribute("data-surl"))
	# 	# driver.get(url[0].get_attribute("data-surl"))


	# url = driver.find_elements(By.XPATH,'//*[@id="rso"]/div[1]/div/div/div/div/div/div[1]/div[1]/div/div/span/a')
	# if len(url) > 0:
	# 	print(url[0].get_attribute("href"))
		# driver.get(url[0].get_attribute("href"))
	

# for nd in not_done:

# 	driver.get('https://movielens.org/movies/{}'.format(nd))
# 	time.sleep(2)
# 	driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")


# 	image = driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/div[4]/div[2]/h2')
# 	frame = []
# 	trailer = []
# 	if len(image) >0:
# 		if 'Images' in image[0].text :
# 			frame.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/div[4]/div[2]/div/div[1]/div/div/a/img'))
# 			frame.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/div[4]/div[2]/div/div[2]/div/div/a/img'))
# 			frame.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/div[4]/div[2]/div/div[3]/div/div/a/img'))
# 			frame.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/div[4]/div[2]/div/div[4]/div/div/a/img'))
# 			frame.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/div[4]/div[2]/div/div[5]/div/div/a/img'))
# 			frame.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/div[4]/div[2]/div/div[6]/div/div/a/img'))
# 			frame.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/div[4]/div[2]/div/div[7]/div/div/a/img'))
# 			frame.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/div[4]/div[2]/div/div[8]/div/div/a/img'))
# 	trailer = driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/div[4]/div[3]/h2')
# 											# //*[@id="main-container"]/ui-view/movie-page/div/div[3]/div[3]/h2
# 	if len(trailer) >0:
# 		if 'Trailers' in trailer[0].text:
# 			print(nd)
# 			time.sleep(10)
# 	trailer = []
# 	image = driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/div[3]/div[2]/h2')
# 	if len(image) >0:
# 		if 'Images' in image[0].text:
# 			frame.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/div[3]/div[2]/div/div[1]/div/div/a/img'))
# 			frame.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/div[3]/div[2]/div/div[2]/div/div/a/img'))
# 			frame.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/div[3]/div[2]/div/div[3]/div/div/a/img'))
# 			frame.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/div[3]/div[2]/div/div[4]/div/div/a/img'))
# 			frame.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/div[3]/div[2]/div/div[5]/div/div/a/img'))
# 			frame.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/div[3]/div[2]/div/div[6]/div/div/a/img'))
# 			frame.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/div[3]/div[2]/div/div[7]/div/div/a/img'))
# 			frame.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/div[3]/div[2]/div/div[8]/div/div/a/img'))
# 	trailer = driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/div[3]/div[3]/h2')
# 	if len(trailer) >0:
# 		if 'Trailers' in trailer[0].text:
# 			print(nd)
# 			time.sleep(10)


# 	for i in range(len(frame)):
# 		if len(frame[i]) > 0:
# 			if not os.path.exists('/Users/caoziyi/Desktop/短视频推荐/MovieLens/frames/{}'.format(nd)):
# 				os.makedirs('/Users/caoziyi/Desktop/短视频推荐/MovieLens/frames/{}'.format(nd))
# 			try:
# 				urllib.request.urlretrieve(frame[i][0].get_attribute("src"), '/Users/caoziyi/Desktop/短视频推荐/MovieLens/frames/{}/{}-{}.jpg'.format(nd,nd,i))
# 			except Exception as e:
# 				print(f"发生错误：{e}")
# 				continue

	

