from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from openpyxl import Workbook
import sys
import pandas as pd
idx = sys.argv[1]

ml_small = pd.read_csv('./MovieLens/ml-latest-small/links.csv')
movielens = ml_small['movieId'].tolist()
gap = len(movielens)//10
this_time = []


if idx == '0':
	this_time = movielens[:gap]
elif idx == '1':
	this_time = movielens[gap:2*gap]
elif idx == '2':
	this_time = movielens[2*gap:3*gap]
elif idx == '3':
	this_time = movielens[3*gap:4*gap]
elif idx == '4':
	this_time = movielens[4*gap:5*gap]
elif idx == '5':
	this_time = movielens[5*gap:6*gap]
elif idx == '6':
	this_time = movielens[6*gap:7*gap]
elif idx == '7':
	this_time = movielens[7*gap:8*gap]
elif idx == '8':
	this_time = movielens[8*gap:9*gap]
elif idx == '9':
	this_time = movielens[9*gap:]



not_done=[32770]
gap = len(not_done)//10
this_time = []
if idx == '0':
	this_time = not_done[:gap]
elif idx == '1':
	this_time = not_done[gap:2*gap]
elif idx == '2':
	this_time = not_done[2*gap:3*gap]
elif idx == '3':
	this_time = not_done[3*gap:4*gap]
elif idx == '4':
	this_time = not_done[4*gap:5*gap]
elif idx == '5':
	this_time = not_done[5*gap:6*gap]
elif idx == '6':
	this_time = not_done[6*gap:7*gap]
elif idx == '7':
	this_time = not_done[7*gap:8*gap]
elif idx == '8':
	this_time = not_done[8*gap:9*gap]
elif idx == '9':
	this_time = not_done[9*gap:]



option = webdriver.ChromeOptions()
option.add_experimental_option('excludeSwitches', ['enable-automation'])
option.add_argument('user-agent="Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.6312.124 Safari/537.36"')
driver = webdriver.Chrome(options=option)
driver.maximize_window()


workbook = Workbook()
sheet = workbook.active



data = ['id','describe','language','directors','cast','poster','genere','tag','youtube']
sheet.append(data)

driver.get("https://movielens.org/movies/1")
time.sleep(20)


for movie_id in this_time:
	driver.get("https://movielens.org/movies/{}".format(movie_id))
	time.sleep(1)
	print("movieid",movie_id)


	describe = ''
	hots = driver.find_elements(By.XPATH,"//div[@class='movie-alt']/div[@class='row movie-details-block']/div[@class='col-md-6']/p");
	# //*[@id="main-container"]/ui-view/movie-page/div/div[2]/div[1]/p
	# //*[@id="main-container"]/ui-view/movie-page/div/div[2]/div[1]/p
	if len(hots) > 0:
		describe = hots[0].text
		print(describe)

	language_l = []
	language_l.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/div[2]/div[2]/div[1]/span[1]/a'));
	language_l.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/div[2]/div[2]/div[1]/span[2]/a'));
	language_l.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/div[2]/div[2]/div[1]/span[3]/a'));
	language_l.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/div[2]/div[2]/div[1]/span[4]/a'));
	language = []
	for i in language_l:
		if len(i) > 0:
			language.append(i[0].text)
	print(language)

	directors = []
	directors.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/div[2]/div[2]/div[2]/span[1]/a'));
	directors.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/div[2]/div[2]/div[2]/span[2]/a'));
	# directors.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/div[2]/div[2]/div[1]/span[3]/a'));
	# directors.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/div[2]/div[2]/div[1]/span[4]/a'));
	direc = []
	for i in directors:
		if len(i) > 0:
			direc.append(i[0].text)
	print(direc)

	cast_l = []
	cast_l.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/div[2]/div[2]/div[3]/span[1]/a'));
	cast_l.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/div[2]/div[2]/div[3]/span[2]/a'));
	cast_l.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/div[2]/div[2]/div[3]/span[3]/a'));
	cast_l.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/div[2]/div[2]/div[3]/span[4]/a'));
	cast = []
	for i in cast_l:
		if len(i) > 0:
			cast.append(i[0].text)
	print(cast)



	image_url = ''
	element = driver.find_elements(By.XPATH, "//img[@class='img-responsive rounded ml4-bordered']")
	if len(element) > 0:
		image_url = element[0].get_attribute("src")
		print('poster',image_url)



	genere = []
	genere.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/div[1]/div[2]/div[2]/div[2]/div[1]/span[1]/a/b'));
	genere.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/div[1]/div[2]/div[2]/div[2]/div[1]/span[2]/a/b'));
	genere.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/div[1]/div[2]/div[2]/div[2]/div[1]/span[3]/a/b'));
	genere.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/div[1]/div[2]/div[2]/div[2]/div[1]/span[4]/a/b'));
	genere.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/div[1]/div[2]/div[2]/div[2]/div[1]/span[5]/a/b'));
	gen = []
	for i in genere:
		if len(i) > 0:
			gen.append(i[0].text)
	print(gen)



	tag_l = []
	tag_l.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/movie-tags/div/div[2]/div[3]/div/div[1]/ml4-tag/div/div/a/span'));
	tag_l.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/movie-tags/div/div[2]/div[3]/div/div[2]/ml4-tag/div/div/a/span'));
	tag_l.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/movie-tags/div/div[2]/div[3]/div/div[3]/ml4-tag/div/div/a/span'));
	tag_l.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/movie-tags/div/div[2]/div[3]/div/div[4]/ml4-tag/div/div/a/span'));
	tag_l.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/movie-tags/div/div[2]/div[3]/div/div[5]/ml4-tag/div/div/a/span'));
	tag_l.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/movie-tags/div/div[2]/div[3]/div/div[6]/ml4-tag/div/div/a/span'));
	tag_l.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/movie-tags/div/div[2]/div[3]/div/div[7]/ml4-tag/div/div/a/span'));
	tag_l.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/movie-tags/div/div[2]/div[3]/div/div[8]/ml4-tag/div/div/a/span'));
	tag_l.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/movie-tags/div/div[2]/div[3]/div/div[9]/ml4-tag/div/div/a/span'));
	tag_l.append(driver.find_elements(By.XPATH,'//*[@id="main-container"]/ui-view/movie-page/div/movie-tags/div/div[2]/div[3]/div/div[10]/ml4-tag/div/div/a/span'));
	tags = []
	for i in tag_l:
		if len(i) > 0:
			tags.append(i[0].text)
	print(tags)


	youtube_url = ''
	element = driver.find_elements(By.XPATH, '//*[@id="main-container"]/ui-view/movie-page/div/div[4]/div[3]/div/div/div/youtube-tile/div/img')
	# //*[@id="main-container"]/ui-view/movie-page/div/div[3]/div[3]/div/div/div/youtube-tile/div/img
	# //*[@id="main-container"]/ui-view/movie-page/div/div[3]/div[3]/div/div/div/youtube-tile/div/img
	if len(element) > 0:
		youtube_url = element[0].get_attribute("src")
		print('youtube_url',youtube_url)
	else:
		element = driver.find_elements(By.XPATH, '//*[@id="main-container"]/ui-view/movie-page/div/div[3]/div[3]/div/div/div/youtube-tile/div/img')
		if len(element) > 0:
			youtube_url = element[0].get_attribute("src")
			print('youtube_url',youtube_url)
	print('\n')
	sheet.append([movie_id,
		describe,
		','.join(language),
		','.join(direc),
		','.join(cast),
		image_url,
		','.join(gen),
		','.join(tags),
		youtube_url])




driver.quit()

workbook.save('./crawler/{}-1.xlsx'.format(idx))


# https://youtu.be/CbfnCflL4-E

# https://img.youtube.com/vi/CbfnCflL4-E/0.jpg


