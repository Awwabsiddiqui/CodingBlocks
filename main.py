import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from multiprocessing import Process
import csv
import requests
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestClassifier
from urllib.request import urlopen
import bs4
from bs4 import BeautifulSoup as soup
import json
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
import time
import sqlite3
from flask import Flask , render_template , redirect





### Numpy  ###

#a=np.array([[2, 4, 6, 8, 10], [3, 6, 9, 12, 15], [4, 8, 12, 16, 20]])
#b=a.T
# print(str(a) +" has shape : "+str(a.shape))
# print(str(b) +" has shape : "+str(b.shape))
# print(a)
# print(b)
# dot=np.dot(b,a)
# print(dot)
# print(str(np.max(dot)) +" at "+str(np.argmin(dot)))
# rand=np.random.randint(1 , 10 , (5,5))
# print(rand)
# print(np.unique(rand , return_counts=True , return_inverse=True))
#print(a[2:3])
# a=np.full((4,5),9)
# print(a)
#
# b=np.zeros((4,5))
# print(b)
#
# c=np.ones((4,4))
# print(c)
#
# d=np.eye((4))
# print(d)
#
# print(a-b)
# print(np.add(c,d))
# print(np.sum(a))
# print(np.mean(a))
# print(np.median(a))
#
#
# print(np.random.choice([1,4,5,6,8,90]))

### matplotlib  ###

# xmp=np.array([1,2,3,4,5,6])
# ymp=xmp**2
#
# plt.style.use("dark_background")
#
# plt.plot(xmp , ymp , label="square")
# plt.scatter(xmp , xmp**3 , label="cube")
# plt.plot(xmp , xmp**4 , label="power(4)")
# plt.title("Square graph")
# plt.legend()
# plt.show()



### pandas ###

# values={"value 1":np.random.randint(1,10,5)  ,  "value 2" : np.random.randint(20,50,5)  ,  "value3" : np.random.randint(100,200,5) , "self":(1,2,3,4,5)}
# df=pd.DataFrame(values)
# print(df)
# print("")
# print(df.head(2))
# print("")
# print(df.columns)
#df.to_csv('values.csv')
#df2=pd.read_csv('values.csv')
#df2.drop(columns=[unnamed:0])
#print(df2)
# values2={'v1' : [1,2,3,4,5] , 'v2':[6,7,8,9,10]}
# df3=pd.DataFrame(values2)
# print(df3)
# print(df3.describe())
# print(df3.iloc[3])
# print(df3.iloc[3,1])
# print(df3.sort_values(by=['v1'] , ascending=False))

### MNIST ###
#fd=pd.read_csv('mnist_train.csv')
# print(fd.shape)
# print(fd.head)
#data=fd.values

#np.random.shuffle(data)
#print(type(data) , data.shape)
# x=data[ : , 1:]
# y=data[ : , 0 ]
#print(x.shape , y.shape)
#
# def draw(x,y,i):
#     plt.imshow(x[i].reshape(28,28) , cmap='gray')
#     plt.title("Label" + str(y[i]))
#     plt.show()
#for i in range(5):
 #   draw(x,y,i)


### train mnist splitting
# split = int(0.80 * x.shape[0])
#print(split)
# xtrain , ytrain = x[:split , :] , y[:split]
# xtest , ytest = x[split: , :] , y[split:]
###using sklearn to split data ,,, same thing
#x_train , y_train , x_test , y_test = train_test_split(x , y , test_size=0.2 , random_state=5)


# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.imshow(xtrain[i].reshape(28, 28), cmap='gray')
#     plt.title(ytrain[i])
#     plt.axis("off")
# plt.show()



### miscelanus processing asynchronous ###
# def square(num=2):
#     print(num**2)
# procs=[]
#
# for i in range(5):
#     procs.append(Process(target=square , args=(5,)))
#
# for proc in procs:
#     proc.start()
# print("hello+")
# for proc in procs:
#     proc.join()



### Data Visualisation ###

# x=np.arange(10)
# y=x**2
# y2=2*x+5 #mx+c
# plt.style.use("seaborn")
# plt.plot(x , y , color="red" , linestyle='dotted' , marker='o')
# #plt.show()
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.plot(x , y2 , color="blue" , linestyle='dashed')
# plt.show()

###-----barGraph
# x=np.array([1,2,3,4,5])
# y=np.array([10,20,30,40,50])
# plt.bar(x,y , color='green' , label='lower')
# plt.bar(x+5,[70,80,90,100,110] , color='red' , label='higher')
# plt.ylim(0,120)
# plt.legend()
# plt.show()

###-----pieChart
# subject=['math' , 'CSE' , 'hindi' ,'science' , 'english']
# marks=[10,6,7,4,1]
# plt.pie(marks , labels=subject , explode=(0,0.5,0,0,0.5) , autopct='%1.1f%%' , shadow=True , startangle=180)
# plt.show()

###----moviesname ###
# df=pd.read_csv('movie_metadata.csv')
# titles=list(df.get('movie_title'))
#
# freq={}
# for title in titles:
#     length=len(title)
#
#     if freq.get(length) is None:
#         freq[length]=1
#     else:
#         freq[length]+=1
#
# x=np.array(list(freq.keys()))
# y=np.array(list(freq.values()))
# plt.style.use('seaborn')
# plt.scatter(x,y , color='black')
# plt.show()


### seaborn ###
#tips = sns.load_dataset('tips')
# sns.barplot(x='sex' , y='total_bill' , data=tips)
# plt.show()
# sns.countplot(x='time' , data=tips)
# plt.show()
# sns.boxplot(x='day' , y='total_bill' , data=tips)
# plt.show()
# sns.distplot(tips['total_bill'])
# plt.show()
# sns.jointplot(x='total_bill' , y='tip' , data=tips , kind='hex')
# plt.show()
# sns.pairplot(tips)
# plt.show()



### Opencv2  ###

# img = cv2.imread('awab.jpg')
# #print(img.shape)
# imgrgb=cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
# plt.imshow(imgrgb)
# plt.show()

### opencv2 facedetect ###
#
# cap = cv2.VideoCapture(0)
# cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
# while True:
#     success , frame = cap.read()
#     gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
#     if  success==False:
#         continue
#     faces = cascade.detectMultiScale(gray , 1.3 , 5)
#
#     for (x,y,w,h) in faces:
#         cv2.rectangle(frame , (x,y) , (x+w , y+h) , (255,0,0) , 2)
#     cv2.imshow("frame", frame)
#
#     key=cv2.waitKey(1) & 0xFF
#     if key==ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()


### Feature Selection in dataset
# datac = pd.read_csv('train.csv')
# #print(datac.head)
# x=datac.iloc[: , 0:20]
# y=datac.iloc[: , -1]

# topfeature =  SelectKBest(score_func=chi2 , k=10)
# fit = topfeature.fit(x,y)
# #print(fit.scores_)
# dfscores = pd.DataFrame(fit.scores_)
# dfcol = pd.DataFrame(x.columns)
# dfans = pd.concat([dfcol , dfscores] , axis=1)
# dfans.columns=["Feature" , "Score"]
# dfans = dfans.sort_values(by="Score" , ascending=False)
# #print(dfans)


### Random Forest Sklearn same as above Feature Selection (x , y)
# model = RandomForestClassifier()
# model.fit(x , y)
# #print(model.feature_importances_)
# featureimportance = pd.DataFrame(model.feature_importances_ , index=x.columns , columns=["Importance"])
# featureimportance=featureimportance.sort_values(by="Importance" , ascending=False)
# #print(featureimportance)
# plt.figure(figsize=(20 , 10))
# plt.bar(featureimportance.index , featureimportance["Importance"])
# plt.show()


### PCA - principal component analysis (Featue selection)
# (xtrain , ytrain) , (xtest , ytest) = mnist.load_data()
# print(xtest.shape)


### BeautifulSoup4 --- WebScraping

# andurl = "https://en.wikipedia.org/wiki/Android_version_history"
# anddata = urlopen(andurl)
# andhtml = anddata.read()
# andsoup = soup(andhtml , 'html.parser')
# print("heading : "+str(andsoup.findAll("h1" , {})))
# tables = andsoup.findAll("table" , {'class' : 'wikitable'})
# print("Number of tables : "+str(len(tables)))
# #print("First table : "+str(tables[0]))
#
# header = andsoup.findAll('th')
# heads=[]
# print("Headers are : "+str(len(header)))
# for i in range(10):
#     heads.append(header[i].text[:-1])
#
# rows = andsoup.findAll('tr')
# rowss=[]
# print("rows are : "+str(len(rows)))
# for i2 in range(10):
#     rowss.append(rows[i2].text[1:])
#
# rowd = andsoup.findAll('td')
# rowdd=[]
# print("rowd are : "+str(len(rowd)))
# for i3 in range(10):
#     rowdd.append(rowd[i3].text[:-1])
#
# print(heads)
# print("---------------------------------------------------------------------------------------------------------------")
# print(rowss)
# print("---------------------------------------------------------------------------------------------------------------")
# print(rowdd)


### Beautiful soup local offline
# with open('helo.html' , encoding='utf-8') as file:
#     pagesoup = soup(file , 'html.parser')
# h1s = pagesoup.findAll('h1')
# print(h1s)


### WEB API weather
# cityname = input("Enter city name")
# apiurl = "https://api.openweathermap.org/data/2.5/weather?q="+str(cityname)+"&appid=afe14ee59f357e079210acfefdfdf813"
# urlres = urlopen(apiurl)
# apidata = urlres.read()
# #print(apidata)
#
# jsondata = json.loads(apidata)
# print(jsondata)


### WEB API FaceBook near usless
# markurl = "http://graph.facebook.com/4/picture?type=large"
# res = requests.get(markurl)
# with open("markface.jpg" , "wb") as f:
#     f.write(res.content)

### WEB API GOOGLE API useless
# gupi = "https://maps.googleapis.com/maps/api/geocode/json?"
# param = {
#     "address" : "coding blocks pitampura",
#     "key" : "AIzaSyCwsrY4Uel9p7zt3b475MmJANabF43OpYE"
# }
# res=requests.get(gupi , params=param)
# print(res.content)

### WEB API IMAGE EXTRACT
# quoteurl = "https://www.passiton.com/inspirational-quotes?page=2"
# res = requests.get(quoteurl)
# soup = bs4.BeautifulSoup(res.content)
# element = soup.findAll("img")
# print(element[0].attrs["src"])


### Chuck Norris Jokes WEB SCRAPING
# for num in range(1 , 500):
#     chkurl = "http://api.icndb.com/jokes/"+str(num)
#
#     res = requests.get(chkurl)
#     soup = str(bs4.BeautifulSoup(res.content , features="html.parser"))
#     try:
#         jsondata = json.loads(soup)
#         s1 = str(jsondata["value"]["joke"])
#         s2 = s1.replace('"', '')
#         rows=rows = [ [str(num) , str(s2)]]
#         with open("noris.csv", 'a') as csvfile:
#             csvwriter = csv.writer(csvfile)
#             csvwriter.writerows(rows)
#     except:
#         rows2  = [[str(num), ]]
#         with open("noris.csv", 'a') as csvfile:
#             csvwriter = csv.writer(csvfile)
#             csvwriter.writerows(rows2)


### SELENIUM
# browser = webdriver.Chrome()
# browser.get("https://seedr.cc")
#
#
# browser.find_element( by = By.LINK_TEXT, value = "Login").click()
#
# browser.find_element( by = By.XPATH, value = "//input[@name='username'][@type='text'][@placeholder = 'email'][@tabindex='1'][@class='s-text-input']").send_keys("tamyah.67@gottakh.com")
# browser.find_element( by = By.XPATH, value = "//input[@name='password'][@type='password'][@tabindex='2'][@class='s-text-input']").send_keys("#Hello123")
# browser.find_element( by = By.XPATH, value = "//input[@type='checkbox'][@name='rememberme'][@tabindex='3']").click()
#
# # elem = browser.find_element(By.XPATH , value="//input[name='sign-in-submit'][@type='submit']")
# # elem.click()
# # browser.find_element(By.NAME , value = "sign-in-submit").send_keys(Keys.ENTER)
# ac = ActionChains(browser)
# ac.move_by_offset(776, 445).click().perform()


### SQL ###
# con=sqlite3.Connection('chinook.db')
# sqlis = pd.read_sql_query("SELECT * FROM employees;" , con)
# print(sqlis)



### FLASK ###
# app = Flask(__name__)
#
# @app.route('/')
# def home():
#     return render_template('helo.html')
#
# @app.route('/about')
# def about():
#     return "Awwab siddiqui"
#
# @app.route('/home')
# def nhome():
#     return redirect('/')
#
# if __name__ =="__main__":
#     app.debug = True
#     app.run()