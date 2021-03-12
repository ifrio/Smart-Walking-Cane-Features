#__import__
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn import linear_model
import pandas as pd
from sklearn import svm
from flask import Flask

# Task:
app = Flask(__name__)

def data_gen(x, y, longi, lati, freqLow, freqHigh, typey):
  row = []
  long = round(random.randint(x - 1, x + 1))
  long = float(str(longi) + str(long))
  lat = round(random.randint(y - 1, y + 1))
  lat = float(str(lati) + str(lat))
  freq = random.randint(freqLow, freqHigh)
  if typey == 0 and freq >= 15:
      wash = 1
  elif typey == 1 and freq >= 55:
      wash = 1
  elif typey == 2:
      wash = 1
  else:
      wash = 0
  row.append(long)
  row.append(lat)
  row.append(freq)
  row.append(wash)
  row.append(typey)
  return row

def park_data():
  #park, 33.6763114050008, -117.74963972385233
  #if touched 15, labeled wash stick now
  return data_gen(8, 3, 33.6742524050008, -117.74216672385233, 0, 30, 0)

def home_data():
    #home, 33.68649735286263, -117.76223689376901
    #estimate of the number of objects the stick would touch
    #if touched 55 <, label wash stick now

    return data_gen(3, 1, 33.68649735286263, -117.76223689376901, 20, 60, 1)

def store_data():
    #target, 33.68559244242906, -117.81338458443688
    #estimate of the number of objects the stick would touch
    #if touched 115 <, label wash stick now
    return data_gen(6, 8, 33.68559244242906, -117.81338458443688, 100, 200, 2)

def makedatatable():
  records = 1000
  data = []

  for i in range(int(records * .5)):
    i = home_data()
    data.append(i)

  for i in range(int(records * .1)):
    i = park_data()
    data.append(i)

  for i in range(int(records * .4)):
    i = store_data()
    data.append(i)

  data = pd.DataFrame(data,
                      columns=['Long', 'Lat', 'Frequency', "RecWash", 'Label'])
  return data

@app.route("/predictFrequency/<long>/<lat>/<label>")
def predictFrequency(long,lat, label):
  data = makedatatable()
  X_train = data[['Long', 'Lat', 'Label']]
  Y_train = data[['Frequency']]
  
  # # Experiment 1- Model 1
  model = linear_model.LinearRegression()
  model.fit(X_train, Y_train)  # learn, train, fit
  return str(model.predict([[long, lat, label]]))

@app.route("/predictLocation/<long>/<lat>/<frequency>")
def predictLocation(long,lat, frequency):
  data = makedatatable()
  X_train = data[['Long','Lat','frequency']]
  Y_train = data[['Label']]
  model = svm.SVC()
  model.fit(X_train, Y_train)
  return model.predict([[long,lat,frequency]])

@app.route("/predictNextClean/<long>/<lat>/<frequency>")
def predictNextClean(long, lat, frequency):
  data = makedatatable()
  X_train = data[['Long','Lat','Frequency']]
  Y_train = data[['RecWash']]
  model = svm.SVC()
  model.fit(X_train, Y_train)
  return model.predict([[long,lat,frequency]])

def researchQ1():
  print("")
  """
  Research Question #1:
  Use longitude, latitude, and label (the three categories of location: home, park, store) to predict frequency
  """
  print("###################### Experiment 1 ######################")

  data = makedatatable()

  X_train, X_test, y_train, y_test = train_test_split(
      data[['Long', 'Lat', 'Label', 'RecWash']],
      data[['Frequency']],
      test_size=0.3,
      random_state=0)

  # # Experiment 1- Model 1
  print("Experiment 1- Model 1: Raised to 1")
  model = linear_model.LinearRegression()
  model.fit(X_train, y_train)  # learn, train, fit
  print('Res:', model.score(X_test, y_test))

  # Experiment 1- Model 2
  # use the polynomial regression on the data to see accuracy 2
  print("Experiment 1- Model 2:Raised to 2")
  model = make_pipeline(PolynomialFeatures(2), Ridge())
  model.fit(X_train, y_train)
  print('Res:', model.score(X_test, y_test))

  # # Experiment 1- Model 3
  # # use the polynomial regression on the data to see accuracy 5
  print("Experiment 1- Model 3: Raised to 5")
  model = make_pipeline(PolynomialFeatures(5), Ridge())
  model.fit(X_train, y_train)
  print('Res:', model.score(X_test, y_test))

  # # Experiment 1- Model 4
  # # use the polynomial regression on the data to see accuracy 7
  print("Experiment 1- Model 4: Raised to 7")
  model = make_pipeline(PolynomialFeatures(7), Ridge())
  model.fit(X_train, y_train)
  print('Res:', model.score(X_test, y_test))
  print("")

def ResearchQ2():
  # # Challenges
  # # - tried polynomial regression base 5 -> thought adding more xs will make it more accurate, but it turned out to not be the case -> google -> it turns out to be a very common situation in machine learning : mention overfitting and underfitting

  # # Experiment 1-1
  # # Res: 0.8818577790172654
  # # Experiment 1-2
  # # Res: 0.8818266011102243
  # # Experiment 1-3
  # # Res: 0.8818241773885248
  # # Experiment 1-4
  # # Res: 0.8818435223834024
  """ 
  Research Question # 2:
  Using the amount of objects we will run into to predict location 
  """
  print("###################### Experiment 2 ######################")
  data = makedatatable()

  X_train, X_test, y_train, y_test = train_test_split(
      data[['Long', 'Lat', 'Frequency']],
      data['Label'],
      test_size=0.3,
      random_state=0)

  print("Experiment 2-1: with Longitude, Latitude, Frequency")
  model = svm.SVC()
  model.fit(X_train, y_train)
  print('Res:', model.score(X_test, y_test))

  X_train, X_test, y_train, y_test = train_test_split(data[['Long', 'Lat']],
                                                      data['Label'],
                                                      test_size=0.3,
                                                      random_state=0)

  print("Experiment 2-2: with Longitude and Latitude")
  model = svm.SVC()
  model.fit(X_train, y_train)
  print('Res:', model.score(X_test, y_test))

  X_train, X_test, y_train, y_test = train_test_split(data[['Frequency']],
                                                      data['Label'],
                                                      test_size=0.3,
                                                      random_state=0)

  print("Experiment 2-3: with Frequency ")
  model = svm.SVC()
  model.fit(X_train, y_train)
  print('Res:', model.score(X_test, y_test))
  print("")

def ResearchQ3():
  """ 
  Research Question # 3:
  predict when they should clean their stick, based on the amount  touched a certain amount or type of thing and location
  """
  print("###################### Experiment 3 ######################")

  data = makedatatable()
  
  X_train, X_test, y_train, y_test = train_test_split(data[['Long', 'Lat']],
                                                      data['RecWash'],
                                                      test_size=0.3,
                                                      random_state=0)

  print("Experiment 3-1: with Longitude and Latitude")
  model = svm.SVC()
  model.fit(X_train, y_train)
  print('Res:', model.score(X_test, y_test))

  X_train, X_test, y_train, y_test = train_test_split(data[['Label']],
                                                      data['RecWash'],
                                                      test_size=0.3,
                                                      random_state=0)

  print("Experiment 3-2: with Label")
  model = svm.SVC()
  model.fit(X_train, y_train)
  print('Res:', model.score(X_test, y_test))

  X_train, X_test, y_train, y_test = train_test_split(data[['Frequency']],
                                                      data['RecWash'],
                                                      test_size=0.3,
                                                      random_state=0)

  print("Experiment 3-3: with Frequency")
  model = svm.SVC()
  model.fit(X_train, y_train)
  print('Res:', model.score(X_test, y_test))

  X_train, X_test, y_train, y_test = train_test_split(
      data[['Long', 'Lat', 'Frequency', 'Label']],
      data['RecWash'],
      test_size=0.3,
      random_state=0)

  print("Experiment 3-4: with Longitude, Latitude, Frequency, Label")
  model = svm.SVC()
  model.fit(X_train, y_train)
  print('Res:', model.score(X_test, y_test))
  print("###################### The END ###########################")

app.run(host="0.0.0.0")