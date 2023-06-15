import os
import pandas
from prettytable import PrettyTable
import tensorflow as tf
import numpy
import math
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, Adamax
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, classification_report

#Create Basic CNN Model
def create_cnn(x_train, model_name):
    cnn = tf.keras.Sequential([
        layers.Conv1D(filters=10, kernel_size=3, activation='relu', input_shape=(32, 1)),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1)
    ])

    # Compile the model
    cnn.compile(optimizer='adam',
                loss='mse',
                metrics=['mae', 'mse', 'accuracy', 'AUC'])
    #cnn.summary()
    #plot_model(cnn, show_layer_names=True)
    return cnn

#Data Structure for holding in Data 
class StockInformationalDataStructure:
    def __init__(self):
        self.startDate = ""
        self.endDate = ""
        self.name=""
        self.totalDataPoints = 0
        self.trainingDataPoints = 0
        self.testingDataPoints = 0
        self.MaxClosingValue = 0
        self.MinClosingValue = 0



#Load Data
PresentWorkingDirectory = os.getcwd()
PathToDataFiles = PresentWorkingDirectory + "/Data/AutomotiveStocks/"
DataFiles = []
for dirname, _, filenames in os.walk(PathToDataFiles):
    for filename in filenames:
        filepath = os.path.join(dirname, filename)
        print(filepath)
        DataFiles.append(filepath)


DataPoints = []
stockInfoArray = [] 
tab = PrettyTable()
tab.field_names = ['Name', 'Start Date', 'Total', 'Train', 'Min Value', 'Max Value']

outputTable = PrettyTable()
outputTable.field_names = ['Name', 'Max Dif', 'Min Dif', 'MSE', 'RMS', 'AME']


for Datafile in DataFiles:
    stock = StockInformationalDataStructure()
    name = Datafile.split('/')[-1]
    name = name.replace('.us.txt', '')
    stock.name = name
    # Read in CSV Data and Grab Key Values 
    data = pandas.read_csv(Datafile)
    stock.startDate =data['Date'].values[0]
    stock.endDate =data['Date'].values[stock.totalDataPoints-1]
    stock.MinClosingValue = data['Close'].min()
    stock.MaxClosingValue = data['Close'].max()

    # Split Training and Test Data
    stock.trainingDataPoints = int(len(data)*0.80)
    stock.totalDataPoints = len(data)
    stock.testingDataPoints = stock.totalDataPoints - stock.trainingDataPoints 
    train, test = data[0:stock.trainingDataPoints],data[stock.trainingDataPoints:stock.totalDataPoints]
    tab.add_row([stock.name, stock.startDate, stock.totalDataPoints, stock.trainingDataPoints, stock.MinClosingValue, stock.MaxClosingValue])
    
    # Normalize the Data
    train = train.loc[:, ["Close"]].values
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train)

    end_len = len(train_scaled)
    xTrain = []
    yTrain = []
    timesteps = 32

    for i in range(timesteps, end_len):
        xTrain.append(train_scaled[i - timesteps:i, 0])
        yTrain.append(train_scaled[i, 0])
    xTrain, yTrain = numpy.array(xTrain), numpy.array(yTrain)

    xTrain = numpy.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))

    model = create_cnn(xTrain, stock.name)
    model.fit(xTrain, yTrain, epochs = 10, batch_size=32)

    real_price = test.loc[:, ["Close"]].values

    dataset_total = pandas.concat((data["Close"], test["Close"]), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(test) - timesteps:].values.reshape(-1,1)
    inputs = scaler.transform(inputs)
    X_test = []

    for i in range(timesteps, inputs.size-1):
        X_test.append(inputs[i-timesteps:i, 0])
    X_test = numpy.array(X_test)

    X_test = numpy.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predict = model.predict(X_test)
    predict = scaler.inverse_transform(predict)
    real_price_temp = numpy.delete(real_price, -1)
    differenceArray = []
    i = 0
    for val in real_price_temp:
        dif = abs(val - predict[i])
        i = i + 1
        differenceArray.append(dif)

    maxDifference = max(differenceArray)
    maxDifferenceItem = round(maxDifference.item(0), 3)
    minDifference = min(differenceArray)
    minDifferenceItem = round(minDifference.item(0), 3)
    meanabsoluteerror = numpy.mean(differenceArray)
    meanabsoluteerror = round(meanabsoluteerror, 3)
    sum = 0
    for item in differenceArray:
        sum = sum + item*item
    meansquaredError = sum/len(differenceArray)
    meansquaredErrorItem = round(meansquaredError.item(0), 3)
    rms = math.sqrt(meansquaredErrorItem)
    rms = round(rms, 3)
    outputTable.add_row([stock.name, maxDifferenceItem,  minDifferenceItem, meansquaredErrorItem, rms, meanabsoluteerror])
    plt.plot(real_price, color = "red", label = "Real Stock Price")
    plt.plot(predict, color = "black", label = "Predict Stock Price")
    plt.title("Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel(stock.name  + " Stock Price")
    plt.legend()
    plt.show()

print(tab)
print(outputTable)




