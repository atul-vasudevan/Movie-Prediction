import sys
import numpy as np
import pandas as pd
import ast
import csv
from datetime import datetime
from collections import defaultdict 
from collections import OrderedDict
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.metrics import precision_score, accuracy_score, recall_score
from scipy.stats.stats import pearsonr

def loop(list_of_cast,key):
    list_of_char=[]
    ld={}
    for dictionary in list_of_cast:
        for i in dictionary:
            if i == key:
                list_of_char.append(dictionary[key])
    return list_of_char

def rating(value,genres_dict):
    for e in value:
        genres_dict[e]+=1
    return genres_dict

def calculateRating(value,genres_dict):
    val=0
    for e in value:
        if e in genres_dict.keys():
            val+=genres_dict[e]
    return val

def orginalLanguageFunction(value):
    if value not in originalLanguageDict.keys():
        originalLanguageDict[value]+=1
    return originalLanguageDict[value]

def topGenresFunction(row,e):
    if e in row:
        return 1
    return 0

def writeCSV(part1Summary,part2Summary):
    with open('z5199180.PART1.summary.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(part1Summary)

    with open('z5199180.PART2.summary.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(part2Summary)

def preprocessing(df,learning):
    genres_dict=defaultdict(int)
    cast_dict=defaultdict(int)
    crew_dict=defaultdict(int)
    keywords_dict=defaultdict(int)
    production_companies_dict=defaultdict(int)
    df["genres"] = df.apply(lambda row: loop(ast.literal_eval(row["genres"]),"name"), axis = 1)
    df["cast"] = df.apply(lambda row: loop(ast.literal_eval(row["cast"]),"name"), axis = 1)
    df["crew"] = df.apply(lambda row: loop(ast.literal_eval(row["crew"]),"name"), axis = 1)
    df["keywords"] = df.apply(lambda row: loop(ast.literal_eval(row["keywords"]),"name"), axis = 1)
    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
    df['release_month']=df['release_date'].apply(lambda row:row.split('-')[1])
    print(df['release_month'])
    df["production_companies"] = df.apply(lambda row: loop(ast.literal_eval(row["production_companies"]),"name"), axis = 1)
    df["cast"].apply(lambda row: rating(row,cast_dict))
    df["crew"].apply(lambda row: rating(row,crew_dict))
    df["genres"].apply(lambda row: rating(row,genres_dict))
    df["production_companies"].apply(lambda row: rating(row,production_companies_dict))
    df["keywords"].apply(lambda row: rating(row,keywords_dict))
    df["cast_length"]=df["cast"].apply(lambda row:len(row))
    df["crew_length"]=df["crew"].apply(lambda row:len(row))
    df["overview_length"]=df["overview"].apply(lambda row:len(row))
    df["genre_rating"]=df["genres"].apply(lambda row:calculateRating(row,genres_dict))
    df["cast_rating"]=df["cast"].apply(lambda row:calculateRating(row,cast_dict))
    df["crew_rating"]=df["crew"].apply(lambda row:calculateRating(row,crew_dict))
    df["keywords_dict_rating"]=df["keywords"].apply(lambda row:calculateRating(row,keywords_dict))
    df["production_companies_rating"]=df["production_companies"].apply(lambda row:calculateRating(row,production_companies_dict))
    movieIdArray=df["movie_id"].values
    originalLanguageDict=defaultdict(int)
    le = LabelEncoder() #encoder for target
    df["original_language_value"] = le.fit_transform(df["original_language"])
    
    if learning == "Regression":
        df1=df[['overview_length','budget','release_month','crew_length','runtime','genre_rating','cast_length','production_companies_rating','original_language_value']]
    elif learning == "Classification":
        df1=df[['keywords_dict_rating','budget','runtime','genre_rating','cast_length','original_language_value']]
    return df1,movieIdArray

def loadRegression(movie_path):
    df = pd.read_csv(movie_path)
    dfFeatures,movieIdArray=preprocessing(df,"Regression")
    X =dfFeatures.values
    Y = df['revenue'].values
    return X,Y, movieIdArray

def loadClassification(movie_path):
    df = pd.read_csv(movie_path)
    dfFeatures,movieIdArray=preprocessing(df,"Classification")
    X =dfFeatures.values
    Y = df['rating'].values
    return X,Y,movieIdArray


if __name__ == "__main__":
    training=sys.argv[1]
    test=sys.argv[2]
    print(training)
    print(test)
    X_train, Y_train,movieIdArrayTrainT = loadRegression(training)
    X_test, Y_test, movieIdArrayTestT = loadRegression(test)
    model = linear_model.LinearRegression()
    model.fit(X_train, Y_train)

    y_pred = model.predict(X_test)

    part1Summary=[]
    part1Summary.append(["MSR","correlation"])
    part1Output=[]
    part1Output.append(["movie_id","predicted_revenue"])
    for i in range(len(Y_test)):
        print("Expected:", Y_test[i], "Predicted:", y_pred[i])
    d={'movie_id':movieIdArrayTestT,'predicted_revenue':y_pred}
    csvDfOutputPart1=pd.DataFrame(data=d)
    csvDfOutputPart1.to_csv(r'PART1.output.csv',index=False,header=True)
    print("Mean squared error: %.2f"
          % mean_squared_error(Y_test, y_pred))
    coeff,p_value=pearsonr(Y_test, y_pred)
    print(coeff)
    print(p_value)
    part1Summary.append([mean_squared_error(Y_test, y_pred),coeff])

    part2Summary=[]
    part2Summary.append([average_precision","average_recall","accuracy"])
    CX_train, CY_train,movieIdArrayTrainV = loadClassification(training)
    CX_test, CY_test,movieIdArrayTestV = loadClassification(test)
    knn = KNeighborsClassifier()
    knn.fit(CX_train, CY_train)

    predictions = knn.predict(CX_test)
    print("Predictions:\n",predictions)
    d={'movie_id':movieIdArrayTestV,'predicted_rating':predictions}
    csvDfOutput=pd.DataFrame(data=d)
    csvDfOutput.to_csv(r'PART2.output.csv',index=False,header=True)
    print("confusion_matrix:\n", confusion_matrix(CY_test, predictions))
    print("precision:\t", precision_score(CY_test, predictions, average=None))
    print("recall:\t\t", recall_score(CY_test, predictions, average=None))
    print("accuracy:\t", accuracy_score(CY_test, predictions))
    prec=np.average(precision_score(CY_test, predictions, average=None))
    print(prec)
    recall=np.average(recall_score(CY_test, predictions, average=None))
    print(recall)
    part2Summary.append([prec,recall,accuracy_score(CY_test, predictions)])
    writeCSV(part1Summary,part2Summary)
