# -*- coding: utf-8 -*-
import time
from sklearn import metrics,datasets
import pandas as pd
from sklearn.model_selection import train_test_split     #將數據分割
from sklearn.preprocessing import LabelEncoder           #用於標準化
# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB  ##樸素貝葉斯分類器的多項式模型
    model = MultinomialNB(alpha=0.01)              ##加法（拉普拉斯/ Lidstone）平滑參數（0表示無平滑）
    model.fit(train_x, train_y)                    ##將x,y傳入
    return model
# KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier      ##KNN分類演算法
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model
# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):           ##邏輯回歸分類演算法
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model
# SVM Classifier
def svm_classifier(train_x, train_y):                       ##支持向量機分類法
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)            ##指定算法中要使用的內核類型。是否啟用概率估計。
    model.fit(train_x, train_y)
    return model
def read_data():
    dataset_train = pd.read_csv('comedy_comparisons_train.csv', low_memory=False)    ##開啟檔案
    dataset_test = pd.read_csv('comedy_comparisons_test.csv', low_memory=False)      ##開啟檔案
    features_train = ['ID1', 'ID2']    #資料行的名稱
    train_x = dataset_train[features_train]  #指定x為訓練特徵
    test_x =dataset_test[features_train]
    train_y = dataset_train["like"]          #指定y為目的
    test_y = dataset_test["like"]
    # train_x['ID1'] = LabelEncoder().fit_transform(train_x['ID1'])   #用來計算各個詞語出現次數
    # train_x['ID2'] = LabelEncoder().fit_transform(train_x['ID2'])   #用來計算各個詞語出現次數

    return train_x, train_y, test_x, test_y
if __name__ == '__main__':
    model_save_file = None
    model_save = {}
    test_classifiers = ['LR','NB', 'KNN',  'SVM']
    classifiers = {'NB': naive_bayes_classifier,
                   'KNN': knn_classifier,
                   'LR': logistic_regression_classifier,
                   'SVM': svm_classifier,
                   }
    train_x, train_y, test_x, test_y = read_data()
    num_train, num_feat = train_x.shape                             ##num_train計算列數,num_feat計算行數
    num_test, num_feat = test_x.shape                               ##num_test計算列數,num_feat計算行數
    print '******************** Data Info *********************'
    print '#training data: %d, #testing_data: %d, dimension: %d' % (num_train, num_test, num_feat)
    for classifier in test_classifiers:
        print '******************* %s ********************' % classifier
        start_time = time.time()                                            ##抓取起始時間
        model = classifiers[classifier](train_x, train_y)
        print 'training took %fs!' % (time.time() - start_time)              ##計算訓練時間
        predict = model.predict(test_x)
        accuracy = metrics.accuracy_score(test_y, predict)
        print 'accuracy: %.2f%%' % (100 * accuracy)                          #準確值計算