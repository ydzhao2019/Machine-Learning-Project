import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, fbeta_score, multilabel_confusion_matrix
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn import linear_model, tree, model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn import ensemble
import matplotlib.pyplot as plt

from get_feature import get_all_feature
from get_data import get_test
from select_feature import Feature_Inputs

plot_method = 'longshort'
scorer = {'accuracy': metrics.make_scorer(metrics.accuracy_score),
           'precision': metrics.make_scorer(metrics.precision_score, average = 'macro'),
           'recall': metrics.make_scorer(metrics.recall_score, average = 'macro'),
           'f1_macro': metrics.make_scorer(metrics.f1_score, average = 'macro'),
           'f1_weighted': metrics.make_scorer(metrics.f1_score, average = 'weighted'),
           'f0.5_macro': metrics.make_scorer(metrics.fbeta_score, beta=0.5, average = 'macro'),}
time_split_sample = False
X, Y, SP= Feature_Inputs(SPget=True, y_method = 'roll_vol') # equal, roll_vol, normal

def DecisionTreeFun(x,y,split= 0.2, average= 'macro', roc= 'micro', n_important_feature=20, grid_score = 'f0.5_macro'):
    '''classifier of decision tree model
    x = dataframe: independent variable
    y = dataframe: depedent variable (need not apply one hot encode)
    split = ratio assign to be test set
    
    average = [ ‘binary’, ‘micro’-(globally count TP, FP...) , ‘macro’, ‘samples’, ‘weighted’]
                more than two class cannot choose binary
    see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html for explain
                
    roc = [‘micro’, ‘macro’, ‘samples’, ‘weighted’, None]
    see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
    
    HyperParameters = [
                        {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 'max_depth': [2, 3, 4, 5], 'min_samples_split': [5]},
                        {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 'max_depth': [3], 'min_samples_split': [3, 5, 7, 9]}
                        ]
    '''
    X_train, X_test, y_train, y_test = Train_Test_Split(x, y, time=time_split_sample)    

    
    estimator = tree.DecisionTreeClassifier()
    parameters = [
                {'criterion': ['gini'], 'splitter': ['best', 'random'], 'max_depth': [2,6,8,12], 'min_samples_split': [3,5]},
                ]
    clf = model_selection.GridSearchCV(estimator, parameters, scoring = scorer[grid_score],cv = model_selection.StratifiedKFold(n_splits = 10, shuffle = True, random_state = 2020))
    clf.fit(X_train, y_train)

    X_names = X_train.columns
    feature_importantance = pd.Series(clf.best_estimator_.feature_importances_, index=X_names).sort_values(ascending=False)[:n_important_feature]
    y_pred = pd.Series(clf.best_estimator_.predict(X_test), index=X_test.index)
    scores = ScoreFunc(y_test, y_pred)
    scores.name = 'DecisionTree_GridSearch'
    print(scores)
    ret = PredictedReturn(y_pred, method=plot_method)
    PredictedReturn(y_test, method=plot_method,title='DecisionTree_GridSearch')
    plt.show()
    return clf, scores, ret, feature_importantance

def ScoreFunc(y_test, y_pred,average= 'macro', roc= 'macro'):
    rst = {}
    rst['Precision_score'] = precision_score(y_test, y_pred, average = average)
    rst['Recall_score'] = recall_score(y_test, y_pred, average = average)
    rst['F1_score'] = f1_score(y_test, y_pred, average = average)
    rst['F0.5_score'] = fbeta_score(y_test, y_pred, average = average, beta=0.5)
    # transform into 3 catagory
    trans = LabelBinarizer().fit(y_test.unique())
    y_test_onehot = trans.transform(y_test)
    y_pred_onehot = trans.transform(y_pred)
    rst['ROC_AUC_score'] = roc_auc_score(y_test_onehot, y_pred_onehot, average = roc)
    rst['ROC_AUC_score_weighted'] = roc_auc_score(y_test_onehot, y_pred_onehot, average = 'weighted')
    for i in range(3):
        rst['Confusion Matrix_'+trans.classes_[i]] = multilabel_confusion_matrix(y_test_onehot, y_pred_onehot)[i]
    s = pd.Series(rst)
    return s

def DecisionTreeBase(x,y,split= 0.2):
    X_train, X_test, y_train, y_test = Train_Test_Split(x, y, time=time_split_sample)    

    
    clf = tree.DecisionTreeClassifier().fit(X_train, y_train)
    y_pred = pd.Series(clf.predict(X_test), index=X_test.index)
    scores = ScoreFunc(y_test, y_pred)
    scores.name = 'DecisionTree_Base'
    print(scores)
    ret = PredictedReturn(y_pred, method=plot_method)
    PredictedReturn(y_test, method=plot_method,title='DecisionTree_Base')
    return scores, ret


def EnsemblingCheck(x,y,time_split_sample=time_split_sample, split= 0.2):
    X_train, X_test, y_train, y_test = Train_Test_Split(x, y, time=time_split_sample)    

    
    score_list = []
    aum_list = []
    method = plot_method
    # bagging
    estimator = tree.DecisionTreeClassifier()
    bagging = ensemble.BaggingClassifier(base_estimator = estimator).fit(X_train,y_train)
    y_pred = pd.Series(bagging.predict(X_test), index=X_test.index)
    scores = ScoreFunc(y_test, y_pred)
    scores.name = 'Bagging'
    print(scores)
    aum_list.append(PredictedReturn(y_pred, method=method,title='Bagging'))
#    plt.show()
    score_list.append(scores)
    
    # randomforest
    rf = RandomForestClassifier().fit(X_train, y_train)
    y_pred = pd.Series(rf.predict(X_test), index=X_test.index)
    scores = ScoreFunc(y_test, y_pred)
    scores.name = 'RandomForest'
    print(scores)
    aum_list.append(PredictedReturn(y_pred, method=method,title='RandomForest'))
#    plt.show()
    score_list.append(scores)
    
    # adaboost
    adaboost = AdaBoostClassifier(n_estimators=100).fit(X_train, y_train)
    y_pred = pd.Series(adaboost.predict(X_test), index=X_test.index)
    scores = ScoreFunc(y_test, y_pred)
    scores.name = 'AdaBoost'
    print(scores)
    aum_list.append(PredictedReturn(y_pred, method=method,title='AdaBoost'))
#    plt.show()
    score_list.append(scores)
    
    # gradientboost
    gboost = GradientBoostingClassifier().fit(X_train, y_train)
    y_pred = pd.Series(gboost.predict(X_test), index=X_test.index)
    scores = ScoreFunc(y_test, y_pred)
    scores.name = 'GradientBoost'
    print(scores)
    aum_list.append(PredictedReturn(y_pred, method=method,title='GradientBoost'))
    plt.show()
    score_list.append(scores)
    DF1 = pd.DataFrame(score_list).T
    DF2 = pd.DataFrame(aum_list, index=['Bagging','RandomForest','AdaBoost','GradientBoost']).T
#    DF2.plot(title='Ensemble')
    return DF1, DF2
#    # gradientboost grid serach
#    estimator = GradientBoostingClassifier()
#    parameters = [
#                {'loss':['deviance'],'max_depth': [2,4,6], 'min_samples_leaf': [1,3,5]},
#                ]
#    clf = model_selection.GridSearchCV(estimator, parameters, scoring = scorer['f0.5_weighted'],
#                                       cv = model_selection.StratifiedKFold(n_splits = 10, shuffle = True, random_state = 2020))
#    clf.fit(X_train, y_train)
#    y_pred = pd.Series(clf.best_estimator_.predict(X_test), index=X_test.index)
#    scores = ScoreFunc(y_test, y_pred)
#    scores.name = 'GradientBoost'
#    print(scores)
#    aum_list.append(PredictedReturn(y_pred, method=method,title='GradientBoost'))
#    plt.show()
#    score_list.append(scores)
#    pd.DataFrame(score_list).T
#    pd.DataFrame(aum_list, index=['Bagging','RandomForest','AdaBoost','GradientBoost']).T.plot(title='Ensemble')
    
def Stacking(x,y,time_split_sample=time_split_sample,split= 0.2):
    X_train, X_test, y_train, y_test = Train_Test_Split(x, y, time=time_split_sample)    

    estimators=[
#                 ('Logist', LogisticRegression(multi_class='multinomial',max_iter=1000)),
                 ('DecisionTree',tree.DecisionTreeClassifier(class_weight='balanced',max_depth=3)),
                 ('SVC', SVC()),
                 ('NB', GaussianNB())
                 ]
    lv2 = [
            ('DecisionTree',tree.DecisionTreeClassifier(class_weight='balanced',max_depth=5),
             [{'criterion': ['gini'], 'splitter': ['best', 'random'], 'max_depth': [2,6,8,12], 'min_samples_split': [3,5]},]
             ),
            ('NB',GaussianNB(),
             [{ 'var_smoothing':[1e-9,1e-11]}]
            ),
            ('Logist', LogisticRegression(multi_class='multinomial'),
             [{'penalty': ['l1', 'l2'],'C':[0.001,.009,0.01]}]
             )
            ]

    stacking_rst = []
    aum_rst = []
    for i in lv2:
        est = i[1]
        para = i[2]
        gs_clf = model_selection.GridSearchCV(est, para, scoring = scorer['f0.5_macro'],
                                           cv = model_selection.StratifiedKFold(n_splits = 10, shuffle = True, random_state = 2020))
        clf = StackingClassifier(estimators=estimators, final_estimator=gs_clf).fit(X_train, y_train)
        y_pred = pd.Series(clf.predict(X_test), index=X_test.index)
        scores = ScoreFunc(y_test, y_pred)
        scores.name = i[0]
        print(scores)
        aum = PredictedReturn(y_pred, method=plot_method,title=i[0])
        plt.show()
        aum_rst.append(aum)
        stacking_rst.append(scores)
    score_rst = pd.concat(stacking_rst, axis=1)
    aum_rst = pd.concat(aum_rst, axis=1)
    aum_rst.columns = [i[0] for i in lv2]
#    aum_rst['Benchmark'] = (PredictedReturn(y_test, method=plot_method))
    aum_rst.plot(title='Stacking Final_estimator GridSearch')
    return score_rst, aum_rst


def PredictedReturn(y_pred, method='long', title=None):
    '''calculate holding return
    y_predict = Series, with up,down, mid
    method:
        long: long only portfolio, sensitive to up precision only
        longshort: long short portfolio, long at up and short at down
    '''
    y_ = y_pred.copy().sort_index()
    y_[y_pred=='up'] = 1
    y_[y_pred=='mid'] = 0
    if method == 'long':
        y_[y_pred=='down'] = 0
    elif method == 'longshort':
        y_[y_pred=='down'] = -1
    if title is None:
        (SP[y_.index]*y_+1).sort_index().cumprod().plot()
        return (SP[y_.index]*y_+1).sort_index().cumprod()
    else:
        (SP[y_.index]*y_+1).sort_index().cumprod().plot(title=title)
        return (SP[y_.index]*y_+1).sort_index().cumprod()

def Train_Test_Split(x, y, split=0.2, time=False):
    if time:
        threshold = int(len(x)*0.8)
        old_index = x.index[:threshold]
        new_index = x.index[threshold:]
        X_train = x.loc[old_index, x.columns]
        y_train = y.loc[old_index]
        X_test = x.loc[new_index, x.columns]
        y_test = y.loc[new_index]
        return X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=split, random_state = 0)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    
    x = X.copy()
    y = Y.copy()
    X_train, X_test, y_train, y_test = Train_Test_Split(x, y, time=time_split_sample)    
    
    PredictedReturn(y_test,method='longshort',title='Benchmark SP Max Return')
    
    scores_base, ret_base = DecisionTreeBase(x,y)
    
    # decision tree grid search
    clf, scores, ret, feature_importantance = DecisionTreeFun(x,y, n_important_feature=50)
    x_1 = x[feature_importantance.index]
    clf2, scores2, ret2, feature_importantance = DecisionTreeFun(x_1,y, n_important_feature=20)
    x_2 = x[feature_importantance.index]
    clf3, scores3, ret3, feature_importantance = DecisionTreeFun(x_2,y, n_important_feature=10)
    x_3 = x[feature_importantance.index]
    clf3, scores4, ret4, feature_importantance = DecisionTreeFun(x_3,y, n_important_feature=5)
    x_4 = x[feature_importantance.index]
    clf4, scores5, ret5, feature_importantance = DecisionTreeFun(x_4,y, n_important_feature=3)
    score_DT = pd.concat([scores_base, scores, scores2, scores3, scores4, scores5], axis=1)
    score_DT.columns = ['Base','Full','50','20','10','5']
    ret_DT = pd.concat([ret_base, ret, ret2, ret3, ret4, ret5], axis=1)
    ret_DT.columns = score_DT.columns
    
    # Ensemble grid serach
    score_eb, ret_eb = EnsemblingCheck(x,y)
    score_eb1, ret_eb1 = EnsemblingCheck(x_2,y)
    
    # Stacking model search
    score_stk, aum_stk = Stacking(x,y)
    score_stk1, aum_stk1 = Stacking(x_2,y)
    
    
    
