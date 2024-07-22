from sklearn.metrics import r2_score
from sklearn.model_selection import KFold 
from sklearn.feature_selection import RFE

def recursive_select_with_cross(x_train, x_test, y_train, y_test, feature_num, selector, step=1):
    '''
    function to do recursive selection takes the training and testing sets, number and features and the sklearn estimator you want
    returns the features selected and its r2 value
    '''

    if feature_num >= x_train.shape[1]: return "Error, too many features for this set"
    x_train_c, x_test_c, y_train_c, y_test_c = x_train.copy(), x_test.copy(), y_train.copy(), y_test.copy()

    RFE_selector = RFE(estimator=selector, n_features_to_select=feature_num, step=step)
    RFE_selector = RFE_selector.fit(x_train_c, y_train_c)

    sel_x_train_c = RFE_selector.transform(x_train_c)
    sel_x_test_c = RFE_selector.transform(x_test_c)

    selector.fit(sel_x_train_c, y_train_c)
    r2_preds = selector.predict(sel_x_test_c)
    
    r2 = round(r2_score(y_test_c, r2_preds), 3)
    
    return((x_train_c.columns[RFE_selector.support_], r2))

def recursive_select_feature_set_with_cross(x_train, y_train, selector, step=1, k=5):
    '''
    returns a list of selected features and r2 values for each possible feature size using a given sklearn estimator
    uses recursive_select() with cross validation
    '''
    feature_range = list()
    for i in range(x_train.shape[1] - 1):
        feature_range.append(i + 1)

    results = list()
    for num in feature_range:
        results.append(cross_val_kfolds(x_train, y_train, num, selector, step, k))

    return results

def cross_val_kfolds(x_train_org, y_train_org, num, selector, step, k):

    kf = KFold(n_splits=k, shuffle=True, random_state=21)

    feature_set = ''
    r2_scores_max = 0

    for i_train, i_test in kf.split(x_train_org, y_train_org):

        x_train = x_train_org.drop(x_train_org.index[i_test])
        x_test = x_train_org.drop(x_train_org.index[i_train])
        y_train = y_train_org.drop(y_train_org.index[i_test])
        y_test = y_train_org.drop(y_train_org.index[i_train])

        feature_set, r2 = recursive_select_with_cross(x_train, x_test, y_train, y_test, num, selector, step)
        if r2 > r2_scores_max: r2_scores_max = r2
    
    return feature_set, r2_scores_max