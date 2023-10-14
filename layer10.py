import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import multiprocessing
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance, GridSearchCV

def evaluate_model(model, test_x, test_y):
    pred_y = model.predict(test_x)
    accuracy = accuracy_score(test_y, pred_y)
    print(f"Accuracy: {accuracy:.2f}")
    scores = cross_val_score(model, test_x, test_y, cv=5)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

# Fetching the data
train = pd.read_csv('./speech-based-classification-layer-10/train.csv')
valid = pd.read_csv('./speech-based-classification-layer-10/valid.csv')
test = pd.read_csv('./speech-based-classification-layer-10/test.csv')

# Selecting the features based on correlation
corr_matrix1 = train.drop(['label_1', 'label_2', 'label_3', 'label_4'], axis=1).corr()
upper = corr_matrix1.where(np.triu(np.ones(corr_matrix1.shape), k=1).astype(np.bool_))
features_to_drop = [column for column in upper.columns if any(upper[column] > 0.7)]

train.drop(columns=features_to_drop, axis=1, inplace=True)
valid.drop(columns=features_to_drop, axis=1, inplace=True)
test.drop(columns=features_to_drop, axis=1, inplace=True)

# Organizing the train and dev data
train1 = train.drop(['label_2', 'label_3', 'label_4'], axis=1)
train2 = train.drop(['label_1', 'label_3', 'label_4'], axis=1)
train3 = train.drop(['label_1', 'label_2', 'label_4'], axis=1)
train4 = train.drop(['label_1', 'label_2', 'label_3'], axis=1)

valid1 = valid.drop(['label_2', 'label_3', 'label_4'], axis=1)
valid2 = valid.drop(['label_1', 'label_3', 'label_4'], axis=1)
valid3 = valid.drop(['label_1', 'label_2', 'label_4'], axis=1)
valid4 = valid.drop(['label_1', 'label_2', 'label_3'], axis=1)

# Dataframe to store the results
result = pd.DataFrame()
result['ID'] = test['ID']
test.drop(['ID'], axis=1, inplace=True)

test1 = test.copy()
test2 = test.copy()
test3 = test.copy()
test4 = test.copy()

le = LabelEncoder()
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3]  
}

# Processing label 1

# Inspecting the class distribution
print(train1['label_1'].value_counts())

# Initial Model
label1 = train1['label_1']
label1_encoded = le.fit_transform(label1)
label1_model = xgb.XGBClassifier(n_estimators=50, max_depth=2, n_jobs=multiprocessing.cpu_count(), tree_method="hist").fit(train1.drop(['label_1'], axis=1), label1_encoded)
evaluate_model(label1_model, valid1.drop(['label_1'], axis=1), le.transform(valid1['label_1']))

# Permutation importance
permutation_importances = permutation_importance(label1_model, train1.drop(['label_1'], axis=1), label1_encoded, n_repeats=2)
feature_names = train1.drop(['label_1'], axis=1).columns
feature_importances = permutation_importances.importances_mean
features_to_drop = feature_names[feature_importances < 0]

print(features_to_drop)
print("Number of features to drop = ", len(features_to_drop))

train1.drop(features_to_drop, axis=1, inplace=True)
valid1.drop(features_to_drop, axis=1, inplace=True)
test1.drop(features_to_drop, axis=1, inplace=True)

# Hyper parameter tuning
model1 = xgb.XGBClassifier(n_estimators=50, max_depth=2, n_jobs=multiprocessing.cpu_count(), tree_method="hist")
grid_search = GridSearchCV(model1, param_grid, cv=5)
grid_search.fit(train1.drop(['label_1'], axis=1), label1_encoded)
best_model1 = grid_search.best_estimator_
evaluate_model(best_model1, valid1.drop(['label_1'], axis=1), le.transform(valid1['label_1']))

predictions = best_model1.predict(test1)
result['label_1'] = le.inverse_transform(predictions)


# Processing label 2

# Dropping NaN values
train2.dropna(inplace=True)
valid2.dropna(inplace=True)

# Inspecting the class distribution
print(train2['label_2'].value_counts())

# Initial Model
label2 = train2['label_2']
label2_encoded = le.fit_transform(label2)
label2_model = xgb.XGBClassifier(n_estimators=50, max_depth=2, n_jobs=multiprocessing.cpu_count(), tree_method="hist").fit(train2.drop(['label_2'], axis=1), label2_encoded)
evaluate_model(label2_model, valid2.drop(['label_2'], axis=1), le.transform(valid2['label_2']))

# Permutation importance
permutation_importances = permutation_importance(label2_model, train2.drop(['label_2'], axis=1), label2_encoded, n_repeats=2)
feature_names = train2.drop(['label_2'], axis=1).columns
feature_importances = permutation_importances.importances_mean
features_to_drop = feature_names[feature_importances < 0]

print(features_to_drop)
print("Number of features to drop = ", len(features_to_drop))

train2.drop(features_to_drop, axis=1, inplace=True)
valid2.drop(features_to_drop, axis=1, inplace=True)
test2.drop(features_to_drop, axis=1, inplace=True)

# Hyper parameter tuning
model2 = xgb.XGBClassifier(n_estimators=50, max_depth=2, n_jobs=multiprocessing.cpu_count(), tree_method="hist")
grid_search = GridSearchCV(model2, param_grid, cv=5)
grid_search.fit(train2.drop(['label_2'], axis=1), label2_encoded)
best_model2 = grid_search.best_estimator_
evaluate_model(best_model2, valid2.drop(['label_2'], axis=1), le.transform(valid2['label_2']))

predictions = best_model2.predict(test2)
result['label_2'] = le.inverse_transform(predictions)


# Processing label 3

# Inspecting the class distribution
print(train3['label_3'].value_counts())

# Initial Model
label3 = train3['label_3']
label3_encoded = le.fit_transform(label3)
label3_model = xgb.XGBClassifier(n_estimators=50, max_depth=2, n_jobs=multiprocessing.cpu_count(), tree_method="hist").fit(train3.drop(['label_3'], axis=1), label3_encoded)
evaluate_model(label3_model, valid3.drop(['label_3'], axis=1), le.transform(valid3['label_3']))

# Permutation importance
permutation_importances = permutation_importance(label3_model, train3.drop(['label_3'], axis=1), label3_encoded, n_repeats=2)
feature_names = train3.drop(['label_3'], axis=1).columns
feature_importances = permutation_importances.importances_mean
features_to_drop = feature_names[feature_importances < 0]

print(features_to_drop)
print("Number of features to drop = ", len(features_to_drop))

train3.drop(features_to_drop, axis=1, inplace=True)
valid3.drop(features_to_drop, axis=1, inplace=True)
test3.drop(features_to_drop, axis=1, inplace=True)

# Hyper parameter tuning
model3 = xgb.XGBClassifier(n_estimators=50, max_depth=2, n_jobs=multiprocessing.cpu_count(), tree_method="hist")
grid_search = GridSearchCV(model3, param_grid, cv=5)
grid_search.fit(train3.drop(['label_3'], axis=1), label3_encoded)
best_model3 = grid_search.best_estimator_
evaluate_model(best_model3, valid3.drop(['label_3'], axis=1), le.transform(valid3['label_3']))

predictions = best_model3.predict(test3)
result['label_3'] = le.inverse_transform(predictions)


# Processing label 4

# Inspecting the class distribution
print(train4['label_4'].value_counts())

# Initial Model
label4 = train4['label_4']
label4_encoded = le.fit_transform(label4)
label4_model = xgb.XGBClassifier(n_estimators=50, max_depth=2, n_jobs=multiprocessing.cpu_count(), tree_method="hist").fit(train4.drop(['label_4'], axis=1), label4_encoded)
evaluate_model(label4_model, valid4.drop(['label_4'], axis=1), le.transform(valid4['label_4']))

# Permutation importance
permutation_importances = permutation_importance(label4_model, train4.drop(['label_4'], axis=1), label4_encoded, n_repeats=2)
feature_names = train4.drop(['label_4'], axis=1).columns
feature_importances = permutation_importances.importances_mean
features_to_drop = feature_names[feature_importances < 0]

print(features_to_drop)
print("Number of features to drop = ", len(features_to_drop))

train4.drop(features_to_drop, axis=1, inplace=True)
valid4.drop(features_to_drop, axis=1, inplace=True)
test4.drop(features_to_drop, axis=1, inplace=True)

# Hyper parameter tuning
model4 = xgb.XGBClassifier(n_estimators=50, max_depth=2, n_jobs=multiprocessing.cpu_count(), tree_method="hist")
grid_search = GridSearchCV(model4, param_grid, cv=5)
grid_search.fit(train4.drop(['label_4'], axis=1), label4_encoded)
best_model4 = grid_search.best_estimator_
evaluate_model(best_model4, valid4.drop(['label_4'], axis=1), le.transform(valid4['label_4']))

predictions = best_model4.predict(test4)
result['label_4'] = le.inverse_transform(predictions)

# Writing to final csv
result.to_csv('./speech-based-classification-layer-10/pred-labels.csv', index=False)