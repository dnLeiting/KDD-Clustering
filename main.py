#!/usr/bin/env python
# coding: utf-8

# In[98]:


import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
import pandas as pd
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# In[172]:


data = pd.read_csv('kddcup.data/kddcup99.csv')
print(len(data))
data.head()


# In[173]:


# One hot encoding
def one_hot_top_x(df, variable, x_labels):
    column = df.columns.get_loc(variable)
    top_x_labels = [x for x in df.iloc[:, column].value_counts(ascending=False).sort_values().tail(x_labels).index]
    for label in top_x_labels:
        df[variable + '_' + label] = np.where(data[variable] == label, 1, 0)
    df = df.drop(variable, axis=1)
    return df


for column_name in ['service', 'protocol_type', 'flag']:
    data = one_hot_top_x(data, column_name, 10)

data.head()

# In[174]:


# Get all features
columns = []
for col in data.columns.drop('label'):
    columns.append(str(col))

# Normalization
min_max = MinMaxScaler()
data[columns] = min_max.fit_transform(data[columns])

# In[175]:


train_ds, test_ds = train_test_split(data, test_size=0.2, random_state=42)
print('Training data length', len(train_ds))
print('Test data length', len(test_ds))
print('Training data length', len(train_ds), 'with',
      "{:.2f}".format(train_ds.label.value_counts()['normal'] / len(train_ds) * 100), '% normal values')
print('Training data length', len(test_ds), 'with',
      "{:.2f}".format(test_ds.label.value_counts()['normal'] / len(test_ds) * 100), '% normal values')

# In[176]:


data_old = data
train_ds_old = test_ds
test_ds_old = test_ds

# Normal Data
attack_data = data[data['label'] != 'normal']
normal_data = data[data['label'] == 'normal']

normal_label = normal_data['label']
normal_data = normal_data.drop('label', axis=1)

label_attack_data = attack_data['label']
attack_data = attack_data.drop('label', axis=1)

label = data['label']
data = data.drop('label', axis=1)

# Train Data Set
attack_train_ds = train_ds[train_ds['label'] != 'normal']
normal_train_ds = train_ds[train_ds['label'] == 'normal']

normal_label_train_ds = normal_train_ds['label']
normal_data_train_ds = normal_train_ds.drop('label', axis=1)

attack_label_train_ds = attack_train_ds['label']
attack_data_train_ds = attack_train_ds.drop('label', axis=1)

label_train_ds = train_ds['label']
data_train_ds = train_ds.drop('label', axis=1)

# Test Data Set
attack_test_ds = test_ds[test_ds['label'] != 'normal']
normal_test_ds = test_ds[test_ds['label'] == 'normal']

normal_label_test_ds = normal_test_ds['label']
normal_data_test_ds = normal_test_ds.drop('label', axis=1)

attack_label_test_ds = attack_test_ds['label']
attack_data_test_ds = attack_test_ds.drop('label', axis=1)

label_test_ds = test_ds['label']
data_test_ds = test_ds.drop('label', axis=1)


# In[177]:


def compare(test_data_predicted, cluster, value):
    for i in range(len(test_data_predicted)):
        if (test_data_predicted[i] == cluster):
            test_data_predicted[i] = value
    return test_data_predicted


def reshape_cluster(train_data_predicted, test_data_predicted, cluster_numbers, labels_normal):
    train_data_predicted_output = train_data_predicted
    for cluster in range(cluster_numbers):
        # Transform training data
        labels_kmean_train_cluster = np.where(train_data_predicted == cluster, 1, 2)
        matches = np.count_nonzero(labels_kmean_train_cluster == labels_normal)
        unique, counts = np.unique(labels_kmean_train_cluster, return_counts=True)
        mismatches = labels_kmean_train_cluster.size - matches
        print('matches', matches, 'counts', counts[0], 'percentage', matches / counts[0])
        if (0.5 <= matches/counts[0]):
            print('Cluster', cluster, ' is a normal cluster')
            test_data_predicted = compare(test_data_predicted, cluster, 1)
            train_data_predicted_output = compare(train_data_predicted_output, cluster, 1)
        else:
            print('Cluster', cluster, ' is not a normal cluster')
            test_data_predicted = compare(test_data_predicted, cluster, 0)
            train_data_predicted_output = compare(train_data_predicted_output, cluster, 0)
    return test_data_predicted, train_data_predicted_output


k_mean = 15
k_mean_start = 2
init = 10
distorsions = []
y_kmeans_all_test_results = []
y_kmeans_all_train = []
y_kmeans_all_train_results = []
loop = 10
label_train_ds = np.where(label_train_ds[:] == 'normal', 1, 0)
for k in range(k_mean_start, k_mean):
    kmeans = KMeans(n_clusters=k, n_init=init)
    if (k == k_mean_start):
        kmeans.fit(data_train_ds)
        y_kmeans_all_train.append(kmeans.predict(data_train_ds))
        test_data_predicted = kmeans.predict(data_test_ds)
        test_data_predicted, train_data_predicted = reshape_cluster(y_kmeans_all_train[k - k_mean_start],
                                                                    test_data_predicted, k, label_train_ds)
        y_kmeans_all_test_results.append(test_data_predicted)
        y_kmeans_all_train_results.append(train_data_predicted)
    else:
        kmeans.fit(data_train_ds)
        y_kmeans_all_train.append(kmeans.predict(data_train_ds))
        test_data_predicted = kmeans.predict(data_test_ds)
        test_data_predicted, train_data_predicted = reshape_cluster(y_kmeans_all_train[k - k_mean_start],
                                                                    test_data_predicted, k, label_train_ds)
        y_kmeans_all_test_results = np.vstack((y_kmeans_all_test_results, test_data_predicted))
        y_kmeans_all_train_results = np.vstack((y_kmeans_all_train_results, train_data_predicted))
    distorsions.append(kmeans.inertia_)
fig = plt.figure(figsize=(15, 5))
plt.plot(range(k_mean_start, k_mean), distorsions)
plt.grid(True)
plt.title('Elbow curve')
plt.savefig('Elbow_curve.png')


# In[198]:


def compate_results(x, y, compare_value):
    count = 0
    base = 0
    for i in range(len(x)):
        if (x[i] == compare_value and y[i] == compare_value):
            count = count + 1
        if (x[i] == compare_value):
            base = base + 1
    return round(count / base, 3)


def tp(x, y, compare_value):
    count = 0
    for i in range(len(x)):
        if (x[i] == compare_value and y[i] == compare_value):
            count = count + 1
    return count


def fp(x, y, compare_value):
    count = 0
    for i in range(len(x)):
        if (x[i] != compare_value and y[i] == compare_value):
            count = count + 1
    return count


def fn(x, y, compare_value):
    count = 0
    for i in range(len(x)):
        if (x[i] == compare_value and y[i] != compare_value):
            count = count + 1
    return count


def tn(x, y, compare_value):
    count = 0
    for i in range(len(x)):
        if (x[i] != compare_value and y[i] != compare_value):
            count = count + 1
    return count


def normal_values_total(x, y, compare_value):
    count = 0
    base = 0
    for i in range(len(x)):
        if (x[i] == compare_value and y[i] == compare_value):
            count = count + 1
        if (y[i] == compare_value):
            base = base + 1
    if (base == 0):
        return 0
    return (count / base)


def values(x, y, compare_value):
    tp_value = tp(x, y, compare_value)
    fp_value = fp(x, y, compare_value)
    fn_value = fn(x, y, compare_value)
    tn_value = tn(x, y, compare_value)
    accuracy = (tp_value + tn_value) / len(x)
    fone = round(f1_score(x, y, average='weighted'), 3)
    result = {
        "accuracy": accuracy,
        "fone": fone,
        "sensitivity": tp_value / (tp_value + fn_value),
        "specificity": tn_value / (tn_value + fp_value),
        "tp_value": tp_value,
        "fp_value": fp_value,
        "fn_value": fn_value,
        "tn_value": tn_value,
    }
    return result


# In[200]:


results_train = []
results_test = []

label_test_ds_t = np.where(label_test_ds[:] == 'normal', 1, 0)
for k in range(0, (k_mean - k_mean_start)):
    print('KMean for K=', (k + k_mean_start))
    results_train.append(values(label_train_ds, y_kmeans_all_train_results[k], 1))
    results_test.append(values(label_test_ds_t, y_kmeans_all_test_results[k], 1))
    print(results_train[k])
    print(results_test[k])


# In[224]:


def get_values(data, value):
    result = []
    for i in range(len(data)):
        result.append(data[i][value])
    return result


# In[228]:


fig = plt.figure(figsize=(15, 5))
train_accuracy = get_values(results_train, 'accuracy')
train_test = get_values(results_test, 'accuracy')
plt.plot(range(k_mean_start, k_mean), train_accuracy)
plt.plot(range(k_mean_start, k_mean), train_test)
plt.grid(True)
plt.title('Acurracy')
plt.savefig('acurracy_all.png')

# In[230]:


fig = plt.figure(figsize=(15, 5))
train_result = get_values(results_train, 'fone')
test_result = get_values(results_test, 'fone')
plt.plot(range(k_mean_start, k_mean), train_result)
plt.plot(range(k_mean_start, k_mean), test_result)
plt.grid(True)
plt.title('F1')
plt.savefig('fone_all.png')

# In[229]:


# TPR = Sensivitivity
fig = plt.figure(figsize=(15, 5))
train_result = get_values(results_train, 'sensitivity')
test_result = get_values(results_test, 'sensitivity')
plt.plot(range(k_mean_start, k_mean), train_result)
plt.plot(range(k_mean_start, k_mean), test_result)
plt.grid(True)
plt.title('Sensitivity')
plt.savefig('sensitivity_all.png')

# In[233]:


# TNR = Sensivitivity
fig = plt.figure(figsize=(15, 5))
train_result = get_values(results_train, 'specificity')
test_result = get_values(results_test, 'specificity')
plt.plot(range(k_mean_start, k_mean), train_result)
plt.plot(range(k_mean_start, k_mean), test_result)
plt.grid(True)
plt.title('Specificity')
plt.savefig('specificity_all.png')

