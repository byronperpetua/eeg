import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.linear_model
import sklearn.metrics
import scipy.stats

# DATA PREPARATION

data = pd.read_csv('data.csv')
data['chunk'] = data['id'].str.extract(r'X(\d*)\..*', expand=False).astype(int)
data['indiv'] = data['id'].str.extract(r'X\d*\.(.*)', expand=False)
data['seizure'] = data['y'] == 1
data = data.drop(['id', 'y'], axis=1)

long = pd.wide_to_long(data, stubnames='X', i=['indiv', 'chunk'], j='reading')
long.index.set_levels(long.index.get_level_values('reading').astype(int),
                      level='reading', inplace=True)
long = long.sort_index()
by_chunk = long.groupby(['indiv', 'chunk']).mean()
indivs = data[['indiv', 'seizure']].drop_duplicates().reset_index()

# Check that each individual has only one label
temp = indivs['indiv'].drop_duplicates()
assert len(temp) == len(indivs)
del temp

# PLOTS

# for r in indivs.iterrows():
#     indiv = r[1]['indiv']
#     seizure = r[1]['seizure']
#     print(indiv)

    # Plots of readings in chunk 5 by individual
    # chunk = 5
    # plt.plot(long.loc[(indiv, chunk), :]['X'])
    # plt.ylim(-1000, 1000)
    # plt.savefig('images/chunk5/' + str(seizure) + indiv + '.png')
    # plt.clf()

    # FALSE: flat; moderate swings without periodicity; single spike
    # TRUE: large swings, moderate swings with periodicity

    # Plots of readings of multiple chunks for select individuals
    # if r[0] % 10 == 0:
    #     for chunk in (10, 11, 12, 15, 20):
    #         plt.plot(long.loc[(indiv, chunk), :]['X'])
    #         plt.ylim(-1000, 1000)
    #         plt.savefig('images/mult_chunks/' + str(seizure) + indiv + '_'
    #                     + str(chunk) + '.png')
    #         plt.clf()
    
    # Patterns are similar across chunks within individuals.

    # Plots of chunk averages by individual
    # plt.plot(by_chunk.loc[indiv, :]['X'])
    # plt.ylim(-100, 100)
    # plt.savefig('images/by_chunk/' + str(seizure) + indiv + '.png')
    # plt.clf()

    # Per above, I don't think this is helpful.

# We should be able to correctly classify at least 95% of non-seizures and 90%
# of seizures. A simple classifier would use the difference between the ~95th and
# ~5th percentile of readings for each individual as the only feature.

# CLASSIFICATION

def range_(x):
    return np.max(x) - np.min(x)

def slope(x):
    return scipy.stats.linregress(range(len(x)), x)[0]

def slope_first_half(x):
    return slope(x[:int(len(x)/2)])

def slope_second_half(x):
    return slope(x[int(len(x)/2):])

X = pd.pivot_table(long, index=['indiv', 'chunk', 'seizure'], values='X',
                   aggfunc=[np.mean, np.std, range_, slope,
                            slope_first_half, slope_second_half])
y = pd.pivot_table(long, index=['indiv', 'chunk'], values='seizure')

# k-fold cross validation, grouped by individual and stratified by class
skf = sklearn.model_selection.StratifiedKFold(n_splits=5)
lr = sklearn.linear_model.LogisticRegression()
precision = []
recall = []
for train, test in skf.split(indivs, indivs['seizure']):
    X_train = X.loc[indivs.iloc[train]['indiv']]
    X_test = X.loc[indivs.iloc[test]['indiv']]
    y_train = y.loc[indivs.iloc[train]['indiv']]
    y_test = y.loc[indivs.iloc[test]['indiv']]
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    precision.append(sklearn.metrics.precision_score(y_test, y_pred))
    recall.append(sklearn.metrics.recall_score(y_test, y_pred))
    print(sklearn.metrics.confusion_matrix(y_test, y_pred))
print("Precision: {}".format(np.mean(precision)))
print("Recall: {}".format(np.mean(recall)))

# Precision: 93.3%, Recall: 81.6%
# Mean adds nothing. Either SD or range is essential, but not both.
# Slopes add nothing.