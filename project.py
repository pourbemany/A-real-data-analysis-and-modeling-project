import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns


# %matplotlib inline

# Set random seed
np.random.seed(42)

identifier_feature = ['RESTAURANT_SERIAL_NUMBER']
continuous_features = ['MEDIAN_EMPLOYEE_AGE', 'MEDIAN_EMPLOYEE_TENURE']
nominal_features = ['RESTAURANT_CATEGORY', 'CITY', 'STATE', 'CURRENT_GRADE',
                    'INSPECTION_TYPE','FIRST_VIOLATION', 'SECOND_VIOLATION',
                    'THIRD_VIOLATION','FIRST_VIOLATION_TYPE','SECOND_VIOLATION_TYPE','THIRD_VIOLATION_TYPE']
numeric_feactures = ['CURRENT_DEMERITS', 'EMPLOYEE_COUNT', 'INSPECTION_DEMERITS',
                     'NUMBER_OF_VIOLATIONS']
target = ['NEXT_INSPECTION_GRADE_C_OR_BELOW']
selected_features = nominal_features+ numeric_feactures+ continuous_features+ target

def analysis_(df):
    # shape and df types of the df
    print(df.shape)
    print(df.dtypes)
    
    # Prnit out the unique values of selected_features	
    for i in selected_features:
        print(i)
        tmp = df[i].unique()
        print((tmp))
        print(df[i].value_counts(dropna=False))
        print('\n')
    
    # Null values Handling
    print(df.isnull().values.any()) # Is there any null value?
    print(df.isnull().sum()) # Print the number of null value for each feature
    print('\n')
    
    # Prnit out the unique values of selected_features	
    for i in selected_features:
        print(i)
        tmp = df[i].unique()
        print((tmp))
        print(df[i].value_counts(dropna=False))
        print('\n')

def preprocessing_(df):
    
    # shape and df types of the df
    print(df.shape)
    print(df.dtypes)
    
    # Prnit out the unique values of selected_features	
    for i in selected_features:
        print(i)
        tmp = df[i].unique()
        print((tmp))
        print(df[i].value_counts(dropna=False))
        print('\n')
    
    # Null values Handling
    print(df.isnull().values.any()) # Is there any null value?
    print(df.isnull().sum()) # Print the number of null value for each feature
    print('\n')
    df = df.dropna(how='all') #Drop Row/Column Only if All the Values are Null
    
    # Delete null df
    for i in selected_features:    
         df = df[~df[i].isnull()]
    
    # Text cleaning
    for i in nominal_features:
        if df[i].dtypes==object:
            df[i] = df[i].str.lower()
            df[i] = df[i].str.strip() # remove leading and trailing whitespace.
            df[i] = df[i].str.replace('[^\w\s]','')
            df[i] = df[i].str.replace('\\b \\b','')
            
    # Remove non numeric df from numeric columns        
    for i in numeric_feactures: 
           df[i] = pd.to_numeric(df[i], errors = 'coerce')
           df = df[~pd.to_numeric(df[i], errors='coerce').isnull()]
          # df = df[df[i].str.isnumeric()]
          
    # Get the statistical information
    for i in numeric_feactures:
        print(i)
        print(df[i].describe())
        print('mean', df[i].mean())
        print('median', df[i].median())
        print('mode', df[i].mode())
        # 	print('std', df[i].std())
        print('\n')          

    # Outlier handling     
    df = df[df['NEXT_INSPECTION_GRADE_C_OR_BELOW'].isin(["0", "1"])]     
    if 'CURRENT_GRADE' in selected_features:
        df = df[df['CURRENT_GRADE'].isin(["a", "b", "c", "x", "o", "n"])]
    if 'INSPECTION_TYPE' in selected_features:
        df = df[df['INSPECTION_TYPE'].isin(["routineinspection", "reinspection"])] 
    if 'FIRST_VIOLATION' in selected_features:
        df = df[(0 < df['FIRST_VIOLATION']) &  (df['FIRST_VIOLATION'] < 311)] 
    if 'SECOND_VIOLATION' in selected_features:
        df = df[(0 < df['SECOND_VIOLATION']) &  (df['SECOND_VIOLATION'] < 311)] 
    if 'THIRD_VIOLATION' in selected_features:
        df = df[(0 < df['THIRD_VIOLATION']) &  (df['THIRD_VIOLATION'] < 311)] 
    if 'CURRENT_DEMERITS' in selected_features:
        df = df[(0 <= df['CURRENT_DEMERITS']) &  (df['CURRENT_DEMERITS'] < 200)] 
    if 'EMPLOYEE_COUNT' in selected_features:
        df = df[(0 < df['EMPLOYEE_COUNT']) &  (df['EMPLOYEE_COUNT'] < 100)] 
    if 'STATE' in selected_features:    
        df = df[df['STATE']=='nevada'] 
    
   
    # Prnit out the unique values of selected_features	
    for i in selected_features:
        print(i)
        tmp = df[i].unique()
        print((tmp))
        print(df[i].value_counts(dropna=False))
        print('\n')
    
    # Get the statistical information
    for i in numeric_feactures:
        print(i)
        print(df[i].describe())
        print('mean', df[i].mean())
        print('median', df[i].median())
        print('mode', df[i].mode())
        # 	print('std', df[i].std())
        print('\n')
    
    # Prnit out the first row 	
    print(df.loc[0,numeric_feactures])  
    print('\n')
    
    # X = preprocessing.StandardScaler().fit(df).transform(df)
    df_new = pd.DataFrame()
    # Binarization
    for i in nominal_features:
        dummies = pd.get_dummies(df[i], prefix=i, drop_first=False)
        df_new = pd.concat([df_new, dummies], axis=1)
    # print(df_new.head())
    
    df_disc = pd.DataFrame()
    # Discretization
    for i in continuous_features:
        disc = pd.cut(df[i], bins=10, labels=np.arange(10), right=False)
        df_disc = pd.concat([df_disc, disc], axis=1)
        
    # Concatenate numeric features and discretized features
    for i in numeric_feactures:
        df_disc = pd.concat([df_disc, df[i]], axis=1)    
        
    # Normalization
    x = df_disc.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_norm = pd.DataFrame(x_scaled, columns=df_disc.columns, index=df_disc.index)    
    
    df_new = pd.concat([df_new, df_norm], axis=1)
        
    print('\n')    
    return df, df_new

# Train_Set and Test_Set import, select desired features, and preprocessing
# Train_Set and Test_Set import
df_trn = pd.read_csv('TRAIN_SET_2021.csv', encoding = "ISO-8859-1", usecols = identifier_feature + selected_features, low_memory = False)	
analysis_(df_trn)
df_trn = df_trn.reindex(sorted(df_trn.columns), axis=1)
df_trn['ds_type'] = 'Train'


df_tst = pd.read_csv('TEST_SET_2021.csv', encoding = "ISO-8859-1", low_memory = False)	
df_tst[target] = "0"
df_tst = df_tst[identifier_feature + selected_features]
df_tst = df_tst.reindex(sorted(df_tst.columns), axis=1)
df_tst['ds_type'] = 'Test'

# Concatenate Train and Test set
df = df_trn.append(df_tst)

# Preprocessing
df, df_new = preprocessing_(df)

# Separate Train and Test set
df_tst_ = df[df['ds_type']=='Test']
df = df[df['ds_type']=='Train']

df_new_tst = df_new.iloc[len(df):,:]
df_new = df_new.iloc[:len(df),:]

#***********************************************
# Specify features columns
X = df_new

# Specify target column
y = df['NEXT_INSPECTION_GRADE_C_OR_BELOW']


######################## Visualize the feature correlation
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data=df.astype({'NEXT_INSPECTION_GRADE_C_OR_BELOW': 'int64'}).corr(),
            annot=True, cmap='coolwarm', cbar_kws={'aspect': 50},
            square=True, ax=ax)
plt.xticks(rotation=30, ha='right');
plt.tight_layout()
plt.show()

from scipy.stats import chi2_contingency
def cramers_corrected_stat(contingency_table):
    """
        Computes corrected Cramer's V statistic for categorial-categorial association
    """
    
    try:
        chi2 = chi2_contingency(contingency_table)[0]
    except ValueError:
        return np.NaN
    
    n = contingency_table.sum().sum()
    phi2 = chi2/n
    
    r, k = contingency_table.shape
    r_corrected = r - (((r-1)**2)/(n-1))
    k_corrected = k - (((k-1)**2)/(n-1))
    phi2_corrected = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    
    return (phi2_corrected / min( (k_corrected-1), (r_corrected-1)))**0.5
def categorical_corr_matrix(df):
    """
        Computes corrected Cramer's V statistic between all the
        categorical variables in the dataframe
    """
    df = df.select_dtypes(include='object')
    cols = df.columns
    n = len(cols)
    corr_matrix = pd.DataFrame(np.zeros(shape=(n, n)), index=cols, columns=cols)
    
    excluded_cols = list()
    
    for col1 in cols:
        for col2 in cols:
            if col1 == col2:
                corr_matrix.loc[col1, col2] = 1
                break
            df_crosstab = pd.crosstab(df[col1], df[col2], dropna=False)
            corr_matrix.loc[col1, col2] = cramers_corrected_stat(df_crosstab)
                
    # Flip and add to get full correlation matrix
    corr_matrix += np.tril(corr_matrix, k=-1).T
    return corr_matrix

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(categorical_corr_matrix(df), annot=True, cmap='coolwarm', 
            cbar_kws={'aspect': 50}, square=True, ax=ax)
plt.xticks(rotation=30, ha='right');
plt.tight_layout()
plt.show()



titles = list(df.select_dtypes(include='object'))


# for title in titles:
#     fig, ax = plt.subplots(figsize=(10, 5))
#     sns.countplot(x=title, data=df, palette='Pastel2', ax=ax)
#     ax.set_title(title)
#     ax.set_xlabel('')
#     plt.xticks(rotation=30, ha='right');
#     plt.tight_layout()
#     plt.show()

################ Train-Test splitting
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit 
RS = 15

# # Split dataframe into training and test/validation set 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RS)

splitter=StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=RS) 
for train,test in splitter.split(X,y):     #this will splits the index
    X_train = X.iloc[train]
    y_train = y.iloc[train]
    X_test = X.iloc[test]
    y_test = y.iloc[test]

# Visualize the classes distributions
sns.countplot(y_train).set_title("Outcome Count")
plt.show()

# summarize the new class distribution
from collections import Counter
counter = Counter(y_train)
print(counter)

# ############## Under_Sampling
# # Import required library for resampling
# from imblearn.under_sampling import RandomUnderSampler

# # Instantiate Random Under Sampler
# rus = RandomUnderSampler(random_state=42)

# # Perform random under sampling
# X_train, y_train = rus.fit_resample(X_train, y_train)

# # Visualize new classes distributions
# sns.countplot(y_train).set_title('Balanced Data Set - Under-Sampling')
# plt.show()

# ############## Over_Sampling
# # define oversampling strategy
from imblearn.over_sampling import SMOTE,SVMSMOTE,ADASYN,BorderlineSMOTE,RandomOverSampler


# transform the dataset
# oversample = RandomOverSampler(sampling_strategy=0.5)
# X_train, y_train = oversample.fit_resample(X_train, y_train)

# oversample = SMOTE(sampling_strategy=0.5)
# X_train, y_train = oversample.fit_resample(X_train, y_train)

oversample = BorderlineSMOTE(sampling_strategy=0.5)
X_train, y_train = oversample.fit_resample(X_train, y_train)

# oversample = SVMSMOTE(sampling_strategy=0.5)
# X_train, y_train = oversample.fit_resample(X_train, y_train)

# oversample = ADASYN(sampling_strategy=0.5)
# X_train, y_train = oversample.fit_resample(X_train, y_train)

# Visualize new classes distributions
sns.countplot(y_train).set_title('Balanced Data Set - Over-Sampling')
plt.show()


# summarize the new class distribution
counter = Counter(y_train)
print(counter)

######################### Modelling
# Import required library for modeling
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from xgboost import XGBClassifier
import xgboost

# Evaluating different classifiers
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    # NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    XGBClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    MLPClassifier(hidden_layer_sizes=(64,64,64), activation='relu', solver='adam', max_iter=500),
    LogisticRegression(random_state=0, class_weight='balanced')    
    ]

# Logging for Visual Comparison
log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)

for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('****Results****')
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Accuracy: {:.4%}".format(acc))
    
    print(confusion_matrix(y_test, train_predictions))
    print(classification_report(y_test,train_predictions))
    
    train_predictions = clf.predict_proba(X_test)
    ll = log_loss(y_test, train_predictions)
    print("Log Loss: {}".format(ll))
    
    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
    log = log.append(log_entry)
   
print("="*30)


# Visual Comparison of different classifier
sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")

plt.xlabel('Accuracy %')
plt.title('Classifier Accuracy')
plt.show()

sns.set_color_codes("muted")
sns.barplot(x='Log Loss', y='Classifier', data=log, color="g")

plt.xlabel('Log Loss')
plt.title('Classifier Log Loss')
plt.show()




# # Inspect the learned Decision Trees
# clf = DecisionTreeClassifier()

# # Fit with all the training set
# clf.fit(X, y)

# # Investigate feature importance
# importances = clf.feature_importances_
# indices = np.argsort(importances)[::-1]
# feature_names = X.columns

# print("Feature ranking:")
# for f in range(X.shape[1]):
#     print("%s : (%f)" % (feature_names[f] , importances[indices[f]]))
    
    
    
    
############################## Select the best classifier for prediction    
# clf = RandomForestClassifier()
# # Fit with all the training set
# clf.fit(X_train,y_train)    
    
test_predictions = clf.predict(df_new_tst) 
test_predictions_proba = clf.predict_proba(df_new_tst)  

df_tst_[target] = test_predictions
df_tst_['Predictions_proba'] = test_predictions_proba.max(axis=1)



# Add predicted value and thir probability to the original TEST_Set
# I did not considered rows with missing values for the predictions (there are 11 rows that have null value)
# Finally, I consider "0" for them as the prediction
df_tst['Predictions_proba'] = "1"
df_tst.loc[df_tst_.index,target]=df_tst_[target]
df_tst.loc[df_tst_.index,'Predictions_proba']=df_tst_['Predictions_proba']


# save the desired columns to a csv file
df = pd.DataFrame()
df = df_tst[['RESTAURANT_SERIAL_NUMBER', 'Predictions_proba', 'NEXT_INSPECTION_GRADE_C_OR_BELOW']]
df.columns = ['RESTAURANT_SERIAL_NUMBER', 'CLASSIFIER_PROBABILITY', 'CLASSIFIER_PREDICTION']
df['CLASSIFIER_PROBABILITY'] = pd.to_numeric(df['CLASSIFIER_PROBABILITY'])
df['CLASSIFIER_PREDICTION'] = pd.to_numeric(df['CLASSIFIER_PREDICTION'])
df.to_csv('predictions_Pourbemany_Jafar_Intern.csv', index = False)