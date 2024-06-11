#!/usr/bin/env python
# coding: utf-8

# # About Dataset

# The dataset provides insights into customer satisfaction levels within an undisclosed airline company. 
# While the specific airline name is withheld, the dataset is rich in information, containing 22 columns 
# and 129,880 rows. It aims to predict whether future customers will be satisfied based on various parameters
# included in the dataset.
# 
# The columns likely cover a range of factors that influence customer satisfaction, such as flight punctuality, 
# service quality, and so. By analyzing this dataset, airlines can gain valuable insights into the factors that 
# contribute to customer satisfaction and tailor their services accordingly to enhance the overall customer experience.
# 

# - **Satisfaction:** Indicates the satisfaction level of the customer.
# - **Customer Type:** Type of customer: 'Loyal Customer' or 'Disloyal Customer’.
# - **Age:** Age of the customer.
# - **Type of Travel:** Purpose of the travel: 'Business travel' or 'Personal Travel’.
# - **Class:**	Class of travel: 'Business', 'Eco', or 'Eco Plus’.
# - **Flight Distance:** The distance of the flight in kilometres
# - **Seat comfort:** Rating of seat comfort provided during the flight (1 to 5).
# - **Departure/Arrival time convenient** Rating of the convenience of departure/arrival time (1 to 5).
# - **Food and drink:** Rating of food and drink quality provided during the flight (1 to 5).
# - **Gate location:**	Rating of gate location convenience (1 to 5).
# - **Inflight wifi service:**	Rating of inflight wifi service satisfaction (1 to 5).
# - **Inflight entertainment:** Rating of inflight entertainment satisfaction (1 to 5).
# - **Online support:** Rating of online customer support satisfaction (1 to 5).
# - **Ease of Online booking:** Rating of ease of online booking satisfaction (1 to 5).
# - **On-board service:** Rating of on-board service satisfaction (1 to 5).
# - **Leg room service:** Rating of leg room service satisfaction (1 to 5).
# - **Baggage handling:** Rating of baggage handling satisfaction (1 to 5).
# - **Checkin service:** Rating of check-in service satisfaction (1 to 5).
# - **Cleanliness:** Rating of cleanliness satisfaction (1 to 5).
# - **Online boarding:** Rating of online boarding satisfaction (1 to 5).
# - **Departure Delay in Minutes:** Total departure delay in minutes.
# - **Arrival Delay in Minutes:** Total arrival delay in minutes.

# # IMPORTING LIBRARIES

# In[147]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sklearn. metrics as metrics
import category_encoders as ce


# # FUNCTIONS FOR PLOT CUSTOMIZATION

# In[148]:


def set_size_style(width, height, style=None):
    plt.figure(figsize=(width, height))
    if style != None:
        sns.set_style(style)

def customize_plot(plot, title:str, xlabel:str,  ylabel:str, title_font:int, label_font:int):
    plot.set_title(title, fontsize = title_font, weight='bold')
    plot.set_xlabel(xlabel, fontsize = label_font, weight='bold')
    plot.set_ylabel(ylabel, fontsize = label_font, weight='bold')


# # DATA EXPLORATION & CLEANING

# In[149]:


data=pd.read_csv(r"Airline_customer_satisfaction.csv")
customer_df= data
print(customer_df)


# In[150]:


customer_df.isnull().sum()


# In[151]:


customer_df['Arrival Delay in Minutes'].fillna(customer_df['Arrival Delay in Minutes'].mean(),inplace=True)


# In[152]:


customer_df.columns


# In[153]:


customer_df.describe()


# In[154]:


customer_df.duplicated().sum()


# # OUTLIER ANALYSIS 
# 

# In[155]:


fig,axs=plt.subplots(2,3,figsize =(10,5))
plt1=sns.boxplot(customer_df['Age'],ax=axs[0,0])
plt2=sns.boxplot(customer_df['Online boarding'],ax = axs[0,1])
plt.tight_layout()


# In[156]:


plt.boxplot(data.Age)
Q1=data.Age.quantile(0.25)
Q3=data.Age.quantile(0.75)
IQR=Q3-Q1
Airline = data[(data.Age >=Q1-1.5*IQR) &(data.Age <= Q3+1.5*IQR)]


# In[157]:


plt.boxplot(customer_df.Cleanliness)
Q1=customer_df.Cleanliness.quantile(0.25)
Q3=customer_df.Cleanliness.quantile(0.75)
IQR = Q3-Q1
Airline = customer_df[(customer_df.Cleanliness >=Q1-1.5*IQR) &(customer_df.Cleanliness <= Q3+1.5*IQR)]


# # EXPLOTARY DATA ANALYSIS

# In[158]:


plt.title("Satisfied vs Dissatisfied", fontsize = 12, weight='bold')
plt.pie(customer_df['satisfaction'].value_counts(),labels=customer_df['satisfaction'].value_counts().index,radius=1,
        autopct='%.2f%%',textprops={'fontsize': 10, 'fontweight': 'bold'}, colors = sns.color_palette('Spectral'))
plt.show()


# .The number of satisfied customers exceeds the number of dissatisfied customers, indicating a prevailing trend towards
# positive experiences with the service or product.

# In[159]:


set_size_style(10,5)
ax = sns.histplot(customer_df['Age'],bins=25,color= sns.color_palette('Spectral')[0],kde=True)
customize_plot(ax,'Age Distribution','Age','Frequency',13,10)


# .The number of satisfied customers exceeds the number of dissatisfied customers,indicating a prevailing trend towards
#  positive experiences with the service or product.

# In[160]:


Airline


# In[161]:


#split into x and y
x=Airline.drop(columns='Age')
y=Airline['Age']
x


# In[162]:


y


# In[163]:


x.info()


# In[164]:


encoder = ce.LeaveOneOutEncoder()
x=encoder.fit_transform(x,y)


# In[165]:


x.info()


# In[166]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.20)


# In[167]:


randomforest = RandomForestRegressor(n_estimators = 7)
decisionforest = DecisionTreeRegressor()
linear = LinearRegression()


# In[168]:


models =[randomforest,decisionforest,linear]


# In[169]:


for model in models:
    print(f"fitting model:{model}")
    model.fit(x_train,y_train)


# In[170]:


#measuring the accuarcy of the model against the train data - score

for model in models:
    print(f" score of {model} for training data: {model.score(x_train,y_train)}")
    


# In[171]:


#feature importance to establish the importance of each feature in decision making process
fs = randomforest.feature_importances_
feature_names = x.columns


# In[172]:


feature_importances = pd.DataFrame(fs,feature_names).sort_values(by=0,ascending=False)
plt.figure(figsize=(12, 9))
plt.title("Feature Importances")
plt.bar(x=feature_importances.index,height=feature_importances[0])
plt.xticks(rotation=90)
plt.show()


# In[173]:


feature_importances


# In[174]:


#train and test using top 5 features:

def regression_results(y_true, y_pred):

    # Regression metrics
    #explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    #median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)

    #print('explained_variance: ', round(explained_variance,4))    
    #print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))
    #print('Median absolute error: ',round(median_absolute_error,4))


# In[175]:


for model in models[:]:
    y_predicted = model.predict(x_test)

    print(f"Report:{model}")
    print(f"{regression_results(y_test, y_predicted)}\n")
    


# # IMPORTING LIBRARIES

# In[176]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score,ConfusionMatrixDisplay, classification_report
import warnings
warnings.filterwarnings('ignore')


# In[129]:


pip install xgboost


# In[177]:


customer_df.shape


# In[69]:


customer_df.info()


# In[178]:


customer_df.describe()


# In[179]:


for col in customer_df.describe(include='object').columns:
    print('Column Name: ',col)
    print(customer_df[col].unique())
    print('-'*50)


# In[180]:


customer_df.isna().sum()


# # HANDLING OUTLIERS

# In[181]:


for col in customer_df.describe().columns:
    set_size_style(16,2,'ticks')
    sns.boxplot(data=customer_df, x=col)
    plt.show()


# In[182]:


customer_df = customer_df.drop(customer_df[customer_df['Departure Delay in Minutes'] > 500 ].index)
customer_df = customer_df.drop(customer_df[customer_df['Arrival Delay in Minutes'] > 500 ].index)
customer_df = customer_df.drop(customer_df[customer_df['Flight Distance'] > 5500 ].index)
customer_df.reset_index(drop=True, inplace=True)
customer_df.shape


# In[183]:


customer_df.columns


# In[184]:


for col in customer_df.describe(include='object').columns:
    print('Column Name: ',col)
    print(customer_df[col].unique())
    print('-'*50)


# # EXPLOTARY DATA ANALYSIS

# In[185]:


set_size_style(12,5)
age_groups = customer_df.groupby('Age')['satisfaction'].value_counts(normalize=True).unstack()
satisfied_percentage = age_groups['satisfied'] * 100
ax =sns.lineplot(x=satisfied_percentage.index, y=satisfied_percentage.values, marker='o', color= sns.color_palette('Spectral')[0])
customize_plot(ax, 'Satisfied Percentage across Age', 'Age', 'Satisfied Percentage',13,10)
plt.grid(True)
plt.show()


# * Individuals in their 40s and 50s exhibit satisfaction with airline services.
# * Conversely, older individuals above the age of 70 express higher levels of dissatisfaction with the services provided

# In[186]:


set_size_style(12,7)
class_ratings = customer_df.groupby('Class').agg({'Cleanliness':'mean',
                                                       'Checkin service' : 'mean',
                                                       'Seat comfort':'mean',
                                                       'Inflight wifi service':'mean', 
                                                       'Leg room service':'mean'}).reset_index()
class_ratings_melted = pd.melt(class_ratings, id_vars='Class', var_name='Category', value_name='Mean Rating')
ax = sns.barplot(x='Class', y='Mean Rating', hue='Category', data=class_ratings_melted, palette='Spectral')
for c in ax.containers:
        ax.bar_label(c)
customize_plot(ax, 'Mean Ratings across Class', 'Class', 'Mean Rating',13,10)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels, title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')


# * Travelers in the business class generally give higher average ratings for cleanliness, check-in experience, in-flight wifi,     and legroom service.
# * Interestingly, passengers in the business class tend to rate seat comfort comparatively lower.

# # Encoding Categorical Features

# In[187]:


dummies=pd.get_dummies(customer_df['Class'], dtype=int)
dummies


# In[188]:


customer_encoded = pd.concat([customer_df,dummies], axis = 'columns')
customer_encoded.drop(columns = ['Class'], inplace=True)
customer_encoded


# In[189]:


customer_encoded['Customer Type'] = customer_encoded['Customer Type'].map({'Loyal Customer': 1, 'disloyal Customer': 0})
customer_encoded['Type of Travel'] = customer_encoded['Type of Travel'].map({'Personal Travel': 1, 'Business travel': 0})
customer_encoded['satisfaction'] = customer_encoded['satisfaction'].map({'satisfied': 1, 'dissatisfied': 0})
customer_encoded


# # SPLITING DATA

# In[190]:


X = customer_encoded.drop(columns = ['satisfaction'])
y = customer_encoded['satisfaction']
X.shape,y.shape


# In[191]:


X


# In[192]:


y


# In[193]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)
X_train.shape, y_train.shape


# # SCALLING DATA

# In[195]:


scaler = StandardScaler()


# In[196]:


X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# # SELECTING BEST MODEL

# In[197]:


models = {
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Random Forest Classifier': RandomForestClassifier(),
    'Decision Tree Classifier': DecisionTreeClassifier(),
}
results = []

for name, model in models.items():
    kf = KFold(n_splits=6, shuffle=True, random_state=42)
    cv_results = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring='accuracy')
    print(f'CV Score (Mean) {name}: {np.mean(cv_results)}')
    results.append(cv_results)

plt.boxplot(results, labels=models.keys())
plt.title('Cross-validation Scores for Classification Models')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.show()


# . Random Forest Classifier outperforms other classification models

# In[198]:


rf = RandomForestClassifier(random_state=42)


# In[199]:


rf.fit(X_train,y_train)


# In[200]:


y_pred = rf.predict(X_test)


# # MODEL EVALUATION

# In[201]:


plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test, cmap='Reds', values_format='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[202]:


report = classification_report(y_test, y_pred)
print(report)


# * The Random Forest Model achieves high precision, recall, and F1-score for both classes, indicating that it performs well in     classifying both dissatisfied and satisfied customers.
# * The overall accuracy of 96% suggests that the model is accurate in predicting the customer satisfaction status.
# 

#            Let's try Extreme Gradient Boosting

# # EXTREME GRADIENT BOOSTING(XG BOOST CLASSIFIER)

# In[203]:


customer_dmatrix = xgb.DMatrix(data=X_train_scaled,label=y_train)
params={'binary':'logistic'}
cv_results = xgb.cv(dtrain=customer_dmatrix,
                    params=params,
                    nfold=4,
                    metrics="error",
                    as_pandas=True,
                    seed=42)


# In[204]:


cv_results['test-accuracy-mean'] = 1 - cv_results['test-error-mean']
mean_accuracy = cv_results['test-accuracy-mean'].iloc[-1]
print("Mean Accuracy (CV):", mean_accuracy)


#  * Now, let's proceed with hyperparameter tuning to improve the accuracy.

# ## HYPERPARAMETER TUNNING

# ### Gridsearch cv

# In[205]:


xgb_param_grid = {
    'learning_rate': [0.01, 0.1, 0.5],
    'n_estimators': [200],
    'max'
    'subsample': [0.3, 0.5, 0.9]
}
xgboost_model = xgb.XGBClassifier(objective = 'binary:logistic',seed=42)
grid_xgboost = GridSearchCV(
    estimator=xgboost_model,
    param_grid=xgb_param_grid,
    scoring='accuracy',
    cv=4,
    verbose=1
)
grid_xgboost.fit(X_train_scaled, y_train)
print("Best parameters found:", grid_xgboost.best_params_)
print("Best Accuracy Score:", grid_xgboost.best_score_)


# ### RandomizedSearchCV

# In[206]:


xgb_param_grid = {
    'learning_rate': np.arange(0.01, 0.2, 0.01), 
    'n_estimators': [200],
    'subsample': np.arange(0.3, 1.0, 0.1), 
    'max_depth': np.arange(3, 10, 1), 
    'colsample_bytree': np.arange(0.3, 1.0, 0.1) 
}
xgboost_model = xgb.XGBClassifier(objective='binary:logistic', seed=42)
random_xgboost = RandomizedSearchCV(
    estimator=xgboost_model,
    param_distributions=xgb_param_grid,
    n_iter=15, 
    scoring='accuracy',
    cv=3,
    verbose=1,
    random_state=42
)

random_xgboost.fit(X_train_scaled, y_train)
print("Best parameters found:", random_xgboost.best_params_)
print("Best Accuracy Score:", random_xgboost.best_score_)


# In[207]:


xgb_model = xgb.XGBClassifier(objective = 'binary:logistic',
                              subsample= 0.7,
                              n_estimators= 200,
                              max_depth = 9,
                              learning_rate = 0.11,
                              colsample_bytree=0.8)
xgb_model.fit(X_train_scaled, y_train)


# ## MODEL EVALUATION

# In[208]:


plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay.from_estimator(xgb_model, X_test_scaled, y_test, cmap='Reds', values_format='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[209]:


report = classification_report(y_test, y_pred)
print(report)


# # CONCLUSION

# * Both Random Forest and XGBoost models exhibit comparable performance metrics, including accuracy, precision, recall, and F1-     score.
# * However, the XGBoost model demonstrates a slightly lower number of false positives and false negatives compared to the Random   Forest model.
# * This suggests that the XGBoost model outperforms the Random Forest model slightly in terms of minimizing classification         errors.
# 
# 
