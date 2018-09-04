
from django.template.response import TemplateResponse, HttpResponse
from django.shortcuts import render
from django.http import HttpResponseRedirect
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
import pickle
import numpy as np
import pandas as pd

#data pull -- temporarily we are loading data from dumped data set
#should use model to pull data from dataset to form such a table
def hist_data_pull():
    data_raw = pd.read_csv('~/Downloads/NYU-ITP/itp_analytics/data/ConversionData.csv')
    return data_raw

def external_data_pull():
    data_ex = pd.read_csv('~/Downloads/NYU-ITP/itp_analytics/data/external_data_zip_based.csv')

#descriptive data preparation
def descriptive_data_prep(request):
    data = hist_data_pull()

    #parse domain name and url path
    data['Attributions: Unified Tracking Entry Path Clean 1'] = data['Attributions: Unified Tracking Entry Path Clean'].map(
        lambda a: str(a)[str(a).find("/") + 1:str(a).find("/", str(a).find("/") + 1)] if str(a)[0] == '/' else np.NaN)
    data['Leads Intended Storage Duration In Months'] = pd.to_numeric(
        data['Leads Intended Storage Duration In Months'].str.replace(",", ""))
    #junk value clean up
    data = data[
        ((data['Leads Zip Code'].str.len() == 5) | (data['Leads Zip Code'].isnull())) & (
            data['Leads Zip Code'].str.isnumeric())]

    # dates conversion
    df = data
    df['Leads Created Date Year'] = df['Leads Created Date'].dt.year
    df['Leads Created Date Month'] = df['Leads Created Date'].dt.month
    df['Leads Created Date Date'] = df['Leads Created Date'].dt.day
    df['Leads Created Date Weekday'] = df['Leads Created Date'].dt.dayofweek
    # Map Months to season 1= Winter, 2 = Spring, 3 = Summer, 4 = Fall
    df['Leads Created Date Season'] = df['Leads Created Date'].dt.month.map(
        {1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 9: 4, 10: 4, 11: 4, 12: 1})

    #rows filter
    data_filtered = df[df['Leads Intended Storage Duration Tiers'] != 'Undefined']
    data_filtered = data_filtered[(data_filtered['Leads Intended Storage Duration In Months'] != '11,231')
                                 & (data_filtered['Leads Intended Storage Duration In Months'] != '11,211')]
    #save for descriptive analytics
    data_filtered.to_csv('descriptive_data.csv',index=false)


# data preparation
def predictive_data_prep(request):
    data_filtered = pd.read_csv('descriptive_data.csv')
    # conver num to cat data
    for row in data_filtered:   
        row['Leads Created Date Weekday'] = str(row['Leads Created Date Weekday']) + 'str'
        row['Leads Created Date Season'] = str(row['Leads Created Date Season']) + 'str'
        row['Leads Created Date Month'] = str('Leads Created Date Month') + 'str'
    
    #columns filter
    fields_model = ['Leads Zip Code','Leads Created Date Month','Leads Created Date Weekday','Leads Created Date Season','Leads Has Storage Plan Quote (Yes / No)'
                   ,'Leads Intended Storage Duration In Months','Leads Is Chat Lead (Yes / No)'
                   ,'Attributions: Unified Tracking Unified Ops Sector','Attributions: Unified Tracking Entry Referrer Domain'
                   ,'Attributions: Unified Tracking Entry Path Clean 1','Attributions: Unified Tracking UTM Medium Categories'
                   ,'Sales Transactions Is Gross Conversion (Yes / No)']

    data_filtered = data_filtered[fields_model]

    #missing value imputation for categorical fields:
    data_main = data_filtered.apply(missing_value_imputer,axis = 1)

    #convert ca

    #data merge -- move out
    data_external = external_data_pull()
    data_model = pd.merge(data_main, data_external, left_on = 'Lead Zip Code', right_on = 'Zip Code',
      how='left')
    data_model = data_model.drop(['Lead Zip Code','Zip Code'], axis = 1)

    return data_model
    
#predictice model training
def model_train():
    data_model = hist_data_prep()

    # set X and Y
    X = data_model.drop('Sales Transactions Is Gross Conversion (Yes / No)', 1)
    Y = data_model['Sales Transactions Is Gross Conversion (Yes / No)']
    le = LabelEncoder()
    le.fit(['No', 'Yes'])
    Y = le.transform(Y)

    # encode all categorical fields
    le2 = LabelEncoder()
    catColumns = X.columns.drop('Leads Intended Storage Duration In Months')
    for col in catColumns:
        n = len(X[col].unique())
        if (n > 2):
            temp = pd.get_dummies(X[col])
            temp = temp.drop(temp.columns[0], axis=1)
            X[temp.columns] = temp
            X.drop(col, axis=1, inplace=True)
        else:
            le2.fit(X[col])
            X[col] = le2.transform(X[col])


    #LR Model training
    steps = [
        ('featureSelection', SelectFromModel(LogisticRegression())),
        ('lr', LogisticRegression()),
    ]

    pipeline = Pipeline(steps)
    grid_lr = dict(lr__C=[10 ** i for i in range(-3, 3)],
                   lr__penalty=['l1', 'l2'],
                   featureSelection__threshold=[0.005, 0.05, 0.5]
                   )

    model_lr_op = GridSearchCV(pipeline, param_grid=grid_lr, scoring='roc_auc')
    model_lr_op.fit(X, Y)

    model = model_lr_op.best_estimator_
    pickle.dump(model, open('finalized_model.sav', 'wb'))

#lead data collection
def lead_data_collect(request):
    new_lead = pd.dataframe
    #...
    return new_lead


#lead prediction
def lead_predict(request):
    X = lead_data_collect(request)
    loaded_model = pickle.load(open(finalized_model.sav, 'rb'))
    result = loaded_model.predict_(X)
    return result

def prep_data_pull():
	q = Conversiondataprep.objects.all().values()
    return pd.DataFrame.from_records(q)

def zip_data_pull():
	q = Zipdata.objects.all().values()
    return pd.DataFrame.from_records(q)

def price_data_pull():
    q = Pricedata.objects.all().values()
    return pd.DataFrame.from_records(q)

#updated to accomodate merge in pricing data
def data_merge(request):
	data_model = prep_data_pull()
	data_external = zip_data_pull()
    data_external_2 = price_data_pull()
    data_merged = pd.merge(data_model, data_external, left_on = 'Leads Zip Code', right_on = 'Zipcode',
      how='left')

    #Maybe  be needed for zipcode conversion and merge based on the converted zipcode
    #data_external_2.zipcode = data_price.zipcode*1.0
    #data_external_2.zipcode = data_price.zipcode.apply(str)

    data_merged = pd.merge(data_merged, data_external_2 , left_on = 'Zipcode', right_on = 'zipcode',
      how='left')

	#missing value imputation:
	num_columns = ['Mean','Median','Population','Homevalue','makespace','extraspace','privatecompanies','publicstorage','cubesmart']
	cat_columns = list(set(data_merged.columns) - set(num_columns) - set('Leads Intended Storage Duration In Months'))
	temp = deepcopy(data_merged)
	data_model = data_merged.apply(missing_value_imputer,axis = 1)
	data_model = data_model.drop(['Leads Zip Code', 'Zipcode','zipcode'],axis=1)
	data_model = data_model.apply(cat_feature_toString,axis = 1)

	data_model.to_csv('data/data_model_4.csv',index = False)
	

def missing_value_imputer(row):
    for c in cat_columns:
        if str(row[c])== 'nan':
            row[c] = 'missing'
        else:
            row[c] = row[c]
    for n in num_columns:
        if str(row[n])=='nan':
            row[n] = np.mean(temp[n])
        else:
            row[n] = row[n]
    return row

def cat_feature_toString(row):
	    #temp = row['Leads Created Date Month']
	    row['Leads Created Date Weekday'] = str(row['Leads Created Date Weekday']) + 'str'
	    row['Leads Created Date Season'] = str(row['Leads Created Date Season']) + 'str'
	    row['Leads Created Date Month'] = str(row['Leads Created Date Month']) + 'str'
	    return row  