import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

dataRob=pd.read_csv('robusta_data_cleaned.csv')

dataArab=pd.read_csv('arabica_data_cleaned.csv')

# print(dataRob.head(5))
# print(dataRob.columns)


# print(dataArab.head(5))
# print(dataArab.columns)


# rename columns
dataRob=dataRob.rename(index=str, columns={"Fragrance...Aroma": "Aroma", "Salt...Acid":"Acidity", 'Bitter...Sweet':"Sweetness"})
# print(dataRob.columns)
dataArab=dataArab.rename(index=str, columns={"Uniformity": "Uniform.Cup", "Body":"Mouthfeel"})
print(dataRob.head(5))
# arrange columns
dataRob = dataRob[['Unnamed: 0', 'Species', 'Owner', 'Country.of.Origin', 'Farm.Name',
       'Lot.Number', 'Mill', 'ICO.Number', 'Company', 'Altitude', 'Region',
       'Producer', 'Number.of.Bags', 'Bag.Weight', 'In.Country.Partner',
       'Harvest.Year', 'Grading.Date', 'Owner.1', 'Variety',
       'Processing.Method', 'Aroma', 'Flavor', 'Aftertaste',
       'Acidity', 'Sweetness', 'Balance', 'Mouthfeel',
       'Uniform.Cup', 'Clean.Cup', 'Cupper.Points', 'Total.Cup.Points', 'Moisture',
       'Category.One.Defects', 'Quakers', 'Color', 'Category.Two.Defects',
       'Expiration', 'Certification.Body', 'Certification.Address',
       'Certification.Contact', 'unit_of_measurement', 'altitude_low_meters',
       'altitude_high_meters', 'altitude_mean_meters']]

dataArab = dataArab[['Unnamed: 0', 'Species', 'Owner', 'Country.of.Origin', 'Farm.Name',
       'Lot.Number', 'Mill', 'ICO.Number', 'Company', 'Altitude', 'Region',
       'Producer', 'Number.of.Bags', 'Bag.Weight', 'In.Country.Partner',
       'Harvest.Year', 'Grading.Date', 'Owner.1', 'Variety',
       'Processing.Method', 'Aroma', 'Flavor', 'Aftertaste',
       'Acidity', 'Sweetness', 'Balance', 'Mouthfeel',
       'Uniform.Cup', 'Clean.Cup', 'Cupper.Points', 'Total.Cup.Points', 'Moisture',
       'Category.One.Defects', 'Quakers', 'Color', 'Category.Two.Defects',
       'Expiration', 'Certification.Body', 'Certification.Address',
       'Certification.Contact', 'unit_of_measurement', 'altitude_low_meters',
       'altitude_high_meters', 'altitude_mean_meters']]

# cleaning data outlier
dataArab=dataArab.drop(dataArab[dataArab['Sweetness']==0].index)
print(dataArab.tail(5))
print(len(dataArab))

# join data frame
dataJoin=pd.concat([dataRob, dataArab], ignore_index=True)
# print(dataJoin.tail(5))
# print(dataJoin.columns)

# clean data
dataJoin.drop(dataJoin.columns[[0]], axis=1, inplace=True)
# print(dataJoin.columns)
# print(dataJoin.iloc[0])

dataML=dataJoin.drop(['Owner', 'Country.of.Origin', 'Farm.Name','Lot.Number', 'Mill', 'ICO.Number', 
       'Company', 'Altitude', 'Region','Producer', 'Number.of.Bags', 'Bag.Weight', 'In.Country.Partner',
       'Harvest.Year', 'Grading.Date', 'Owner.1', 'Variety','Processing.Method','Category.One.Defects', 'Quakers',
       'Color', 'Category.Two.Defects','Expiration', 'Certification.Body', 'Certification.Address','Certification.Contact', 
       'unit_of_measurement', 'altitude_low_meters','altitude_high_meters', 'altitude_mean_meters', 'Total.Cup.Points','Uniform.Cup', 'Clean.Cup','Moisture', 'Cupper.Points'], axis =1)
# print(dataML.columns)
# print(dataML.iloc[0])
# print(dataML.info())

# f, axx = plt.subplots(figsize=(10,10))
# sn.heatmap(dataML.iloc[:,1:].corr(), linewidths=0.5, cmap="Blues", annot=True,fmt=".1f", ax=axx)
# plt.show()

# 1 Labelling
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
y = label.fit_transform(dataML['Species'])
# print(label.classes_)       #['Arabica' 'Robusta']
print(y)                      #[    0        1     ]     

x=dataML.drop('Species', axis='columns')
print(x.iloc[1308])
# print(x.values[0])
# print(x.values[29])
# ['Aroma' 'Flavor' 'Aftertaste' 'Acidity' 'Sweetness' 'Balance' 'Mouthfeel'
#  'Uniform.Cup' 'Clean.Cup' 'Cupper.Points']

# Splitting
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(
    x,
    y,
    test_size = .1
)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
modelLR = LogisticRegression(solver = 'liblinear')
modelLR.fit(xtrain, ytrain)
# print('Logistic Regression : ',round(modelLR.score(xtest, ytest) * 100, 4), '%')
# print(modelLR.predict([[ 7.83,  8.08,  7.75,  7.92,  7,    7.92,  8.25  ]]))
# print(modelLR.predict([[8.75 , 8.67 , 8.5,   8.58, 10, 8.42,8.42   ]]))

# ML decision tree
from sklearn import tree
modelDT = tree.DecisionTreeClassifier(criterion='entropy')        #criterion untuk macam2 draw tree
modelDT.fit(xtrain,ytrain)
# print('Decision Tree : ',round(modelDT.score(xtest,ytest)*100,4), '%')       # ,2 =>bnyaknya angka desimal dibelakang koma 
# print(modelDT.predict([[ 7.83,  8.08,  7.75,  7.92,  7,    7.92,  8.25  ]]))
# print(modelDT.predict([[8.75 , 8.67 , 8.5,   8.58, 10, 8.42,8.42   ]]))

# random forest
from sklearn.ensemble import RandomForestClassifier
modelRF = RandomForestClassifier(n_estimators=10)
modelRF.fit(xtrain,ytrain)
# print('RF : ',round(modelRF.score(xtest,ytest)*100,4), '%')    
# print(modelRF.predict([[ 7.83,  8.08,  7.75,  7.92,  7,    7.92,  8.25  ]]))
# print(modelRF.predict([[8.75 , 8.67 , 8.5,   8.58, 10, 8.42,8.42   ]]))

# # k means
# from sklearn.cluster import KMeans
# modelKM=KMeans(n_clusters=2, random_state=1) 
# modelKM.
# fit(xtrain,ytrain)
# print('K Means : ',round(modelKM.score(xtest,ytest)*100,4), '%')    

# # SVM support vector machine
from sklearn.svm import SVC
modelSVM=SVC(gamma='auto')     #gamma=berapa titik terluar di luar garis cluster untuk menentutukan titik pemisahnya
modelSVM.fit(xtrain,ytrain)
# print('SVM : ',round(modelSVM.score(xtest,ytest)*100,4), '%')    
# print(modelSVM.predict([[ 7.83,  8.08,  7.75,  7.92,  7,    7.92,  8.25  ]]))
# print(modelSVM.predict([[8.75 , 8.67 , 8.5,   8.58, 10, 8.42,8.42   ]]))

#  KNN(K-Nearest Neighbors)
from sklearn.neighbors import KNeighborsClassifier
modelKNN = KNeighborsClassifier(n_neighbors=7)
modelKNN.fit(xtrain,ytrain)
# print('KNN : ',round(modelKNN.score(xtest,ytest)*100,4), '%')    
# print(modelKNN.predict([[ 7.83,  8.08,  7.75,  7.92,  7,    7.92,  8.25  ]]))
# print(modelKNN.predict([[8.75 , 8.67 , 8.5,   8.58, 10, 8.42,8.42   ]]))

from sklearn.externals import joblib
joblib.dump(modelRF,'model1')
'''

# add graphic robusta vs arabica
from math import pi

meanAroma1=dataRob['Aroma'].mean()
meanFlavor1=dataRob['Flavor'].mean()
meanAftertaste1=dataRob['Aftertaste'].mean()
meanAcid1=dataRob['Acidity'].mean()
meanSweet1=dataRob['Sweetness'].mean()
meanBalance1=dataRob['Balance'].mean()
meanMouthfeel1=dataRob['Mouthfeel'].mean()

meanAroma2=dataArab['Aroma'].mean()
meanFlavor2=dataArab['Flavor'].mean()
meanAftertaste2=dataArab['Aftertaste'].mean()
meanAcid2=dataArab['Acidity'].mean()
meanSweet2=dataArab['Sweetness'].mean()
meanBalance2=dataArab['Balance'].mean()
meanMouthfeel2=dataArab['Mouthfeel'].mean()

# Set data
df = pd.DataFrame({
'group': ['rob','arab'],
'Aroma': [meanAroma1, meanAroma2],
'Flavor': [meanFlavor1, meanFlavor2],
'Aftertaste': [meanAftertaste1, meanAftertaste2],
'Acidity': [meanAcid1, meanAcid2],
'Sweetness': [meanSweet1, meanSweet2],
'Balance': [meanBalance1, meanBalance2],
'Mouthfeel': [meanMouthfeel1, meanMouthfeel2]
})

 # ------- PART 1: Create background
 
# number of variable
categories=list(df)[1:]
N = len(categories)
 
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
 
# Initialise the spider plot
ax = plt.subplot(111, polar=True)
 
# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
 
# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories)
 
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([4,8,12], ["4","8","12"], color="grey", size=10)
plt.ylim(0,10)
 
 
# ------- PART 2: Add plots
 
# Plot each individual = each line of the data
# I don't do a loop, because plotting more than 3 groups makes the chart unreadable
 
# Ind1
values=df.loc[0].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Robusta")
ax.fill(angles, values, 'b', alpha=0.1)
 
# Ind2
values=df.loc[1].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="Arabica")
ax.fill(angles, values, 'r', alpha=0.1)
 
# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
# plt.show()

# recommendation with content based
# 1. find quantile in features

quantileAromaVS2=dataArab['Aroma'].quantile(0.2)
quantileAromaS2=dataArab['Aroma'].quantile(0.4)
quantileAromaM2=dataArab['Aroma'].quantile(0.6)
quantileAromaH2=dataArab['Aroma'].quantile(0.8)
quantileAromaVH2=dataArab['Aroma'].quantile(1)

quantileFlavorVS2=dataArab['Flavor'].quantile(0.2)
quantileFlavorS2=dataArab['Flavor'].quantile(0.4)
quantileFlavorM2=dataArab['Flavor'].quantile(0.6)
quantileFlavorH2=dataArab['Flavor'].quantile(0.8)
quantileFlavorVH2=dataArab['Flavor'].quantile(1)

quantileAftertasteVS2=dataArab['Aftertaste'].quantile(0.2)
quantileAftertasteS2=dataArab['Aftertaste'].quantile(0.4)
quantileAftertasteM2=dataArab['Aftertaste'].quantile(0.6)
quantileAftertasteH2=dataArab['Aftertaste'].quantile(0.8)
quantileAftertasteVH2=dataArab['Aftertaste'].quantile(1)

quantileAcidityVS2=dataArab['Acidity'].quantile(0.2)
quantileAcidityS2=dataArab['Acidity'].quantile(0.4)
quantileAcidityM2=dataArab['Acidity'].quantile(0.6)
quantileAcidityH2=dataArab['Acidity'].quantile(0.8)
quantileAcidityVH2=dataArab['Acidity'].quantile(1)
print(quantileAcidityVS2)
quantileSweetnessVS2=dataArab['Sweetness'].quantile(0.2)
quantileSweetnessS2=dataArab['Sweetness'].quantile(0.4)
quantileSweetnessM2=dataArab['Sweetness'].quantile(0.6)
quantileSweetnessH2=dataArab['Sweetness'].quantile(0.8)
quantileSweetnessVH2=dataArab['Sweetness'].quantile(1)
print(type(quantileAcidityH2))
print(type(10.1))

quantileBalanceVS2=dataArab['Balance'].quantile(0.2)
quantileBalanceS2=dataArab['Balance'].quantile(0.4)
quantileBalanceM2=dataArab['Balance'].quantile(0.6)
quantileBalanceH2=dataArab['Balance'].quantile(0.8)
quantileBalanceVH2=dataArab['Balance'].quantile(1)

quantileMouthfeelVS2=dataArab['Mouthfeel'].quantile(0.2)
quantileMouthfeelS2=dataArab['Mouthfeel'].quantile(0.4)
quantileMouthfeelM2=dataArab['Mouthfeel'].quantile(0.6)
quantileMouthfeelH2=dataArab['Mouthfeel'].quantile(0.8)
quantileMouthfeelVH2=dataArab['Mouthfeel'].quantile(1)

# 2. categorized (very soft, soft, medium, hard, very hard) based on feature

def labelAroma2(row):
   if row['Aroma'] < quantileAromaVS2 and row['Aroma'] >0.9 :
      return 'vsa'
   if row['Aroma'] >= quantileAromaVS2 and row['Aroma'] < quantileAromaS2 :
      return 'sa'
   if row['Aroma'] >= quantileAromaS2 and row['Aroma'] < quantileAromaM2 :
      return 'ma'
   if row['Aroma'] >= quantileAromaH2 and row['Aroma'] < quantileAromaVH2 :
      return 'ha'
   if row['Aroma'] >= quantileAromaVH2 and row['Aroma'] < 10.1 :
      return 'vha'
   return 'Other'

def labelFlavor2(row):
   if row['Flavor'] < quantileFlavorVS2 and row['Flavor'] >0.9 :
      return 'vsf'
   if row['Flavor'] >= quantileFlavorVS2 and row['Flavor'] < quantileFlavorS2 :
      return 'sf'
   if row['Flavor'] >= quantileFlavorS2 and row['Flavor'] < quantileFlavorM2 :
      return 'mf'
   if row['Flavor'] >= quantileFlavorH2 and row['Flavor'] < quantileFlavorVH2 :
      return 'hf'
   if row['Flavor'] >= quantileFlavorVH2 and row['Flavor'] < 10.1 :
      return 'vhf'
   return 'Other'

def labelAftertaste2(row):
   if row['Aftertaste'] < quantileAftertasteVS2 and row['Aftertaste'] >=0.9 :
      return 'vsat'
   if row['Aftertaste'] >= quantileAftertasteVS2 and row['Aftertaste'] < quantileAftertasteS2 :
      return 'sat'
   if row['Aftertaste'] >= quantileAftertasteS2 and row['Aftertaste'] < quantileAftertasteM2 :
      return 'mat'
   if row['Aftertaste'] >= quantileAftertasteH2 and row['Aftertaste'] < quantileAftertasteVH2 :
      return 'hat'
   if row['Aftertaste'] >= quantileAftertasteVH2 and row['Aftertaste'] < 10.1 :
      return 'vhat'
   return 'Other'

def labelAcidity2(row):
   if row['Acidity'] >0.9 and row['Acidity'] < quantileAcidityVS2 :
      return 'vsac'
   if row['Acidity'] >= quantileAcidityVS2 and row['Acidity'] < quantileAcidityS2 :
      return 'sac'
   if row['Acidity'] >= quantileAcidityS2 and row['Acidity'] < quantileAcidityM2 :
      return 'mac'
   if row['Acidity'] >= quantileAcidityH2 and row['Acidity'] < quantileAcidityVH2 :
      return 'hac'
   if row['Acidity'] >= quantileAcidityVH2 and row['Acidity'] < 10.1 :
      return 'vhac' 
   return 'Other'

def labelSweetness2(row):
   if row['Sweetness'] < quantileSweetnessVS2 and row['Sweetness'] >0.9 :
      return 'vssw'
   if row['Sweetness'] >= quantileSweetnessVS2 and row['Sweetness'] < quantileSweetnessS2 :
      return 'ssw'
   if row['Sweetness'] >= quantileSweetnessS2 and row['Sweetness'] < quantileSweetnessM2 :
      return 'msw'
   if row['Sweetness'] >= quantileSweetnessH2 and row['Sweetness'] < quantileSweetnessVH2 :
      return 'hsw'
   if row['Sweetness'] >= quantileSweetnessVH2 and row['Sweetness'] < 10.1 :
      return 'vhsw' 
   return 'Other'   

def labelBalance2(row):
   if row['Balance'] < quantileBalanceVS2 and row['Balance'] >0.9 :
      return 'vsb'
   if row['Balance'] >= quantileBalanceVS2 and row['Balance'] < quantileBalanceS2 :
      return 'sb'
   if row['Balance'] >= quantileBalanceS2 and row['Balance'] < quantileBalanceM2 :
      return 'mb'
   if row['Balance'] >= quantileBalanceH2 and row['Balance'] < quantileBalanceVH2 :
      return 'hb'
   if row['Balance'] >= quantileBalanceVH2 and row['Balance'] < 10.1 :
      return 'vhb'
   return 'Other'

def labelMouthfeel2(row):
   if row['Mouthfeel'] < quantileMouthfeelVS2 and row['Mouthfeel'] >0.9 :
      return 'vsmf'
   if row['Mouthfeel'] >= quantileMouthfeelVS2 and row['Mouthfeel'] < quantileMouthfeelS2 :
      return 'smf'
   if row['Mouthfeel'] >= quantileMouthfeelS2 and row['Mouthfeel'] < quantileMouthfeelM2 :
      return 'mmf'
   if row['Mouthfeel'] >= quantileMouthfeelH2 and row['Mouthfeel'] < quantileMouthfeelVH2 :
      return 'hmf'
   if row['Mouthfeel'] >= quantileMouthfeelVH2 and row['Mouthfeel'] < 10.1 :
      return 'vhmf'
   return 'Other'

dataArab['Aroma2'] = dataArab.apply (lambda row: labelAroma2(row), axis=1)
dataArab['Flavor2'] = dataArab.apply (lambda row: labelFlavor2(row), axis=1)
dataArab['Aftertaste2'] = dataArab.apply (lambda row: labelAftertaste2(row), axis=1)
dataArab['Acidity2'] = dataArab.apply (lambda row: labelAcidity2(row), axis=1)
dataArab['Sweetness2'] = dataArab.apply (lambda row: labelSweetness2(row), axis=1)
dataArab['Balance2'] = dataArab.apply (lambda row: labelBalance2(row), axis=1)
dataArab['Mouthfeel2'] = dataArab.apply (lambda row: labelMouthfeel2(row), axis=1)

# print(dataArab[['Aroma2', 'Flavor2', 'Aftertaste2', 'Acidity2', 'Sweetness2', 'Balance2', 'Mouthfeel2']])



# dataRob['Aroma1']=[False if each < meanAroma1 else True for each in dataRob['Aroma']]
# dataRob['Flavor1']=[False if each < meanFlavor1 else True for each in dataRob['Flavor']]
# dataRob['Aftertaste1']=[False if each < meanAftertaste1 else True for each in dataRob['Aftertaste']]
# dataRob['Acidity1']=[False if each < meanAcid1 else True for each in dataRob['Acidity']]
# dataRob['Sweetness1']=[False if each < meanSweet1 else True for each in dataRob['Sweetness']]
# dataRob['Balance1']=[False if each < meanBalance1 else True for each in dataRob['Balance']]
# dataRob['Mouthfeel1']=[False if each < meanMouthfeel1 else True for each in dataRob['Mouthfeel']]
# print((dataArab['Aroma'].max()+dataArab['Aroma'].min())/2)
# print(dataArab['Aroma'].max())
# print(dataArab['Aroma'].min())

# print(meanAroma1)
# print(quantileAroma1)
# print(dataRob[['Aroma', 'Aroma1', 'Flavor', 'Flavor1']])
# print(dataRob)

# dataArab['Aroma2']=["very soft" if each < quantileAroma1 elif "soft" for each >= quantileAroma1 or each < quantileAroma2 
#        elif "medium" for each >= quantileAroma2 or each < quantileAroma3
#        elif "hard" for each >= quantileAroma3 or each < quantileAroma3
#        elif "very hard" for each >= quantileAroma3 or each <= 10 
#        for each in dataArab['Aroma']]
# cut_points = [np.quantile(dataArab['Aroma'], i) for i in [.2, .4, .6 ,.8, 1]]
# dataArab['Aroma2'] = 1
# for i in range(5):
#     dataArab['Aroma2'] = dataArab['Aroma2'] + (dataArab['Aroma'] < cut_points[i])
# print(dataArab['Aroma2'])
# # if dataArab['Aroma2']
# dataArab['Flavor2']=["no" if each < meanFlavor2 else "yes" for each in dataArab['Flavor']]
# dataArab['Aftertaste2']=["no" if each < meanAftertaste2 else "yes" for each in dataArab['Aftertaste']]
# dataArab['Acidity2']=["no" if each < meanAcid2 else "yes" for each in dataArab['Acidity']]
# dataArab['Sweetness2']=["no" if each < meanSweet2 else "yes" for each in dataArab['Sweetness']]
# dataArab['Balance2']=["no" if each < meanBalance2 else "yes" for each in dataArab['Balance']]
# dataArab['Mouthfeel2']=["no" if each < meanMouthfeel2 else "yes" for each in dataArab['Mouthfeel']]



# print(dataArab[['Aroma', 'Aroma2']])
# print(dataArab)
# print(meanAroma2)
# print(dataArab[['Aroma', 'Aroma2', 'Flavor', 'Flavor2']])

# clean data(erase NaN and "Other")
dataArab=dataArab.replace({'Variety':'Other',
       'Aroma2':'Other',
       'Flavor2':'Other', 
       'Aftertaste2':'Other',
       'Acidity2':'Other',
       'Balance2':'Other', 
       'Mouthfeel2':'Other' },{
       'Variety':np.NaN,
       'Aroma2':np.NaN, 
       'Flavor2':np.NaN, 
       'Aftertaste2':np.NaN, 
       'Acidity2':np.NaN,
       'Balance2':np.NaN, 
       'Mouthfeel2':np.NaN})

dataArab=dataArab.dropna(subset=['Variety', 'Aroma2', 'Flavor2', 'Aftertaste2', 'Acidity2', 'Sweetness2', 'Balance2', 'Mouthfeel2'])
# print(dataArab[['Aroma2', 'Flavor2', 'Aftertaste2', 'Acidity2', 'Sweetness2', 'Balance2', 'Mouthfeel2']])

# print(dataArab.isnull().sum())
# print(dataArab['Variety'])
# print(dataArab.columns)
# print(type(dataArab))
dataArab1=dataArab.drop(['Unnamed: 0', 'Species', 'Owner', 'Country.of.Origin', 'Farm.Name',
       'Lot.Number', 'Mill', 'ICO.Number', 'Company', 'Altitude', 'Region',
       'Producer', 'Number.of.Bags', 'Bag.Weight', 'In.Country.Partner',
       'Harvest.Year', 'Grading.Date', 'Owner.1',
       'Processing.Method', 'Aroma', 'Flavor', 'Aftertaste',
       'Acidity', 'Sweetness', 'Balance', 'Mouthfeel',
       'Uniform.Cup', 'Clean.Cup', 'Cupper.Points', 'Total.Cup.Points', 'Moisture',
       'Category.One.Defects', 'Quakers', 'Color', 'Category.Two.Defects',
       'Expiration', 'Certification.Body', 'Certification.Address',
       'Certification.Contact', 'unit_of_measurement', 'altitude_low_meters',
       'altitude_high_meters', 'altitude_mean_meters'], axis=1, )
dataArab1 = dataArab1.drop_duplicates(subset='Variety', keep='first')
dataArab1=dataArab1[['Variety', 'Aroma2', 'Flavor2', 'Aftertaste2', 'Acidity2', 'Sweetness2'
   ,'Balance2', 'Mouthfeel2'
]]
print("dataArab1")
# print(type(dataArab1))
print(dataArab1.columns)
# print(type(dataArab1))

# add a new col :
def mergeCol(i):
    return str(i['Aroma2']) + ' '+str(i['Flavor2'])+ ' '+str(i['Aftertaste2'])+ ' '+str(i['Acidity2'])+ ' '+str(i['Sweetness2']) + ' '+str(i['Balance2'])+ ' '+str(i['Mouthfeel2'])

dataArab1['features'] = dataArab1.apply(
    mergeCol,
    axis=1
)
print(dataArab1)

# print(dataArab1.head(10))
# print(len(dataArab1['Aroma2'].unique()))
# print(len(dataArab1['Flavor2'].unique()))
# print(len(dataArab1['Aftertaste2'].unique()))
# print(len(dataArab1['Acidity2'].unique()))
# print(len(dataArab1['Sweetness2'].unique()))
# print(len(dataArab1['Balance2'].unique()))
# print(len(dataArab1['Mouthfeel2'].unique()))


# count vectorizer
from sklearn.feature_extraction.text import CountVectorizer
model=CountVectorizer(tokenizer=lambda x: x.split(' '))      #=>tokenizer=format nya word
matrixFeature=model.fit_transform(dataArab1['features'])
features=model.get_feature_names()
jmlFeatures = len(features)
# print(features)
# print(jmlFeatures)
# print('matrix feature')
# print(matrixFeature.toarray()[0])


# cosinus similarity
from sklearn.metrics.pairwise import cosine_similarity
score=cosine_similarity(matrixFeature)
# print('enumerate score')
# print(enumerate(score[0]))

# =============================================
# testing
# print(type(dataArab1[['Variety']]))
sukaKopi = 'Catimor'
indexSuka = dataArab1[dataArab1['Variety']=='Catimor'].index.values[0]
# print(dataArab1[dataArab1['Variety']=='Catimor'])

# print('index suka')
# print(type(indexSuka))

daftarScore = list(enumerate(score[18]))
# print(daftarScore)

sortDaftarScore = sorted(
    daftarScore,
    key = lambda j: j[1],
    reverse = True
)

print(sortDaftarScore[:5])

# =======================================
# show top 5 recommendation randomly
similarGames = []
for i in sortDaftarScore:
    if i[1] > .8:
        similarGames.append(i)

# print(similarGames)

import random
rekomendasi = random.choices(similarGames, k=5)     #k jumlah recommendation   
# print(np.unique(np.array(rekomendasi)))

for i in rekomendasi:
    data=dataArab1.iloc[i[0]].values
   #  print(dataArab1.iloc[i[0]].values)
   #  print(data[0], data[1], data[2])



# ['ha', 'hac', 'hat', 'hb', 'hf', 'hmf',                                                   'ma', 'mac', 'mat', 'mb', 'mf',                                                 'mmf',     'sa', 'sac', 'sat', 'sb', 'sf',                                            'smf', 'vhsw', 'vsa', 'vsac', 'vsat', 'vsb', 'vsf', 'vsmf']
hard aroma, hard acidity, hard aftertaste, hard balance, hard flavor, hard mouthfeel, medium aroma, medium acidity, medium aftertaste, medium balance, medium flavor, medium mouthfeel,soft aroma, soft acidity, soft aftertaste, soft balance, soft flavor, soft mouthfeel , veryhard sweetness,verysoft aroma, verysoft acidity, verysoft aftertaste, verysoft balance, verysoft flavor, verysoft mouthfeel 

# ['ha', 'hac', 'hat', 'hb', 'hf', 'hmf',                                                   'ma', 'mac', 'mat', 'mb', 'mf',                                                 'mmf',     'sa', 'sac', 'sat', 'sb', 'sf',                                            'smf', 'vhsw', 'vsa', 'vsac', 'vsat', 'vsb', 'vsf', 'vsmf']
hard aroma, hard acidity, hard aftertaste, hard balance, hard flavor, hard mouthfeel, medium aroma, medium acidity, medium aftertaste, medium balance, medium flavor, medium mouthfeel,soft aroma, soft acidity, soft aftertaste, soft balance, soft flavor, soft mouthfeel , veryhard sweetness,verysoft aroma, verysoft acidity, verysoft aftertaste, verysoft balance, verysoft flavor, verysoft mouthfeel 

hard aroma,
medium aroma,
soft aroma
verysoft aroma

 hard acidity, 
 medium acidity
soft acidity
verysoft acidity

 hard aftertaste, 
 medium aftertaste
soft aftertaste
verysoft aftertaste,

 hard balance, 
 medium balance
soft balance
verysoft balance

 hard flavor, 
 medium flavor
soft flavor
verysoft flavor

 hard mouthfeel,  , , , , ,,, , , , , , ,, ,  , ,   
medium mouthfeel
soft mouthfeel 
verysoft mouthfeel

veryhard sweetness

'''
