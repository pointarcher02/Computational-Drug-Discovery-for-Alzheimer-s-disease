import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score



dataset=pd.read_csv('acetylcholinesterase_final_dataset_pubchem.csv')
print(dataset)

X_data=dataset.drop(['pIC50'],axis=1)
Y_data=dataset.iloc[:,-1]

def remove_low_variance(input_data, threshold=0.1):
    selection = VarianceThreshold(threshold)
    selection.fit(input_data)
    return input_data[input_data.columns[selection.get_support(indices=True)]]

X_data = remove_low_variance(X_data, threshold=0.1)
X_data.to_csv('descriptor_list.csv', index = False)

# in the app we use the same descriptor list to generate the descriptor list of 218 columns for the input molecule
# by using the column names of this desciptor list
model = RandomForestRegressor(n_estimators=500, random_state=42)
model.fit(X_data, Y_data)
y_pred=model.predict(X_data)
r2 = model.score(X_data, Y_data)

#The r2 score and mean sqaured error are ways of metric evaluation
r2score=r2_score(Y_data,y_pred)
print("The R2 score for the RandomForestRegressor model is ",r2score)
mse=mean_squared_error(Y_data,y_pred)
print("The mean squared error for the model is ",mse)

plt.figure(figsize=(6,6))
plt.scatter(x=Y_data, y=y_pred, c="#7CAE00", alpha=0.3)

z = np.polyfit(Y_data, y_pred, 1)
p = np.poly1d(z)

plt.plot(Y_data,p(Y_data),"#F8766D")
plt.ylabel('Predicted pIC50')
plt.xlabel('Experimental pIC50')
plt.savefig('plot.pdf')
plt.show()

#saving model as pickle object
pickle.dump(model,open('acetylcholinesterase_model.pkl', 'wb'))

