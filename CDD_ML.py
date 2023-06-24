# We are working on a project in which we will be extracting bioactivity value
# for target protein or organism from the chembl database and building a ML model.

# Bioactivity refers to the ability of a molecule or compound to interact with a biological target or system and produce a 
# biological effect bioactivity is a crucial property that indicates the potential of a compound to modulate a biological target's function 
# and potentially lead to a therapeutic effect. here we are building an ML based Bioactivity prediction app.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lazypredict
from rdkit import Chem
from rdkit.Chem import Descriptors,Lipinski
from chembl_webresource_client.new_client import new_client
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error
from lazypredict.Supervised import LazyRegressor



# Now below we are searching for the target protein 

target=new_client.target
target_query=target.search('coronavirus')
targets=pd.DataFrame.from_dict(target_query)
#print(targets)

# Now from above dataset we see that there are three organisms which have target_type as Single Protein
# We can see that out of this three organsims if we pick any as input molecule it will act on a target protein and and predict it's biological acitivity

# Therefore now we have selected the target input molecule as SARS coronavirus 3C-like proteinase	

selected_target=targets.target_chembl_id[6]
print(selected_target)

# IC50 (half maximal inhibitory concentration) is a quantitative measure used in pharmacology and drug discovery to assess the potency of a compound or drug in 
# inhibiting a specific biological target or process. 

# Here, we will retrieve only bioactivity data for coronavirus 3C-like proteinase (CHEMBL3927) 
# #that are reported as IC 50 values in nM (nanomolar) unit.

activity=new_client.activity
res=activity.filter(target_chembl_id=selected_target).filter(standard_type="IC50")
df=pd.DataFrame.from_dict(res)
print(df)

# now we need to understand that the standard_value denotes the potentcy of the drug, therefore the lower the potentcy the better is the drug.
# therefore now we save this dataset for our further use 
df.to_csv('bioactivity_data.csv', index=False)

# Now we will do the preprocessing of the data

df2=df[df.standard_value.notna()]
# print(df2)

# Now we will be forming a new preprocessed dataframe that will contain only few rows and one extra row



#The bioactivity data is in the IC50 unit. Compounds having values of less than 1000 nM will be considered to be active while those greater than 10,000 nM will be considered to be inactive. 
#As for those values in between 1,000 and 10,000 nM will be referred to as intermediate.

bioactivity_class=[]
standard_value = []
canonical_smiles = []
mol_id=[]

for i in df2.standard_value:
  if float(i)>=10000:
    bioactivity_class.append("inactive")
  elif float(i)<=1000:
    bioactivity_class.append("active")
  elif (float(i)>1000 and float(i)<10000):
    bioactivity_class.append("intermediate")

for i in df2.molecule_chembl_id:
  mol_id.append(i)

for i in df2.canonical_smiles:
  canonical_smiles.append(i)

for i in df2.standard_value:
  standard_value.append(i)

data_tuples=list(zip(mol_id, canonical_smiles, bioactivity_class, standard_value))
df3=pd.DataFrame(data_tuples, columns=['molecule_chembl_id', 'canonical_smiles', 'bioactivity_class', 'standard_value'])
print(df3)

# we can also have concatenated the bioactivity_class to df2 then directly 
# used df3=pd.DataFrame(df2, columns=['molecule_chembl_id', 'canonical_smiles', 'bioactivity_class', 'standard_value'])


df3.to_csv('bioactivity_preprocessed_data.csv', index=False)

# Now we calculate the lipinski descriptors by defining the function

# It is used for evaluating the druglikeness of compounds, based on the ADME rule with specifications as

# Molecular weight < 500 Dalton
# Octanol-water partition coefficient (LogP) < 5
# Hydrogen bond donors < 5
# Hydrogen bond acceptors < 10:

df = pd.read_csv('bioactivity_preprocessed_data.csv')
print(df)

def lipinski (smiles, verbose=False):
  mol_data=[]
  for elem in smiles:
    mol=Chem.MolFromSmiles(elem)
    mol_data.append(mol)

  base_data=np.arange(1,1)
  k=0

  for mol in mol_data:
    desc_Molwt=Descriptors.MolWt(mol)
    desc_Mollogp=Descriptors.MolLogP(mol)
    desc_NumHDonors=Lipinski.NumHDonors(mol)
    desc_NumHAcceptors=Lipinski.NumHAcceptors(mol)
  
    row = np.array([desc_Molwt,desc_Mollogp,desc_NumHDonors,desc_NumHAcceptors]) 

    if(k==0):
      base_data=row
    else:
      base_data=np.vstack([base_data,row]) # now from here we can and learn that vstack is also a great presence of mind implementation
    k=k+1

  column=["MW","LogP","NumHDonors","NumHAcceptors"]
  descriptors=pd.DataFrame(data=base_data,columns=column)
  return descriptors


df_lipinski = lipinski(df.canonical_smiles)
print(df_lipinski)

# Now we will be combining both of these dataframes that are the df and df_lipinski

df_combined=pd.concat([df,df_lipinski],axis=1)
print(df_combined)

# Now we should notice that the IC50 has some non uniform values therefore we will apply pIC50
# which is the negative logarithmic value of the IC50 intially which was standard_value
# but this standard_value needs to be normalised to standard_value_norm

def norm(input):
  normalised_std_val=[]

  for i in input['standard_value']:
    if i > 100000000:
        i = 100000000
    normalised_std_val.append(i)

  input['standard_value_norm']=normalised_std_val
  x=input.drop('standard_value',1)

  return x

df_norm=norm(df_combined)
print(df_norm)

def neg_IC50(input):
  pIC50=[]
  
  for i in input['standard_value_norm']:
    molar = i*(10**-9) # Converts nM to M
    pIC50.append(-np.log10(molar))

  input['pIC50']=pIC50
  x=input.drop('standard_value_norm',1)
  return x

df_final=neg_IC50(df_norm)
print(df_final)


#now removing the intermediate bioactivity class from our bioactivity_class column

df_new=df_final[df_final.bioactivity_class != 'intermediate']
print(df_new)

# Now performing Exploratory data analysis via lipinski descriptors
sns.set(style='ticks')

#plotting the frequency plot of active and inactive class of bioactivity_class
plt.figure(figsize=(6,6))
sns.countplot(x='bioactivity_class',data=df_new, edgecolor='black')
plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('Frequency', fontsize=14, fontweight='bold')
plt.savefig('plot_freq_bioactivity_class.pdf')
# plt.show()


# Scatter plot of MW versus LogP


plt.figure(figsize=(6,6))
sns.scatterplot(x='MW', y='LogP', data=df_new, hue='bioactivity_class', size='pIC50', edgecolor='black', alpha=0.7)
plt.xlabel('MW', fontsize=14, fontweight='bold')
plt.ylabel('LogP', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
plt.savefig('plot_MW_vs_LogP.pdf')
# plt.show()

# Drawaing the Box plots

plt.figure(figsize=(6,6))
sns.boxplot(x = 'bioactivity_class', y = 'pIC50', data = df_new)
plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('pIC50 value', fontsize=14, fontweight='bold')
plt.savefig('plot_ic50.pdf')
# plt.show()


plt.figure(figsize=(6,6))
sns.boxplot(x = 'bioactivity_class', y = 'MW', data = df_new)
plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('Molecular weight', fontsize=14, fontweight='bold')
plt.savefig('plot_molecular_weight.pdf')
# plt.show()

plt.figure(figsize=(6,6))
sns.boxplot(x = 'bioactivity_class', y = 'NumHDonors', data = df_new)
plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('Number of H donors', fontsize=14, fontweight='bold')
plt.savefig('plot_numhdonors.pdf')
# plt.show()

plt.figure(figsize=(6,6))
sns.boxplot(x = 'bioactivity_class', y = 'NumHAcceptors', data = df_new)
plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('Number of H acceptors', fontsize=14, fontweight='bold')
plt.savefig('plot_numhacceptors.pdf')
# plt.show()

# Now we will be calculating molecular descriptors that are 
# essentially quantitative description of the compounds in the dataset

# We are using the the similar environment but instaed of SARS COV-2 going for acetylcholinesterase input molecule
# here in this protein we have abundance of bioactivity data therfore will be good 
# for model building

df4=pd.read_csv('acetylcholinesterase_04_bioactivity_data_3class_pIC50.csv')
print(df4)
selection=['canonical_smiles','molecule_chembl_id']
df4_selection=df4[selection]
print(df4_selection)

df4_selection.to_csv('molecule.smi', sep='\t', index=False, header=False)

# Now we calculate the fingerprint descriptors , 
# therefore here it basically means that now ideally we should load 
# padel.sh in the backend , this file performs the work of cleaning the chemical structure so that there are no impurities left
# this includes removing the salts in the structure and small organic acids, but we directly use the descriptors_output.csv
# while this file has been formed by processing bash padel.sh , while this contains the fingerprints of a lot of compounds in the structure in encoded form

df5=pd.read_csv('descriptors_output.csv')

#PubChem fingerprints, also known as PubChem substructure fingerprints or PubChem fingerprints (PCFP), 
#are a type of molecular representation used to encode the presence or absence of various chemical substructures in a molecule. 
#They are derived from the PubChem Compound database, which is a large public repository of chemical structures and associated information.

# Now creating the X and Y data to be used for model building later

df5_X=df5.drop(columns=['Name'])
df5_Y=df4['pIC50']
dataset=pd.concat([df5_X,df5_Y], axis=1)
print(dataset)

dataset.to_csv('acetylcholinesterase_final_dataset_pubchem.csv', index=False)

# The *Acetylcholinesterase* data set contains 881 input features and 1 output variable (pIC50 values).
X=dataset.drop('pIC50',axis=1)
Y=dataset['pIC50']

# Now a lot of our input features are of low variance , hence we omit them
selection2=VarianceThreshold(threshold=(.8*(1-.8)))
X=selection2.fit_transform(X)

#splitting the data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

# Building a regression model therefore using the RandomForestRegressor
model=RandomForestRegressor(n_estimators=100)
model.fit(X_train,Y_train)
#The R-squared value ranges from 0 to 1, where 1 indicates a perfect fit of the model to the data, 
#and 0 indicates that the model does not explain any of the variability in the data.
score=model.score(X_test,Y_test) # similar to r2score
print("The score for the RandomForestRegressor model is ",score)

y_pred=model.predict(X_test)

# now we know that we do not predict the accuracy for the regressive models.
mse=mean_squared_error(Y_test, y_pred)
print("The mean squared error for the above model is ", mse)

# Now we can plot the results
sns.set(color_codes=True)
sns.set_style("white")

#regplot() is for regression plots
axis=sns.regplot(x=Y_test,y=y_pred,scatter_kws={'alpha':0.4})
axis.set_xlabel('Original pIC50 values',fontsize='large',fontweight='bold')
axis.set_ylabel('Predicted pIC50 values',fontsize='large',fontweight='bold')
axis.set_xlim(0,12)
axis.set_ylim(0,12)
axis.figure.set_size_inches(6,6)
plt.savefig('Scatterplot_RandomForestRegressor.pdf')
# plt.show()

# Now we will be comparing several machine learning algorithms and analyzing the results obtained
clf=LazyRegressor(verbose=0,ignore_warnings=True,custom_metric=None)
models_train,prediction_train=clf.fit(X_train,X_train,Y_train,Y_train)
models_test,predictions_test=clf.fit(X_train,X_test,Y_train,Y_test)

print(prediction_train)
print(predictions_test)

