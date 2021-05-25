import numpy as np
import numpy.ma as ma
import pandas as pd
from rdkit.Chem import MACCSkeys
from rdkit import Chem
import pubchempy as pcp 
import joblib
import sys
import json

import warnings
warnings.filterwarnings("ignore")
#feature extraction
def get_feature(PubChem_ID):
    feature=[]
    physioCP = ['atom_stereo_count','bond_stereo_count','canonical_smiles','charge',
                'complexity','covalent_unit_count','defined_atom_stereo_count','defined_bond_stereo_count',
                'h_bond_acceptor_count','h_bond_donor_count','heavy_atom_count','isotope_atom_count',
                'molecular_weight','monoisotopic_mass','rotatable_bond_count','tpsa',
                'undefined_atom_stereo_count','undefined_bond_stereo_count','xlogp', 'cid']
    for cid in PubChem_ID:
        try:
            c = pcp.Compound.from_cid(cid)
            c_feature=pcp.compounds_to_frame(c, properties = physioCP)
            c_f_columns = c_feature.columns.tolist()
            c_f_columns.extend(['cid'])
            c_feature["cid"]=c_feature.index
            c_feature=c_feature.values.tolist()
            feature.append(c_feature[0])
           #print(cid)
        except:
            continue
            
    feature=pd.DataFrame(feature)
    feature.columns = c_f_columns
    feature['molecular_weight']=feature.molecular_weight.astype(float)
    feature['monoisotopic_mass']=feature.monoisotopic_mass.astype(float)
    feature.cid=feature.cid.astype(str)
    feature = feature[physioCP]

    return feature

def mw_over(feature):
    molecular_weight = np.array(feature['molecular_weight'])
    ind = np.where(molecular_weight > 1500)
    cid_over_mw = feature.iloc[ind]['cid']
    print('The following compounds exceed MW of 1500 Dalton:')
    print(cid_over_mw)
    index = np.where(molecular_weight <= 1500)
    feature = feature.iloc[index]
    return feature 

def smi_to_maccs(smi):
    MACCS_SIZE = 167
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        return np.array(MACCSkeys.GenMACCSKeys(mol))
    else:
        return np.zeros(MACCS_SIZE)
    
def get_maccs(SMILES):
    MACC = []
    for i, smi in enumerate(SMILES):
        try:
            macc = smi_to_maccs(smi)
            MACC.append(macc)
        except:
            continue
    MACC=pd.DataFrame(MACC)
    MACC = MACC.drop([0],axis= 1)
    MACC.columns = range(MACC.shape[1])
    return MACC

def get_test_feature(PubchemID):
    Feature = get_feature(PubchemID)
    Feature = mw_over(Feature)
    SMILES = Feature['canonical_smiles']
    MACCs= get_maccs(SMILES)
    MACCs['cid'] = Feature['cid']
    F_fea = pd.merge(Feature, MACCs, on="cid")
    return F_fea

def fill(a):
    a=np.where(np.isnan(a), ma.array(a, mask=np.isnan(a)).mean(axis=0), a)
    print()
    return a

def predict(newD_feature):
    X_test=newD_feature.drop(columns=['canonical_smiles','cid'])
    X_test = np.array(X_test)
    X_test = fill(X_test)

    newD_pred=pd.DataFrame(newD_feature['cid'].values,columns=['PubChemID'])
    newD_pred['non-TD/TD']=0
    newD_pred['MD/CD']=0
    newD_pred['MDon/MDoff']=0


    for i in range(10):

        TD_NTD_model = joblib.load('models/model_NTD_TD/model_NTD_TD_'+str(i)+'.pkl')
        CD_MD_model = joblib.load('models/model_MD_CD/model_MD_CD_'+str(i)+'.pkl')
        MDon_off_model = joblib.load('models/model_MDon_off/model_MDon_MDoff_'+str(i)+'.pkl')

        TD_NTD_predict = TD_NTD_model.predict(X_test)
        CD_MD_predict = CD_MD_model.predict(X_test)
        MDon_off_predict = MDon_off_model.predict(X_test)

        for j in range(len(X_test)):
            newD_pred['non-TD/TD'][j] += TD_NTD_predict[j]
            newD_pred['MD/CD'][j] += CD_MD_predict[j]
            newD_pred['MDon/MDoff'][j] += MDon_off_predict[j]


    column = ['non-TD/TD','MD/CD','MDon/MDoff']
    pos = ['non-TD','MD','MDon']
    neg = ['TD','CD','MDoff']

    for i, row in newD_pred.iterrows():
        for j in range(3):
            if row[column[j]] >= 5:
                count = row[column[j]]
                newD_pred.loc[i,column[j]] = pos[j]+ ' (' +str(count) +')'
            else:
                count = 10 - row[column[j]]
                newD_pred.loc[i,column[j]] = neg[j]+ ' (' +str(count) +')'
                
    return newD_pred

if __name__ == "__main__":
    filename = str(sys.argv[1])
    print('predicting FDA approvability for ' + str(filename))
    with open(filename, "r") as fp:
        DrugID = json.load(fp)
        
    feature = get_test_feature(DrugID)
    prediction = predict(feature)
    prediction.to_csv("result.csv", index = None)
    print("prediction complete!")
