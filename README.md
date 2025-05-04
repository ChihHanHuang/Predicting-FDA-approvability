<p align="center">
  <img src="https://github.com/ChihHanHuang/Predicting-FDA-approvability/blob/main/Graphical_abstract.png">
</p>

# Predicting FDA approvability of small-molecule drugs
This study has been published and is available at the following link:

https://www.biorxiv.org/content/10.1101/2022.10.15.512330v1

Huang, C. H., Hsu, J., Yang, L. Y., Chen, T. M., Shih, E. S., & Hwang, M. J. (2022). Predicting FDA approvability of small-molecule drugs. bioRxiv, 2022-10.

This work suggests that machine learning can provide the possibility of whether a drug can enter clinical trial, be approved by government, or be withdrawn from the market.

## Data
The data folder contains the dateset used to train the models proposed in this work. 

## Requirements
You'll need to install the following in order to run the code
* Python 3.7
* scikit-learn 0.23.1 
* rdkit
* pubchempy
* joblib 
* pandas

## Usage 
To predict FDA-approvability of new small-molecule compounds of your interest with our trained models, you'll need to prepare a list of PubChem Compound ID (CID) for the compounds and save it in a txt file in current directory. Follow instructions in README in the model folder for the model files. 

For example, the list of PubChem CIDs of compounds of interest is saved as 'testComps.txt'. Run
```
python predict.py testComps.txt
```
This returns a 'result.csv' file, which contains a table like the following:

| PubChemID  | non-TD/TD | MD/CD | MDon/MDoff |
| ---------- | --------- | ----- | ---------- |
| 9822750 | non-TD (10) | MD (10) | MDon (7) |
| 5834  | non-TD (10)  | MD (10) | MDon (10) |
| 6675  | non-TD (10)  | MD (7) | MDon (10) |
| 28693  | non-TD (10)  | CD (10) | MDoff (10) |
| 155774  | non-TD (10)  | MD (10) | MDon (9) |

This table used the newly FDA-approved drugs from our study as example. The results are predicted by the 10 models of the three classification (non-TD/TD, MD/CD, and MDon/MDoff). Number in parenthesis indicates the number of models predicting the class. TD, toxic compound; CD, drug in clinical trial; MD, drug approved by FDA; MDoff, drug withdrawn from the market; MDon, drug currently on the market. 
