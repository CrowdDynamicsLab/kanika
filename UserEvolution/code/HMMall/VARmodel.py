import numpy as np
import pandas
import statsmodels.api as sm
from statsmodels.tsa.api import VAR, DynamicVAR

def VarModel(UserFile):
    Users = set()
    data = []
    with open(UserFile, 'r') as f:
        for line in f:
            if "User" in line and len(Users)!=0:
                #estimate the VarModel
                print("Length of user sessions are ", len(data))
                data = []
            elif "User" not in line and line:
                session_data = line.strip().split("\t")[1:]
                data.append(session_data)



























file = '/Users/kanika/Dropbox/UserEvolution/individual_evolution/MatlabStackExchangeData/english/SessionData.txt'
VarModel(file)
