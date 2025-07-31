

import pandas as pd
import numpy as np
from sklearn import preprocessing
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import loadbond_data




# ----Reading in data----


vix_price, bond_yield, fedfunds_rate = loadbond_data.read_data(
    vix_path=r'C:\Users\PCadmin\Downloads\VIX_History.csv'
    ,bond_path=r'C:\Users\PCadmin\Downloads\bond_prices2.csv'
    ,fedfunds_path=r'C:\Users\PCadmin\Downloads\fedfundsrate.csv')

vix_price.rename(columns={ vix_price.columns[0] : 'Date',
                          vix_price.columns[4] : 'VIX 5-Day Returns'}
                         ,inplace=True)
vix_price.drop(columns=['OPEN','HIGH','LOW',], inplace=True)


# changing dtype of dates from str to datetime64[ns] for time series calculations

def convert_datetime(dates_1,dates_2):
    new_dates_1 = pd.to_datetime(dates_1)
    new_dates_2 = pd.to_datetime(dates_2)
    return new_dates_1, new_dates_2


vix_price['Date'], bond_yield['observation_date'] = convert_datetime(
            dates_1=vix_price['Date'],
            dates_2=bond_yield['observation_date'])

# -----renaming columns, merging datasets------

bond_yield['Federal Funds Rate'] = fedfunds_rate['DFF']
bond_yield.rename(columns={bond_yield.columns[0]: 'Date',
                           bond_yield.columns[1]: '10-Year Treasury Bond Yield'},
                  inplace=True)
jan5th_yield = vix_price['VIX 5-Day Returns'].pct_change(periods=3) *100
vix_price['VIX 5-Day Returns'] = vix_price['VIX 5-Day Returns'].pct_change(periods=5) *100
vix_price.iloc[3,1] = jan5th_yield.iloc[3]

mixed_df = bond_yield.merge(vix_price, on='Date', how='inner') 
fixeddf = mixed_df[mixed_df.columns[1:]]
means = np.mean(fixeddf, axis=0)
std = np.std(fixeddf,axis=0)

# Keeping means and std for destandazrding values 

bond_mu, ffr_mu, vix_mu = means.iloc[0],means.iloc[1],means.iloc[2]
bond_sigma, ffr_sigma, vix_sigma = std.iloc[0],std.iloc[1],std.iloc[2]

# ---- standardizing the data vals ----

def standardize_data(dataframe, data_points ):
    selected_cols = dataframe.columns[1:]
    scalar = preprocessing.StandardScaler()
    
    scalar.fit(dataframe[selected_cols])
    scaled_data = scalar.transform(dataframe[selected_cols])
    table_scaled = pd.DataFrame(scaled_data, columns=['Scaled Yield'
                                       ,'Scaled Fed Funds Rate',
                                       'Scaled 5-Day Vix Returns']
                                        )
    scaled_data= []
    for col in table_scaled.columns:
        data_points = table_scaled[col].to_numpy()
        scaled_data.append(data_points)
        
    return table_scaled, scaled_data


scaled_df, data_points = standardize_data(mixed_df,True)


# lagging sequences, looking using a data point x at a time t-n to predict future values at time t+1
# past yield data, federal funds rate, and vix 5-day returns are 20 timesteps behind


def shape_data(data, timesteps):
    X, y = [], []
    
    for i in range(len(data) - timesteps):
        X.append(data[i:i + timesteps])
        y.append(data[i + timesteps,0:1])
    
    
    
    return np.array(X) , np.array(y) 


scaled_vector =  scaled_df.to_numpy()
X,y = shape_data(scaled_vector, 20)

# convert to tensor and tensor dataset for more efficent batching

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)


X_tensor_train, X_tensor_test, X_tensor_val = X_tensor[:1500] , X_tensor[1500:1700], X_tensor[1700:]  
y_tensor_train, y_tensor_test, y_tensor_val  = y_tensor[:1500] , y_tensor[1500:1700], y_tensor[1700:]

data_Traintensor = TensorDataset(X_tensor_train,y_tensor_train)

data_Testtensor = TensorDataset(X_tensor_test, y_tensor_test)

train_dataset = DataLoader(data_Traintensor,batch_size = 32, shuffle=True)
test_dataset = DataLoader(data_Testtensor,batch_size = 32, shuffle=False)
train_features, test_features = next(iter(train_dataset))

# model arch

class LSTMModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, out_features=output_size)
    
    def forward(self, x):
        out,_ = self.lstm(x)
        out = self.linear(out[:,-1,:])
        return out 
        


model = LSTMModel()
batch_size = 32
lr = 1e-3

def training(dataloader, model, optimizer, loss_fn):
    size = len(dataloader.dataset)
    model.train()
    for batch, (x_data, y_data) in enumerate(dataloader):
        y_pred = model(x_data)
        loss = loss_fn(y_pred, y_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
                
        
        
        if batch % 10 == 0:
            loss, current = loss.item(), batch * batch_size + len(x_data)
            print(f'loss: {loss:.5f} [{current :>5d}/{size:>5d}]')
     
            
def testing(dataloader, model, loss_fn):
    model.eval()
    num_batches = len(dataloader)
    test_loss= 0
    with torch.no_grad():
        for x, y in dataloader:
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
            
    test_loss /= num_batches 
    print(f'Test Error: \n|Avg loss: {test_loss:>8f}|\n')       


crit = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
epochs = 55

if __name__ == '__main__':
    
    for t in range(epochs):
        print(f'epoch: {t+1}\n -----------------------')
        training(train_dataset, model, optimizer, crit)
        testing(test_dataset, model, crit)
        print('Done')
    

