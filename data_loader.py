import numpy as np
import requests



# API Key to grab data from FRED website

api_key = 'd48605b363c3ec002b7dfc61977e451c'


base_url = 'https://api.stlouisfed.org/fred/series/observations'

class Data_Loader():
    
    def __init__(self, api_key, series_id ,url=base_url, data_values=None, mean=None,stdev=None,data_dates=None):
        
        self.api_key = api_key
        self.url = url
        self.series_id = series_id
        
        
        self.mean = mean
        self.stdev = stdev
        
        self.data_vals = data_values
        self.data_dates = data_dates

    def get_data(self, file_type, start_date=None, end_date=None):
        
       full_url = self.url + '?series_id={}&'.format(self.series_id)+'api_key={}'.format(self.api_key)+'&file_type={}'.format(file_type) 
    
       params = {
               'observation_start' : start_date,
               'observation_end' : end_date,
               } 
            
       res = requests.get(full_url, params=params)
       data = res.json()
       
        
       n = len(data['observations'])
       data = data['observations']
       

       self.data_vals = np.zeros(n)
       self.data_dates = [] 
        
       for i in range(n):
           
           if data[i]['value'] == '.':
              data[i]['value'] = 4.50 
           else:
                data[i]['value'] = float(data[i]['value']) 
          
           self.data_vals[i] = data[i]['value']
           
           self.data_dates.append(data[i]['date'])    
           
       self.mean = np.mean(self.data_vals)
       
       self.stdev = np.std(self.data_vals)
    

    def standardize(self):
        
        self.data_vals = (self.data_vals - self.mean)/self.stdev
        
    def prepare_data(self, steps, cross_valid=None):
        
        data = self.data_vals.reshape(self.data_vals.shape[0],1)

        x_data, y_data = [], []
        
        for i in range(len(data) - steps):

            x_data.append(data[i:i+steps,])
            y_data.append(data[i+steps,])
            
        x_data = np.array(x_data)
        y_data = np.array(y_data)
       
        if cross_valid == True:
           
            x_folds = np.array_split(x_data,5)
            y_folds = np.array_split(y_data,5)    
            return x_folds, y_folds
        
        elif cross_valid == False:
            num_train = 14000
            num_test = 15000
            x_splits = np.split(x_data,[num_train,num_test])
            y_splits = np.split(y_data,[num_train,num_test])

            return x_splits, y_splits 
        
        elif cross_valid == None:
            raise NameError('Need to set cross_valid True or False')


start_date = '1776-07-04'
end_date = '9999-12-31'
hold = Data_Loader(api_key=api_key, series_id='DGS10')

data = hold.get_data('json', start_date=start_date, end_date=end_date)
standard_data = hold.standardize()
       
print(hold.prepare_data(10,False))

