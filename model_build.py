from torch import tensor,empty, zeros, exp, square, cat, numel, randn
from torch import sum as tensor_sum
from torch.nn.init import xavier_normal_
from torch.optim import Adam
import torch.nn as nn 

def sigmoid(x):
    return 1/ (1+exp(-x))

def tanh(x):    
    return (exp(x) - exp(-x))/(exp(x) + exp(-x))

def mse(y_obs, y_pred):
    return (1/numel(y_obs)) * tensor_sum(square(y_obs - y_pred)) 


class LSTMModel():
    def __init__(self,  input_dim=1, hidden_size=64, layers=1,output_dim=1):
        
        self.sigmoid = sigmoid
        self.tanh = tanh
        
        self.hidden_size = hidden_size 
        
        concat_size = hidden_size + input_dim

        self.W_f = xavier_normal_(empty(hidden_size,concat_size))
        self.b_f = zeros(hidden_size, requires_grad=True)  
        
        self.W_i = xavier_normal_(empty(hidden_size,concat_size))
        self.b_i = zeros(hidden_size, requires_grad=True)  
        
        self.W_c = xavier_normal_(empty(hidden_size,concat_size))
        self.b_c = zeros(hidden_size, requires_grad=True)  
        
        self.W_o = xavier_normal_(empty(hidden_size,concat_size))
        self.b_o = zeros(hidden_size, requires_grad=True)  
   
    def forward(self, x):
        batch_size = x.shape[0]
 
        c_0 = zeros(batch_size, self.hidden_size)
        h_0 = zeros(batch_size, self.hidden_size)
        
        concat_xh = cat([x,h_0], dim=1 )        
        
        forget_gate = self.sigmoid(concat_xh @ self.W_f.T + self.b_f)
        input_gate = self.sigmoid(concat_xh @ self.W_i.T + self.b_i)
        cell_gate = self.tanh(concat_xh @ self.W_c.T + self.b_c)
        output_gate = self.sigmoid(concat_xh @ self.W_o.T + self.b_o)
        
        self.c_t = forget_gate * c_0 + cell_gate * input_gate
        self.h_t = output_gate * self.tanh(self.c_t)
        
        return self.h_t, self.c_t

model = LSTMModel()
x_test = randn(3,1)   
model.forward(x_test)
params = list(model.__dict__.keys())
params = params[3:-2] # only optimize the weights and biases and not all the other nonsense

lr = 1e-3
optimizer = Adam(parms, lr)


def training(dataloader, model, optimizer, loss_fn):
    model.train()
    size = len(dataloader.dataset)
    linear = nn.Linear(64,1)
     
    for batch, (x_data, y_data) in enumerate(dataloader):
        batch_size = x_data.shape[0]  
        h_t, c_t = model(x_data, h_0, c_0)
       
        y_linear = linear(h_t)
        loss = loss_fn(y_linear, y_data)
   
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
    

