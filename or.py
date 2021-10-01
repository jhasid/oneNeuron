from utils.model import Perceptron
from utils.all_utils import prepare_data , save_model, save_plot
import pandas as pd
import numpy as np

OR = {"x1":[0,0,1,1],
       "x2":[0,1,0,1],
       "y":[0,1,1,1]
       }

df = pd.DataFrame(OR)       
print(df)

x,y = prepare_data(df)
print(f"prepare data x  value :{x}")
print(f"prepare data y  value :{y}")
eta = 0.3 #anything bw 0 and 1
epochs = 10

model = Perceptron(eta=eta,epochs=epochs)
model.fit(x,y)

_ = model.total_loss()

save_model(model,filename="or.model")

save_plot(df,"orPlot.png",model)
