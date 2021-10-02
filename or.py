from utils.model import Perceptron
from utils.all_utils import prepare_data , save_model, save_plot
import pandas as pd
import numpy as np

def main(data,eta,epochs,modelfilename,plotfilename):


df = pd.DataFrame(data)      
print(df)

x,y = prepare_data(df)
print(f"prepare data x  value :{x}")
print(f"prepare data y  value :{y}")
eta = 0.3 #anything bw 0 and 1
epochs = 10

model = Perceptron(eta=eta,epochs=epochs)
model.fit(x,y)

_ = model.total_loss()

save_model(model,filename=modelfilename)

save_plot(df,plotfilename,model)

if __name__ == '__main__': #entry point

    OR = {"x1":[0,0,1,1],
          "x2":[0,1,0,1],
          "y":[0,1,1,1]
         }




     eta = 0.3 #anything bw 0 and 1
     epochs = 10       
       
       main(data=OR,eta,epochs,modelfilename="or.model",plotfilename="orPlot.png")
