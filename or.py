from utils.model import Perceptron
from utils.all_utils import prepare_data , save_model, save_plot
import pandas as pd
import numpy as np
import logging
import os

logging_str ="[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
#logging.basicConfig(level=logging.INFO,format=logging_str)
log_dir = "logs"
os.makedirs(log_dir,exist_ok=True)
#logging.basicConfig(filename=os.path.join(log_dir,"orGateModelTrainingLogs.logs"),level=logging.INFO,format=logging_str)

logging.basicConfig(filename=os.path.join(log_dir,"orGateModelTrainingLogs.logs"),level=logging.INFO,format=logging_str,filemode="a")

def main(data,eta,epochs,modelfilename,plotfilename):
  df = pd.DataFrame(data)      
  #print(df)
  logging.info(f"This is actual dataframe {df}")

  x,y = prepare_data(df)
  logging.info(f"prepare data x  value :{x}")
  logging.info(f"prepare data y  value :{y}") 
  eta = 0.3 #anything bw 0 and 1
  epochs = 10

  model = Perceptron(eta=eta,epochs=epochs)
  model.fit(x,y)

  _ = model.total_loss()

  save_model(model,filename=modelfilename)

  save_plot(df,plotfilename,model)

if __name__ == '__main__':
  OR = {"x1":[0,0,1,1],
          "x2":[0,1,0,1],
          "y":[0,1,1,1]
         }




  eta = 0.3 #anything bw 0 and 1
  epochs = 10       
  try:
    logging.info(">>>>>>>> Starting the model training >>>>>>>>>>>>>>>>>")
    main(data=OR,eta=eta,epochs=epochs,modelfilename="or.model",plotfilename="orPlot.png")
    #main(data=OR,eta=eta,epochs=epochs,modelfilename="or.model",plotfilename="orPlot.png",test=1)
    logging.info(">>>>>>>>> Training Finished successfully <<<<<<<<<<<<<<\n")
  except Exception as e:
    logging.info(e)  
    raise e   #incase we want error in terminal
