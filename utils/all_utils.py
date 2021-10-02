import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib    #for saving model as lib
from matplotlib.colors import ListedColormap
import os
import logging
plt.style.use("fivethirtyeight")  #style of graphs

def prepare_data(df):
  """it prepares data by separating features (dependent) and labels (independent)

  Args:
      df (pd.DataFrame): takes dataFrame as input

  Returns:
      tuple: returns tuples  of dependent and independent variables
  """
  logging.info("preparing data by separating independent and dependent variables")
  x = df.drop("y",axis =1)
  y = df["y"]
  return x,y

def save_model(model,filename):
  """saves model

  Args:
      model (python object): trained model
      filename (str): path to save model name
  """
  logging.info("saving the model")
  model_dir = "models"
  os.makedirs(model_dir,exist_ok=True) #create only when mode_dir doesn't exist
  filepath = os.path.join(model_dir,filename) #concatenate model_dir/filename
  joblib.dump(model,filepath)
  logging.info(f"saved the model at {filepath}")

def save_plot(df,filename,model):
  """
    :param df : its a dataframe
    :param filename : its path to save the plot
    :param model : trained model
  """
  logging.info("start plotting the model")
  def _create_base_plot_(df):
    logging.info("creating a base plot")
    df.plot(kind="scatter",x="x1",y="x2",c="y",s=100,cmap="winter")
    plt.axhline(y=0,color="black",linestyle="--",linewidth=1)
    plt.axvline(x=0,color="blue",linestyle="--",linewidth=1)
    
    figure = plt.gcf() #get current figure
    figure.set_size_inches(10,8)

  def _plot_decision_region_(x,y,classifier,resolution=0.2):
    logging.info("plotting decision region")
    colors =("red","blue","green","pink","gray")
    cmap = ListedColormap(colors[: len(np.unique(y))])  #bases of y value colors,here there are 2 so 2 colors

    x = x.values #as array
    x1_min,x1_max = x[:,0].min() -1,x[:,0].max() +1
    x2_min,x2_max = x[:,1].min() -1,x[:,1].max() +1

    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution), 
                         np.arange(x2_min,x2_max,resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,Z,alpha=0.2,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    plt.plot()

  x,y = prepare_data(df)
  _create_base_plot_(df)   
  _plot_decision_region_(x,y,model)

  plot_dir = "plots"
  os.makedirs(plot_dir,exist_ok=True)
  plotpath = os.path.join(plot_dir,filename)
  plt.savefig(plotpath)  
  logging.info(f"saving the plot at { plotpath}")