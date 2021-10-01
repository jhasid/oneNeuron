import numpy as np

class Perceptron:
  def __init__(self,eta,epochs):        #eta is learning rate
    self.weights = np.random.randn(3) * 1e-4   #making small weight init
    print(f"initial weights before training :\n {self.weights}")
    self.eta = eta
    self.epochs = epochs

  def activationfunction(self,inputs,weights):
    z = np.dot(inputs,weights)  # z = w * x  a matrix
    print("#"*10)
    #print(f"activation  z vale :\n{z}")
    #print("#"*10)
    return np.where(z > 0,1,0)

  def fit(self,x,y):
     self.x = x
     self.y = y
     x_with_bias = np.c_[self.x,-np.ones((len(self.x),1))] #concatenation of x and bias to give matrix
     print(f"x with bais : \n{x_with_bias}")

     for epoch in range(self.epochs):
       print("#"*10)
       print(f"for epoch :{epoch}")
       print("#"*10)

       y_hat = self.activationfunction(x_with_bias,self.weights)  # z value 0 or 1  #forward propogation
       print(f"predicted value after forward pass :\n{y_hat}")
       print(f"expected value after forward pass :\n{y}")
       self.error = self.y - y_hat  #error = y - y_hat
       print(f"error :\n{self.error}")
       self.weights = self.weights + self.eta * np.dot(x_with_bias.T,self.error) # new weight = old weight + n(y_hat) #backward propogation
       print(f"update weights after epoch :\n{epoch}/{self.epochs} : {self.weights}")
       print("#"*10)

  def predict(self,x):
     x_with_bias = np.c_[x,-np.ones((len(x),1))]
     return self.activationfunction(x_with_bias,self.weights)  # x with updated weights

  def total_loss(self):
      total_loss = np.sum(self.error)   
      print(f"total loss :{total_loss}")
      return total_loss