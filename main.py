import numpy as np

# Gives 100 equally spaced values from -pi to pi
x = np.linspace(-np.pi, np.pi, 100)
y = np.sin(x) # the answer of problem .. since its easily calculatable


from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(x.reshape(-1, 1), y) #This is where training happens, it takes 100 x and y vallues and finds ....
                                # ....the best cofficient for the linear formula y = mx + c


print("Intercept:", model.intercept_) #c value
print("Slope:", model.coef_[0]) #m value

y_pred = model.predict(x.reshape(-1, 1)) # checking the output of training

#Both curves in plotting
import matplotlib.pyplot as plt
plt.plot(x, y, label="True sin(x)")
plt.plot(x, y_pred, label="Linear model")

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

'''After looking at this graph, I realized that linear method we 
     are trying can not get anywhere close to getting sinx right.
     It is very obvious but for some reasoon I thought it would get close 
     to it. It ofcourse can't. So, is this the limitation of ML? :(
'''


'''And another thing, the line is closest we can get to sinx curve using this method. 
    So, we can say that this is the best linear approximation of sinx curve. 
    So, My project is over right?? just 20 lines of code and 30 lines of comments XD
    Our model was actually a very good model, it literally gave us the best linear approximation of sinx curve.
    Hence, this model is a good model. But, we can not use this method to get sinx curve.
'''

# Hence, I conclude that linear regression is not a good method to get  sinx curve.