import numpy as np

def energy(X,W_fixed):
    """ 
    Input X is a vector (N by 1), and the shape of W_fixed is N by N.\n
    Return a 2D array E with shape (1,1)
    """
    return (-1/2) * np.dot(np.dot(np.transpose(X),W_fixed),X)

def f(W,X):
    """
    W : W_fixed after storaging phase with shape (N,N) \n
    X : testing pattern X with shape (N,1) \n
    Return X_new with shape (N,1)
    """
    i = 0
    v = np.dot(W,X)
    for value in v :        
        if value > 0 :
            v[i,0] = 1
        elif value < 0 :
            v[i,0] = -1
        else :
            v[i,0] = X[i,0]
        i += 1
    return v    

X0 = np.array([ 1,-1, 1,-1, 1,-1])
X2 = np.array([-1, 1,-1, 1,-1, 1])
X4 = np.array([ 1, 1, 1, 1, 1, 1])
X6 = np.array([-1,-1,-1,-1,-1,-1])

X0      = X0.reshape(1,X0.size)
X2      = X2.reshape(1,X2.size)
X4      = X4.reshape(1,X4.size)
X6      = X6.reshape(1,X6.size)
X       = np.array([X0,X2,X4,X6])
W_fixed = np.zeros([X0.shape[1],X0.shape[1]])

for p in range(X.shape[0]):      
    W_fixed = W_fixed + np.dot(X[p,:,:].reshape(-1,1),X[p,:,:].reshape(1,-1))  
for i in range(X.shape[2]):
    for j in range(X.shape[2]):
        if i==j:
            W_fixed[i,j]=0

X_test = X0.reshape(-1,1)                    #-- change label index
E = energy(X_test,W_fixed)
#------------  obtain new X  -------------
E_cycle = [E]
print('-------------------------------------')
print('E(0) : ',E[0,0])
print('X(0) : ',X_test[:,0])
X_new = X_test

for cycle in range(100):
    X_new = f(W_fixed,X_new)
    E = energy(X_new,W_fixed)
    E_cycle.append(E[0,0])
    
    print('-------------------------------------')
    print('E({}) : {}'.format(cycle+1,E[0,0]))
    print('X({}) : {}'.format(cycle+1,X_new[:,0]))
    
    if E_cycle[cycle+1] == E_cycle[cycle] :
        print('-------------------------------------')
        print('cycle :',cycle+1)
        print('E converges : {}'.format(E[0,0]))
        print('X final : {}'.format(X_new[:,0]))
        break