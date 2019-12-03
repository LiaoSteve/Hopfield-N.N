import numpy as np 
import matplotlib.pyplot as plt
import csv

def bw(X):
    '''
    input X with shape (10,10)
    '''
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i,j] == -1:
                X[i,j] = 255
            elif X[i,j] == 1:
                X[i,j] = 0
            else :
                X[i,j] = 100
    return X

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

def add_noise(X,percent):
    '''
    The shape of X is 10 by 10. \n
    Percent is the percentage noise of X between 0 ~ 1.\n 
    Return the X with percentage of noise
    '''
    n = int(100 * percent)
    pos = np.random.randint(0,10,[n,2])
    for i in enumerate(pos):        
        if X[i[1][0],i[1][1]] == -1:
            X[i[1][0],i[1][1]] = 1
        elif X[i[1][0],i[1][1]] == 1:
            X[i[1][0],i[1][1]] = -1
        else:
            pass
    return X

def add_unknown(X,percent):
    '''
    The shape of X is 10 by 10. \n
    Percent is the percentage unknown of X.(note that percentage is 0,0.1,0.2,0.3,...,1.0)\n 
    Return the X with percentage of unknown
    '''
    n = int((100 * percent)/10)
    row = X.shape[0] - n    
    for i in range(row,X.shape[0]):
        for j in range(X.shape[1]):        
            X[i,j] = 0
    return X
#trining pattern
X0 = np.array([ [-1,-1,-1, 1, 1, 1, 1,-1,-1,-1],
                [-1, 1, 1, 1, 1, 1, 1, 1, 1,-1],
                [-1, 1, 1,-1,-1,-1,-1, 1, 1,-1],
                [ 1, 1, 1,-1,-1,-1,-1, 1, 1, 1],
                [ 1, 1, 1,-1,-1,-1,-1, 1, 1, 1],
                [ 1, 1, 1,-1,-1,-1,-1, 1, 1, 1],
                [ 1, 1, 1,-1,-1,-1,-1, 1, 1, 1],
                [-1, 1, 1,-1,-1,-1,-1, 1, 1,-1],
                [-1, 1, 1, 1, 1, 1, 1, 1, 1,-1],
                [-1,-1,-1, 1, 1, 1, 1,-1,-1,-1]])

X2 = np.array([ [-1, 1, 1, 1, 1, 1, 1, 1, 1,-1],
                [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [-1,-1,-1,-1,-1,-1,-1,-1, 1, 1],
                [-1,-1,-1,-1,-1,-1,-1,-1, 1, 1],
                [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [ 1, 1, 1, 1, 1, 1, 1, 1, 1,-1],
                [ 1, 1,-1,-1,-1,-1,-1,-1,-1,-1],
                [ 1, 1,-1,-1,-1,-1,-1,-1,-1,-1],
                [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

X4 = np.array([ [-1,-1,-1,-1, 1, 1,-1, 1, 1,-1],
                [-1,-1,-1, 1, 1,-1,-1, 1, 1,-1],
                [-1,-1, 1, 1,-1,-1,-1, 1, 1,-1],
                [-1, 1, 1,-1,-1,-1,-1, 1, 1,-1],
                [ 1, 1,-1,-1,-1,-1,-1, 1, 1,-1],
                [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [-1,-1,-1,-1,-1,-1,-1, 1, 1,-1],
                [-1,-1,-1,-1,-1,-1,-1, 1, 1,-1],
                [-1,-1,-1,-1,-1,-1,-1, 1, 1,-1]])

X6 = np.array([ [-1,-1, 1, 1, 1, 1, 1,-1,-1,-1],
                [-1, 1, 1, 1, 1, 1, 1, 1,-1,-1],
                [-1, 1, 1,-1,-1,-1,-1,-1,-1,-1],
                [-1, 1, 1, 1, 1, 1, 1,-1,-1,-1],
                [-1, 1, 1, 1, 1, 1, 1, 1,-1,-1],
                [-1, 1, 1,-1,-1,-1, 1, 1,-1,-1],
                [-1, 1, 1,-1,-1,-1, 1, 1,-1,-1],
                [-1, 1, 1,-1,-1,-1, 1, 1,-1,-1],
                [-1, 1, 1, 1, 1, 1, 1, 1,-1,-1],
                [-1,-1, 1, 1, 1, 1, 1,-1,-1,-1]])

X8 = np.array([ [-1,-1,-1, 1, 1, 1, 1,-1,-1,-1],
                [-1,-1, 1, 1, 1, 1, 1, 1,-1,-1],
                [-1, 1, 1,-1,-1,-1,-1, 1, 1,-1],
                [-1, 1, 1,-1,-1,-1,-1, 1, 1,-1],
                [-1,-1, 1, 1, 1, 1, 1, 1,-1,-1],
                [-1, 1, 1, 1, 1, 1, 1, 1, 1,-1],
                [ 1, 1,-1,-1,-1,-1,-1,-1, 1, 1],
                [ 1, 1,-1,-1,-1,-1,-1,-1, 1, 1],
                [-1, 1, 1, 1, 1, 1, 1, 1, 1,-1],
                [-1,-1, 1, 1, 1, 1, 1, 1,-1,-1]])

X0_2=np.array([ [-1,-1,-1, 1, 1, 1, 1,-1,-1,-1],
                [-1, 1, 1, 1, 1, 1, 1, 1, 1,-1],
                [-1, 1, 1,-1,-1,-1,-1, 1, 1,-1],
                [ 1, 1, 1,-1,-1,-1,-1, 1, 1, 1],
                [ 1, 1, 1,-1,-1,-1,-1, 1, 1, 1],
                [ 1, 1, 1,-1,-1,-1,-1, 1, 1, 1],
                [ 1, 1, 1,-1,-1,-1,-1, 1, 1, 1],
                [-1, 1, 1,-1,-1,-1,-1, 1, 1,-1],
                [-1, 1, 1, 1, 1, 1, 1, 1, 1,-1],
                [-1,-1,-1, 1, 1, 1, 1,-1,-1,-1]])

X4_2=np.array([ [-1,-1,-1,-1, 1, 1,-1, 1, 1,-1],
                [-1,-1,-1, 1, 1,-1,-1, 1, 1,-1],
                [-1,-1, 1, 1,-1,-1,-1, 1, 1,-1],
                [-1, 1, 1,-1,-1,-1,-1, 1, 1,-1],
                [ 1, 1,-1,-1,-1,-1,-1, 1, 1,-1],
                [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [-1,-1,-1,-1,-1,-1,-1, 1, 1,-1],
                [-1,-1,-1,-1,-1,-1,-1, 1, 1,-1],
                [-1,-1,-1,-1,-1,-1,-1, 1, 1,-1]])

X2_2=np.array([ [-1, 1, 1, 1, 1, 1, 1, 1, 1,-1],
                [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [-1,-1,-1,-1,-1,-1,-1,-1, 1, 1],
                [-1,-1,-1,-1,-1,-1,-1,-1, 1, 1],
                [-1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [ 1, 1, 1, 1, 1, 1, 1, 1, 1,-1],
                [ 1, 1,-1,-1,-1,-1,-1,-1,-1,-1],
                [ 1, 1,-1,-1,-1,-1,-1,-1,-1,-1],
                [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

X6_2=np.array([ [-1,-1, 1, 1, 1, 1, 1,-1,-1,-1],
                [-1, 1, 1, 1, 1, 1, 1, 1,-1,-1],
                [-1, 1, 1,-1,-1,-1,-1,-1,-1,-1],
                [-1, 1, 1, 1, 1, 1, 1,-1,-1,-1],
                [-1, 1, 1, 1, 1, 1, 1, 1,-1,-1],
                [-1, 1, 1,-1,-1,-1, 1, 1,-1,-1],
                [-1, 1, 1,-1,-1,-1, 1, 1,-1,-1],
                [-1, 1, 1,-1,-1,-1, 1, 1,-1,-1],
                [-1, 1, 1, 1, 1, 1, 1, 1,-1,-1],
                [-1,-1, 1, 1, 1, 1, 1,-1,-1,-1]])

X8_2=np.array([ [-1,-1,-1, 1, 1, 1, 1,-1,-1,-1],
                [-1,-1, 1, 1, 1, 1, 1, 1,-1,-1],
                [-1, 1, 1,-1,-1,-1,-1, 1, 1,-1],
                [-1, 1, 1,-1,-1,-1,-1, 1, 1,-1],
                [-1,-1, 1, 1, 1, 1, 1, 1,-1,-1],
                [-1, 1, 1, 1, 1, 1, 1, 1, 1,-1],
                [ 1, 1,-1,-1,-1,-1,-1,-1, 1, 1],
                [ 1, 1,-1,-1,-1,-1,-1,-1, 1, 1],
                [-1, 1, 1, 1, 1, 1, 1, 1, 1,-1],
                [-1,-1, 1, 1, 1, 1, 1, 1,-1,-1]])

label = ['0','2','4','6','8']
ith = 4  # label[ith]
noise = 0.2
unknown = 0.1
#np.random.seed(150)
X_test = add_unknown(add_noise(X8_2,noise),unknown).reshape(-1,1)

# Storage Phase
# flatten the pattern
X0      = X0.reshape(1,X0.size)
X2      = X2.reshape(1,X2.size)
X4      = X4.reshape(1,X4.size)
X6      = X6.reshape(1,X6.size)
X8      = X8.reshape(1,X8.size)
X       = np.array([X0,X2,X4,X6,X8])
W_fixed = np.zeros([X0.shape[1],X0.shape[1]])

for p in range(X.shape[0]):      
    W_fixed = W_fixed + np.dot(X[p,0,:].reshape(-1,1),X[p,0,:].reshape(1,-1))  
for i in range(X.shape[2]):
    for j in range(X.shape[2]):
        if i==j:
            W_fixed[i,j]=0

# W is fixed now 
# Let's retrieve the pattern X with some noise

E = energy(X_test,W_fixed)
E_cycle = [E]
print('-------------------------------------')
print('E(0) : ',E)
print('X(0) : ',X_test[:,0])
X_new = np.zeros([10,10])
X_new = X_test
# obtain new X  
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

#-----------------  display result  ---------------
X_test = X_test.reshape(10,10)
X_new = X_new.reshape(10,10)
csv_E = []
with open('Retrieval_phase.csv', newline='') as csvfile:  
  rows = csv.DictReader(csvfile)  
  for row in rows:           
    #print(row['Pattern'], row['E'])    
    csv_E.append(float(row['E']))    

csv_E = np.array(csv_E)
result = label[np.argmin(abs(csv_E-E_cycle[-1]))]

fig = plt.figure(num='Retreive phase',figsize=(18,5))
ax = fig.add_subplot(141)
ax.set_title('Pattern {} with noise {}% and unknown {}%'.format(label[ith],noise*100,unknown*100)) #-- change label index
ax.imshow(bw(X_test),cmap = plt.cm.gray)

ax2 = fig.add_subplot(142)
ax2.set_title('Retrieval and Recognition result : {}'.format(result))
ax2.imshow(bw(X_new),cmap = plt.cm.gray)

ax3 = fig.add_subplot(122)
ax3.plot(range(len(E_cycle)),E_cycle)
ax3.set_title('E_final : {}'.format(E_cycle[-1]))
ax3.set_xlabel('cycle')
ax3.set_ylabel('Energy')
plt.show()








