import numpy as np 
import matplotlib.pyplot as plt
import csv

def bw(X):
    '''
    input X with shape (10,10)
    '''
    for i in range(10):
        for j in range(10):
            if X[i,j] == -1:
                X[i,j] = 0
            else :
                X[i,j] = -1
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

#---------------------------------------------
#-------------  Traning Pattern  -------------
#---------------------------------------------

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


label = ['0','2','4','6','8']
ith = 4   #--  label[ith]
noise = 0.15
np.random.seed(1243)
X_test = add_noise(X8,noise).reshape(-1,1)
#------------------------------------------------
#-------------  Storage Phase -------------------
#------------------------------------------------

#------------  flatten the pattern  -------------
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
#------------- Wp is fixed now -----------

#-----------  Retrival Phase  ------------ 

#------------ input testing pattern X ----
#X_test = X8.reshape(-1,1)                    #-- change label index
E = energy(X_test,W_fixed)
#------------  obtain new X  -------------
E_cycle = [E]
print('-------------------------------------')
print('E(0) : ',E)
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
label = ['0','2','4','6','8']
result = label[np.argmin(abs(csv_E-E_cycle[-1]))]


plt.figure(num='Retreive phase',figsize=(18,5))
plt.subplot(141)
plt.title('Pattern {} with noise {}%'.format(label[ith],noise*100)) #-- change label index
plt.imshow(bw(X_test),cmap = plt.cm.gray)

plt.subplot(142)
plt.title('Result : {}'.format(result))
plt.imshow(bw(X_new),cmap = plt.cm.gray)

plt.subplot(122)
plt.plot(range(len(E_cycle)),E_cycle)
plt.title('Energy : {}'.format(E_cycle[-1])) #-- change label index
plt.xlabel('cycle')
plt.ylabel('Energy')
plt.show()

'''
#--------------------------------------------------------------------
table = [
    #['Pattern' , 'E'],
    [ label[ith] , E_cycle[-1]] #-- change label index
]
with open('Retrieval_phase.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(table)'''







