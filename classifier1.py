
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
get_ipython().magic('matplotlib notebook')
import load_raw as Raw
import seaborn as sns
from scipy.special import expit as logit

df_cleaned = pd.read_csv('/data/ortho/TestDataAllPatients.csv', sep= ';')
#df_cleaned = df_cleaned.rename(columns={0: "thorax_r_x", 1: "thorax_r_y", 2: "thorax_r_z"})
#df_cleaned = df_cleaned.rename(columns={3: "clavicula_r_x", 4: "clavicula_r_y", 5: "clavicula_r_z"})
#df_cleaned = df_cleaned.rename(columns={6: "scapula_r_x", 7: "scapula_r_y", 8: "scapula_r_z"})
#df_cleaned = df_cleaned.rename(columns={9: "humerus_r_x", 10: "humerus_r_y", 11: "humerus_r_z"})
#df_cleaned = df_cleaned.rename(columns={12: "ellebooghoek_r"})
#df_cleaned = df_cleaned.rename(columns={15: "thorax_l_x", 16: "thorax_l_y", 17: "thorax_l_z"})
#df_cleaned = df_cleaned.rename(columns={18: "clavicula_l_x", 19: "clavicula_l_y", 20: "clavicula_l_z"})
#df_cleaned = df_cleaned.rename(columns={21: "scapula_l_x", 22: "scapula_l_y", 23: "scapula_l_z"})
#df_cleaned = df_cleaned.rename(columns={24: "humerus_l_x", 25: "humerus_l_y", 26: "humerus_l_z"})
#df_cleaned = df_cleaned.rename(columns={27: "ellebooghoek_l"})

#df_cleaned = df_cleaned.values

# x is naar voren
# y is omhoog
# z is opzij
print(df_cleaned.columns.values)
print(df_cleaned['Oorsprong'])


# In[2]:

df_cleaned['bias'] = 1
df_cleaned['ja'] = ['Cat3' in vincent for vincent in df_cleaned['Oorsprong']]
X = np.matrix(df_cleaned[['bias','clavicula_r_x','clavicula_r_y']])
y = np.matrix(df_cleaned[['ja']])

print(df_cleaned[['Oorsprong','ja']])


def scatter():
    cat3 = df_cleaned.where(df_cleaned['ja'])
    geen_cat3 = df_cleaned.where(~df_cleaned['ja'])
    plt.plot(cat3['clavicula_r_x'], cat3['clavicula_r_y'], '.', color='blue', markersize=2)
    plt.plot(geen_cat3['clavicula_r_x'], geen_cat3['clavicula_r_y'], '.', color='red', markersize=2)
    plt.title('clavicula links/rechts')
    plt.ylabel('rechts')
    plt.xlabel('links');
    plt.xlim(-100,100)
    plt.ylim(-100,100)
scatter()


def logit(z):
    return 1.0 / (1.0 + np.exp(-z))

def h(X, theta):
    return logit(X * theta)

def predict(X, theta):
    return h(X, theta) >= 0.5

def fit_model(X, y, alpha=0.00001, iterations=50000):
    m = X.shape[1]            # het aantal coefficienten
    print(m)
    theta = np.zeros((m, 1))  # initialiseer theta
    for iter in range(iterations):
        theta -= (alpha / m) * X.T * ( h(X, theta) - y )
    return theta

#%%time
theta = fit_model(X, y)
print(theta)

def evaluate(theta, X, y):
    return sum( predict(X, theta) == y ) / len(X)

def plot_decision_boundary(theta):
    ax = plt.gca()
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 1),
                         np.arange(y_min, y_max, 1))

    X = np.matrix(np.vstack([np.ones(xx.shape[0] * xx.shape[1]), xx.ravel(), yy.ravel()])).T
    boundary = logit(X * theta)
    boundary = boundary.reshape(xx.shape)

    ax.contour(xx, yy,
           boundary,
           levels=[0.5])
    
scatter()
plot_decision_boundary(theta)




