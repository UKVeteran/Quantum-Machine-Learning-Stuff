#Generate The Histogram for 20,000 data points centered at ~150 and s.d. 50

%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

houseprice = np.random.normal(150.0, 50.0, 20000)

plt.hist(houseprice, 75)
plt.show()

#Calculate Standard Deviation

houseprice.std()

#Calculate Variance

houseprice.var()