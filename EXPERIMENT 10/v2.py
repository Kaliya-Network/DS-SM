import numpy as np, matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
d=np.random.normal(70,10,30)
m=d.mean(); s=d.std(ddof=1); n=len(d)

print("Mean:",m,"\nStd:",s)

t=stats.t.ppf(0.975,n-1)*(s/np.sqrt(n))
print("\n95% t-CI:",(m-t,m+t))

z=stats.norm.ppf(0.975)*(s/np.sqrt(n))
print("95% z-CI:",(m-z,m+z))

p=40/100
se=np.sqrt(p*(1-p)/100)
print("95% Prop-CI:",(p-z*se,p+z*se))

boot=[np.mean(np.random.choice(d,n,1)) for _ in range(1000)]
lb,ub=np.percentile(boot,[2.5,97.5])
print("95% Bootstrap-CI:",(lb,ub))

plt.hist(boot,bins=25,edgecolor='black')
plt.axvline(lb,color='r'); plt.axvline(ub,color='g')
plt.show()
