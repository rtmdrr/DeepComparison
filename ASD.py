from scipy.stats import wasserstein_distance as WD
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

data1 = []
data2 = []
def read(file):
	data = []
	with open(file, 'r') as f:
		for row in f:
			num = row.rstrip('\n')
			if num != '':
				data.append(float(num))
	return data


def phi(x):
    # 'Cumulative distribution function for the standard normal distribution'
    return (1.0 + erf(x / sqrt(2.0))) / 2.


def frange(start, stop, step):
     x = start
     while x < stop:
         yield x
         x += step

F_k = []
G_k = []
n = 0
m = 0
def build(f, g):
	global F_k
	global G_k
	global n
	global m
	F_k = np.sort(f)
	n = len(F_k)
	G_k = np.sort(g)
	m = len(G_k)
	print("F_k ", np.around(F_k, decimals=4))
	print("G_k ", np.around(G_k, decimals=4))
	print("n ", n)
	print("m ", m)


def buildnew(f, g):
	global Fb
	global Gb
	Fb = np.sort(f)
	Gb = np.sort(g)


def invG(p):
	index = int(np.ceil(p*m))
	if index >= m:
		return G_k[m-1]
	elif index == 0:
		return G_k[0]
	return G_k[index-1]


def invF(p):
	index = int(np.ceil(p*n))
	if index >= n:
		return F_k[n-1]
	elif index == 0:
		return F_k[0]
	return F_k[index-1]


def invGnew(p):
	index = int(np.ceil(p*M))
	if index >= M:
		return Gb[M-1]
	elif index == 0:
		return Gb[0]
	return Gb[index-1]


def invFnew(p):
	index = int(np.ceil(p*N))
	if index >= N:
		return Fb[N-1]
	elif index == 0:
		return Fb[0]
	return Fb[index-1]


def eW():
	s = 0
	se = 0
	for p in frange(0, 1, dp):
		temp = invF(p)-invG(p)
		tempe = max(temp, 0)
		s = s+temp*temp*dp
		se = se+tempe*tempe*dp
	if s != 0:
		return [se/s]
	else:
		print("The denominator is 0")
		return 0


def eWnew():
	s = 0
	se = 0
	for p in frange(0, 1, dp):
		temp = invFnew(p)-invGnew(p)
		tempe = max(temp, 0)
		s = s+temp*temp*dp
		se = se+tempe*tempe*dp
	if s != 0:
		return [se/s]
	else:
		print("The denominator is 0")
		return 0


######################################################################
data1 = read('./comparisons/dropout NER/NER_IOBES_variational_dropout.txt')
data2 = read('./comparisons/dropout NER/NER_IOBES_no_dropout.txt')

build(data1, data2)
e = 0.01
alpha = 0.05
dp = 0.005

N = 1000
M = 1000
B = 1000

lamda = (0.0+N)/(N+M)
const = np.sqrt((1.0*N*M)/(N+M+0.0))
phi_1alpha = scipy.stats.norm.ppf(alpha)
theta = []
for b in range(0, B):
	Fb = []
	Gb = []
	Fvalues = []
	Gvalues = []
	uniF = np.random.uniform(0, 1, N)
	uniG = np.random.uniform(0, 1, M)
	for i in range(0, N):
		Fvalues.append(invF(uniF[i]))
	for j in range(0, M):
		Gvalues.append(invG(uniG[j]))
	buildnew(Fvalues, Gvalues)
	distance = eWnew()
	theta.append(distance)

sigma = np.sqrt(np.var(theta))
eW = eW()

print("sigma = ", sigma)
print("eW = ", eW)
print("e = ", eW-(1/const)*sigma*phi_1alpha)
print("AVG ", np.average(data1), np.average(data2))
print("SD ", np.sqrt(np.var(data1)), np.sqrt(np.var(data2)))
print("MEDIAN ", np.median(data1), np.median(data2))

x = []
y1 = []
y2 = []
for a in frange(0, 1, 0.01):
	x.append(a)
	y1.append(invF(a))
	y2.append(invG(a))
	# print F(a)-G(a)


# plt.hist(F_k)
# plt.hist(G_k)

plt.plot(x, y1, 'g', label='data1')
plt.plot(x, y2, 'r--', label='data2')
plt.legend()
plt.show()






