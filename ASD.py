import sys
from scipy.stats import norm as normal
from scipy.stats import mannwhitneyu as Utest
import numpy as np
# import matplotlib.pyplot as plt


F = []
G = []
n = 0
m = 0
def buildOrigCDFs(f, g):
    global F
    global G
    global n
    global m
    F = np.sort(f)
    n = len(F)
    G = np.sort(g)
    m = len(G)


def buildNewCDFs(f, g):
    global Fb
    global Gb
    Fb = np.sort(f)
    Gb = np.sort(g)


def invG(p):
    index = int(np.ceil(p*m))
    if index >= m:
        return G[m-1]
    elif index == 0:
        return G[0]
    return G[index-1]


def invF(p):
    index = int(np.ceil(p*n))
    if index >= n:
        return F[n-1]
    elif index == 0:
        return F[0]
    return F[index-1]


def invGnew(p, M):
    index = int(np.ceil(p*M))
    if index >= M:
        return Gb[M-1]
    elif index == 0:
        return Gb[0]
    return Gb[index-1]


def invFnew(p, N):
    index = int(np.ceil(p*N))
    if index >= N:
        return Fb[N-1]
    elif index == 0:
        return Fb[0]
    return Fb[index-1]


def epsilon(dp):
    s = 0.0
    se = 0.0
    for p in np.arange(0, 1, dp):
        temp = invG(p)-invF(p)
        tempe = max(temp, 0)
        s = s+temp*temp*dp
        se = se+tempe*tempe*dp
    if s != 0:
        return se/s
    else:
        print("The denominator is 0")
        return 0.0


def epsilonNew(dp, N, M):
    denom = 0.0
    numer = 0.0
    for p in np.arange(0, 1, dp):
        diff = invGnew(p, M) - invFnew(p, N)  # check when F-1(t)<G-1(t)
        posdiff = max(diff, 0)
        denom += diff * diff * dp
        numer += posdiff * posdiff * dp
    if denom != 0.0:
        return numer/denom
    else:
        print("The denominator is 0")
        return 0.0


#############


def COS(data_A, data_B):
    # collection of statistics table
    print("AVG ", np.average(data_A), np.average(data_B))
    print("STD ", np.std(data_A), np.std(data_B))
    print("MEDIAN ", np.median(data_A), np.median(data_B))
    print("MIN ", np.min(data_A), np.min(data_B))
    print("MAX ", np.max(data_A), np.max(data_B))


def MannWhitney(data_A, data_B):
    # Mann-Whitney U test for stochastic dominance
    # Use only when the number of observation in each sample is > 20
    if n<20 or m<20:
        print("Use only when the number of observation in each sample is > 20")
        return 1.0
    _, pval = Utest(data_A, data_B, alternative='less')
    return pval


##############################################################
def main():
    if len(sys.argv) < 3:
        print("Not enough arguments\n")
        sys.exit()

    filename_A = sys.argv[1]    # scores from algorithm A
    filename_B = sys.argv[2]    # scores from algorithm B
    alpha = float(sys.argv[3])        # significance level of statistical test

    with open(filename_A) as f:
        data_A = f.read().splitlines()

    with open(filename_B) as f:
        data_B = f.read().splitlines()

    data_A = list(map(float, data_A))
    data_B = list(map(float, data_B))

    buildOrigCDFs(data_A, data_B)

    # constants
    dp = 0.005  # differential of the variable p - for integral calculations
    N = 1000    # num of samples from F for sigma estimate
    M = 1000    # num of samples from G for sigma estimate
    B = 1000    # bootstrap iterations for sigma estimate

    # calculate the epsilon quotient
    eps_FnGm = epsilon(dp)

    # estimate the variance
    lamda = (0.0 + N) / (N + M)
    const = np.sqrt((1.0 * N * M) / (N + M + 0.0))
    samples = []
    for b in range(B):
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
        buildNewCDFs(Fvalues, Gvalues)
        distance = epsilonNew(dp, N, M)
        samples.append(distance)

    sigma = np.std(samples)

    min_epsilon = min(max(eps_FnGm - (1/const) * sigma * normal.ppf(alpha), 0.0), 1.0)
    print("The minimal epsilon for which Algorithm A is almost "
          "stochastically greater than algorithm B is ", min_epsilon)
    if min_epsilon <= 0.5 and min_epsilon > 0.0:
        print("since epsilon <= 0.5 we will claim that A is "
              "better than B with significance level alpha=", alpha)
    elif min_epsilon == 0.0:
        print('since epsilon = 0, algorithm A is stochatically dominant over B')

    else:
        print("since epsilon > 0.5 we will claim that A "
              "is not better than B with significance level alpha=", alpha)

#     print(MannWhitney(data_A, data_B)<alpha)
#     COS(data_A, data_B)


if __name__ == "__main__":
    main()
