import math
import random
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def sigmafunc(x):
    return 1 / (1 + math.exp(-x))

def customfunc(x):
    return x / (1 + abs(x))

def sigmagrad(x):
    s = sigmafunc(x)
    return s * (1 - s)

def customgrad(x):
    return 1 / ((1 + abs(x)) * (1 + abs(x)))

def func(acttype):
    data = load_iris()
    datax = data.data.tolist()
    datay = data.target.tolist()

    for i in range(len(datay)):
        if datay[i] == 0:
            datay[i] = [1,0,0]
        elif datay[i] == 1:
            datay[i] = [0,1,0]
        else:
            datay[i] = [0,0,1]

    w1 = [[random.uniform(-1,1) for j in range(4)] for i in range(6)]
    b1 = [0.0]*6
    w2 = [[random.uniform(-1,1) for j in range(6)] for i in range(3)]
    b2 = [0.0]*3

    lr = 0.01
    losslist = []
    acclist = []
    classok = [0,0,0]
    classcnt = [0,0,0]

    for e in range(200):
        loss = 0
        correct = 0
        for i in range(len(datax)):
            h = []
            z2 = []

            for j in range(6):
                v = datax[i][0]*w1[j][0] + datax[i][1]*w1[j][1] + datax[i][2]*w1[j][2] + datax[i][3]*w1[j][3] + b1[j]
                if acttype == 0:
                    h.append(sigmafunc(v))
                else:
                    h.append(customfunc(v))

            out = []
            for j in range(3):
                v = h[0]*w2[j][0] + h[1]*w2[j][1] + h[2]*w2[j][2] + h[3]*w2[j][3] + h[4]*w2[j][4] + h[5]*w2[j][5] + b2[j]
                z2.append(v)
                out.append(sigmafunc(v))

            pred = out.index(max(out))
            real = datay[i].index(1)

            classcnt[real] += 1
            if pred == real:
                correct += 1
                classok[real] += 1

            for j in range(3):
                err = datay[i][j] - out[j]
                loss += err*err
                d = err * sigmagrad(z2[j])
                for k in range(6):
                    w2[j][k] += lr * d * h[k]
                b2[j] += lr * d

        losslist.append(loss/len(datax))
        acclist.append(correct/len(datax))

    classacc = [classok[0]/classcnt[0], classok[1]/classcnt[1], classok[2]/classcnt[2]]
    return losslist, acclist, classacc

x = [i/10 for i in range(-50,51)]
y1 = [sigmafunc(i) for i in x]
y2 = [customfunc(i) for i in x]

plt.plot(x,y1)
plt.plot(x,y2)
plt.show()

losssig, accsig, classsig = func(0)
losscus, acccus, classcus = func(1)

plt.plot(losssig)
plt.plot(losscus)
plt.show()

plt.plot(accsig)
plt.plot(acccus)
plt.show()

plt.bar([0,1,2], classsig)
plt.bar([0,1,2], classcus)
plt.show()

plt.bar(["sigmoid","custom"], [accsig[-1], acccus[-1]])
plt.show()

