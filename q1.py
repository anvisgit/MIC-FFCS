import math
import matplotlib.pyplot as plt

def sigmafunc(x):
    return 1 / (1 + math.exp(-x))

def func():
    datax = [[0,0],[0,1],[1,0],[1,1]]
    datay = [0,1,1,1]

    w1 = [[0.5, -0.5],[0.3, 0.8]]
    b1 = [0.0, 0.0]

    w2 = [0.7, -0.6]
    b2 = 0.0

    lr = 0.5
    losslist = []

    for e in range(200):
        totloss = 0
        for i in range(4):
            h1 = sigmafunc(datax[i][0]*w1[0][0] + datax[i][1]*w1[0][1] + b1[0])
            h2 = sigmafunc(datax[i][0]*w1[1][0] + datax[i][1]*w1[1][1] + b1[1])

            out = sigmafunc(h1*w2[0] + h2*w2[1] + b2)

            err = datay[i] - out
            totloss += err * err

            dout = err * out * (1 - out)

            dw20 = dout * h1
            dw21 = dout * h2
            db2 = dout

            dh1 = dout * w2[0] * h1 * (1 - h1)
            dh2 = dout * w2[1] * h2 * (1 - h2)

            w2[0] += lr * dw20
            w2[1] += lr * dw21
            b2 += lr * db2

            w1[0][0] += lr * dh1 * datax[i][0]
            w1[0][1] += lr * dh1 * datax[i][1]
            w1[1][0] += lr * dh2 * datax[i][0]
            w1[1][1] += lr * dh2 * datax[i][1]

            b1[0] += lr * dh1
            b1[1] += lr * dh2

        losslist.append(totloss / 4)

    plt.plot(losslist)
    plt.show()

func()

