import matplotlib.pyplot as plt


plt.ion()
def lossplot(x,y):
    plt.plot(x, y, linestyle='-', color='b')
    plt.title('Average loss per epoch')
    plt.xlabel('epoch')
    plt.ylabel('avg loss')
    plt.show()
    plt.pause(0.0001)

def dscm(x, y):
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(x.shape[0]):
        if x[i] == 1:
            if x[i] == y[i]:
                tp += 1
            else:
                fp += 1
        else:
            if x[i] == y[i]:
                tn += 1
            else:
                fn += 1

    # acc = (tp+tn)/(tp + fp + tn + fn)
    # sen = tp/(tp+fn)
    # spe = tn/(tn+fp)
    if (tp + fp + tn + fn) != 0:
        print(f"acc = {(tp + tn) / (tp + fp + tn + fn)}")
    else:
        print("ERROR - acc : (tp + fp + tn + fn) = 0")
    if (tp + fn) != 0:
        print(f"sen = {tp / (tp + fn)}")
    else:
        print("ERROR - sen : (tp+fn) = 0")
    if (tn + fp) != 0:
        print(f"spe = {tn / (tn + fp)}")
    else:
        print("ERROR - spe : (tn+fp) = 0")

    return tp, fp, tn, fn
    # return acc, sen, spe