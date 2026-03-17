import matplotlib.pyplot as plt

def training_curve(train_loss,val_loss):

    plt.plot(train_loss,label="train")

    plt.plot(val_loss,label="val")

    plt.legend()

    plt.savefig("results/figures/training_curve.png")