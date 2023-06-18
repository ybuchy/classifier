from cl_module.preproc import *
from cl_module.classifier import *


def main():
    # input layer size: 28^2 (amount of pixels)
    # hidden layer size: try different ones to prevent overfitting (why?)
    # first try: 500 (so about 64%)
    # (try out Grid search(hyperparameter optimization) later
    # output layer size: 10 (classification to number)
    batch_size = 768
    learning_rate = 1e-3
    il_size = 28 ** 2
    hl_size = 300
    ol_size = 10

    print("loading mnist data...")

    files = ["data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte",
             "data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte"]
    preproc = Preproc(*files)
    preproc.load()
    tr_set = preproc.get_training_set(batch_size)
    val_set = preproc.get_validation_set()
    test_set = preproc.get_test_set()

    print("finished")
    print("training model...")

    net = NN_classifier(learning_rate, batch_size, 1, il_size, ol_size, hl_size)

    costs = []
    val_loss = []
    epoch = 0
    while True:
        if epoch > 1 and val_loss[-1] - val_loss[-2] < 1e-4:
            break
        for num, (ibatch, wbatch) in enumerate(tr_set):
            costs.append(net.mini_batch_gd(ibatch, wbatch))
        val_loss.append(net.check_acc(val_set))
        print(val_loss[-1])
        epoch += 1

    print("training done")

    plt.plot(costs)
    plt.show()

    print(net.check_acc(test_set))


if __name__ == "__main__":
    main()
