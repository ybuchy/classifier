from cl_module.preproc import *
from cl_module.classifier import *


def main():
    # input layer size: 28^2 (amount of pixels)
    # hidden layer size: try different ones to prevent overfitting (why?)
    # first try: 500 (so about 64%)
    # (try out Grid search(hyperparameter optimization) later
    # output layer size: 10 (classification to number)
    batch_size = 500
    learning_rate = .0001
    il_size = 28 ** 2
    hl_size = 300
    ol_size = 10

    print("loading mnist data...")

    files = ["data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte",
             "data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte"]
    preproc = Preproc(*files)
    preproc.load()
    tr_set = preproc.get_training_set(batch_size)
    val_set = preproc.get_validation_set(batch_size)
    test_set = preproc.get_test_set(batch_size)

    print("finished")
    print("training model...")

    net = NN_classifier(learning_rate, batch_size, 1, il_size, ol_size, hl_size)

    costs = []
    for num, (ibatch, wbatch) in enumerate(test_set):
        costs.append(net.mini_batch_gd(ibatch, wbatch))
        if num == 5:
            break

    print("training done")

    plt.plot(costs)
    plt.show()


if __name__ == "__main__":
    main()
