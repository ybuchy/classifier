from preproc import *
from classifier import *


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

    files = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte",
             "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]
    preproc = Preproc(*files)
    preproc.load()
    tr_set = preproc.get_training_set(batch_size)
    val_set = preproc.get_validation_set(batch_size)
    test_set = preproc.get_test_set(batch_size)

    print("finished")
    print("training model...")

    net = NN_classifier(learning_rate, batch_size, 1, il_size, ol_size, hl_size)

    retrain = True

    if retrain:
        costs = []
        for num, (ibatch, wbatch) in enumerate(test_set):
                costs.append(net.mini_batch_gd(ibatch, wbatch))
                net.gradient_check(ibatch[0], wbatch[0])
                exit()
            

        print("training done")

    #else:
        # weights_hid_out = np.load("who.npy")
        # weights_in_hid = np.load("wih.npy")

    plt.plot(costs)
    plt.show()

    print("testing model...")
    num_correct = 0
    for num, (image, label) in enumerate(zip(images, labels)):
        net.set_input_layer(image)
        net.forward()
        output = net.layers[-1].get_nodes()
        wanted = np.zeros(10)
        wanted[label] = 1
        highest = list(output).index(max(output))
        if highest == label:
            num_correct += 1
        if num == 3000:
            break

    print(num_correct / 3000)


if __name__ == "__main__":
    main()
