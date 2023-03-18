# Welcome! 

This is my quickly thrown together site for CSE455. I really didn't have a whole of time to work on this, but I hope it gives you a little insight on my project. For a overview, please watch the video below! Additionally, for a more in-depth dive, read my written summary.

**Video** \\
[![Video Link](https://img.youtube.com/vi/bh5WCtkUrRk/0.jpg)](https://youtu.be/bh5WCtkUrRk)

**Summary** \\
My project was to see if I could train a convolutional neural network that classifies a boardgame as either Ticket to Ride or Settlers of Catan. Additionally, I wanted to train this classifier entirely on screen shots from a virtual session of those board games; my interest was - does this generation of data in the virtual world have an applicability in classifying the analogous situations in the real one? As such, I recorded myself playing one game of Catan and one game of Ticket to Ride, both lasting more or less 70 minutes. I used ffmpeg to delete duplicate frames, sample the videos, and resize them down to around 120 324x576 pixel frames.

From then, I loaded the data into a convolutional neural network, that used ReLu, cross entropy loss, and stochastic gradient descent. Most of my initial choices were based of what seemed popular among begineer networks. However, I did try and switching all of these parameters with known alternatives when I first ran across problems, which was essentially at the first run; my model had a validation accuracy of about 20%, and converged at a training accuracy of 50%. I really struggled with this. Over this last half week, I think I probably manually changed every parameter five or six times, and I still couldn't really beat more than 60% on validation data!

Fundamentally, I think this project was flawed in two ways. First, as to my troubles specifically, 120 datapoints is not enough to train a classifer of images; I manually grid searched learning rates, network depth, and batch size to no avail. Convergence is always reached, and always at a disappointingly low level of accuracy. Sampling at a greater rate to create more data also doesn't help, as essentially, we are making twenty nearly identitcal data points from one. Second, there's a disconnect between images of tabletop simulator and images of people playing boardgames on a table top: All the frames of my virtual session essentially looked the same, except for a few pieces and rotation. The background, shadows, obstructions that make computer vision hard aren't present, and thus accounted for. I think a good analogy is: trying to train a model that understand human conversation in a busy cafe between sarcastic people, using their discord messages as the training set.

If I could go back and redo this project, I would want to try using trying to match points of interest between a picture of the base boardgame and real images; essentially using the code we wrote for panoramas, to classify how mappable each board game is to a certain picture. I think it would be more conceptually pretty and easier to trouble shoot; additionally, I think it even has more potential to be accurate. But to be completely honest, I was too exhausted from dead week and finals to restart.

**Graphs** \\
The Graphs I used in the presentation:

![129SGD43.png](graphs/129SGD43.png) \\
*First attempts at training the model*

![129SGD32.png](graphs/129SGD32.png) \\
*Turning down learning rate and model complexity in hopes of converging slower*

![1294SGD43.png](graphs/1294SGD43.png) \\
*Resulting unstable model after sampling the data every 2 seconds*

![1294SGD54.png](graphs/1294SGD54.png) \\
*Using the same sample rate, but tuning the model to reduce variance*

[**Code**](main.py)

    import torch
    from torch import nn
    from torch.optim import SGD
    from torch.utils.data import DataLoader, TensorDataset, random_split
    import torchvision
    from torchvision import transforms

    from typing import Tuple, Union, List, Callable
    from PIL import Image

    import matplotlib.pyplot as plt
    from tqdm import tqdm, trange

    assert torch.cuda.is_available(), "GPU is not available, check the directions above (or disable this assertion to use CPU)"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE: " + DEVICE)

    batch_size = 10
    lr = 0.778

    def conv_model(n: int, k: int, s: int) -> nn.Module:
      """
      Instantiate my convultionary model and send it to device.

      Args:
        n: The output channel for convolution filters.
        k: Kernel size for convolution filters.
        s: Smoothing size for maxpool.

      Returns:
        Neural Network for images.
      """
      num_layers = 3
      h, w = 324, 576
      for i in range(0, num_layers):
        h = (h - k + 1) // s
        w = (w - k + 1) // s

      model =  nn.Sequential(
                nn.Conv2d(3, n, k),
                nn.ReLU(),
                nn.MaxPool2d(s, s),
                nn.Conv2d(n, n, k),
                nn.MaxPool2d(s, s),
                nn.Conv2d(n, 1, k),
                nn.MaxPool2d(s, s),
                nn.Flatten(),
                nn.Linear(1 * h * w, 10),
                nn.Linear(10, 10),
                nn.Linear(10, 20),
                nn.Linear(20, 10),
                nn.Linear(10, 2)
             )
      return model.to(DEVICE)

    def train(
        model: nn.Module, optimizer: SGD,
        train_loader: DataLoader, val_loader: DataLoader,
        epochs: int = 20
        )-> Tuple[List[float], List[float], List[float], List[float]]:
      """
      Trains a model for the specified number of epochs using the loaders.

      Returns:
        Lists of training loss, training accuracy, validation loss, validation accuracy for each epoch.
      """

      loss = nn.CrossEntropyLoss()

      train_losses = []
      train_accuracies = []
      val_losses = []
      val_accuracies = []
      i = 1

      for e in range(epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        # Main training loop; iterate over train_loader. The loop
        # terminates when the train loader finishes iterating, which is one epoch.
        for (x_batch, labels) in train_loader:
          # One Batch of Training
          x_batch, labels = x_batch.to(DEVICE), labels.to(DEVICE)

          optimizer.zero_grad()
          labels_pred = model(x_batch)
          batch_loss = loss(labels_pred, labels)
          train_loss += batch_loss.item()

          labels_pred_max = torch.argmax(labels_pred, 1)
          batch_acc = torch.sum(labels_pred_max == labels)
          train_acc += batch_acc.item()


          batch_loss.backward()
          optimizer.step()

        print(str(i) + ": " + str(train_acc / (batch_size * len(train_loader))))
        i += 1

        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_acc / (batch_size * len(train_loader)))

        # Validation loop; use .no_grad() context manager to save memory.
        model.eval()
        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():
          for (v_batch, labels) in val_loader:
            v_batch, labels = v_batch.to(DEVICE), labels.to(DEVICE)
            labels_pred = model(v_batch)
            v_batch_loss = loss(labels_pred, labels)
            val_loss += v_batch_loss.item()

            v_pred_max = torch.argmax(labels_pred, 1)
            batch_acc = torch.sum(v_pred_max == labels)
            val_acc += batch_acc.item()

          val_losses.append(val_loss / len(val_loader))
          val_accuracies.append(val_acc / (batch_size * len(val_loader)))

      return train_losses, train_accuracies, val_losses, val_accuracies


    def evaluate(
        model: nn.Module, loader: DataLoader
    ) -> Tuple[float, float]:
      """Computes test loss and accuracy of model on loader."""
      loss = nn.CrossEntropyLoss()
      model.eval()
      test_loss = 0.0
      test_acc = 0.0

      with torch.no_grad():
        for (batch, labels) in loader:
          batch, labels = batch.to(DEVICE), labels.to(DEVICE)
          y_batch_pred = model(batch)
          batch_loss = loss(y_batch_pred, labels)
          test_loss = test_loss + batch_loss.item()

          pred_max = torch.argmax(y_batch_pred, 1)
          batch_acc = torch.sum(pred_max == labels)
          test_acc = test_acc + batch_acc.item()

        test_loss = test_loss / len(loader)
        test_acc = test_acc / (batch_size * len(loader))
        return test_loss, test_acc


    def load_images():
      """Loads the datasets from their respective folders"""
      train_dataset = torchvision.datasets.ImageFolder(root="Train/", transform=transforms.ToTensor())
      test_dataset = torchvision.datasets.ImageFolder(root="Test/", transform=transforms.ToTensor())

      return train_dataset, test_dataset


    def main():
        # Load in Boardgame Images
        train_dataset, test_dataset = load_images()

        # Split Train into Training Data and Validation
        train_dataset, val_dataset = random_split(train_dataset, [int(0.9 * len(train_dataset)), len(train_dataset) - int(0.9 * len(train_dataset))])

        # Create Loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        # Init Convolutionary Model and Optimizer
        model = conv_model(4, 3, 3)
        optimizer = SGD(model.parameters(), lr)

        # Train Model
        train_loss, train_acc, val_loss, val_acc = train(
            model,
            optimizer,
            train_loader,
            val_loader,
            epochs=30
        )

        # Make 'Plots'
        plt.plot(train_acc, label="train")
        plt.plot(val_acc, label="test")

        plt.xlabel("Epoch Number")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

        # Evaluate the Model on Real Photos
        test_loss, test_acc = evaluate(model, test_loader)
        print(f"Test Accuracy: {test_acc}")

    if __name__ == "__main__":
        main()

[**Final Training Data**](https://drive.google.com/drive/folders/1ndq3LGVre4r7xSa80bFePWvEAPejoSE_?usp=share_link)

**Thank You** \\
Thank you for taking the time to review this! Please email me at jzhang66@uw.edu if you have any questions or things needed for grading.
