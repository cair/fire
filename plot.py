import keras
import matplotlib.pyplot as plt
import os
import json
dir_path = os.path.dirname(os.path.realpath(__file__))

class PlotLosses(keras.callbacks.Callback):
    def __init__(self, name):
        super().__init__()
        self.i = 0
        self.fig = plt.figure()
        self.x = []
        self.losses = []
        self.val_losses = []
        self.logs = []
        self.name = name
    #def on_train_begin(self, logs={}):

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        if self.i % 100 != 1:
            return False
        #clear_output(wait=True)

        plt.title(self.name)
        plt.plot(self.x, self.losses, label="loss")
        #plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        file_dest = dir_path + "/output/%s/%s.%s"
        file_name = self.name.replace(" ", "_")

        plt.savefig(file_dest % ("figures", file_name, "png"))
        plt.savefig(file_dest % ("figures", file_name, "pdf"))
        plt.savefig(file_dest % ("figures", file_name, "eps"))
        plt.show()
        json.dump({
            "name": self.name,
            "x": self.x,
            "y": self.losses
        }, open(file_dest % ("plot_data", file_name, "json"), "w+"))

class PlotPerformance:
    def __init__(self):
        self.i = 0
        self.fig = plt.figure()
        self.name = "N/A"

        self.x = []
        self.y = []

    def new(self, name):
        self.name = name
        self.start()
        self.x = []
        self.y = []
        plt.clf()
        plt.cla()

    def start(self):
        pass


    def log(self, x, y):
        print(x, y)
        self.x.append(int(x))
        self.y.append(int(y))
        self.i += 1

        #if self.i % 5 != 1:
        #    return False
        #clear_output(wait=True)

        plt.plot(self.x, self.y, label=self.name)

        plt.legend()

        file_dest = dir_path + "/output/%s/%s.%s"
        file_name = self.name.replace(" ", "_")

        plt.savefig(file_dest % ("figures", file_name, "png"))
        plt.savefig(file_dest % ("figures", file_name, "pdf"))
        plt.savefig(file_dest % ("figures", file_name, "eps"))
        plt.show()
        json.dump({
            "name": self.name,
            "x": self.x,
            "y": self.y,
        }, open(file_dest % ("plot_data", file_name, "json"), "w+"))

