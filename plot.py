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
        plt.clf()
        plt.cla()
        plt.title(self.name)
        plt.plot(self.x, self.losses, label="loss")
        #plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        file_dest = dir_path + "/output/%s/%s.%s"
        file_name = self.name.replace(" ", "_")

        plt.savefig(file_dest % ("figures", file_name, "png"), dpi=1200)
        plt.savefig(file_dest % ("figures", file_name, "pdf"), dpi=1200)
        plt.savefig(file_dest % ("figures", file_name, "eps"), dpi=1200)
        #plt.show(block=False)
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

    def new(self, name, prefix):
        self.prefix = prefix
        self.name = name
        self.start()
        self.x = []
        self.y = []

    def start(self):
        pass

    def log(self, x, y):
        self.x.append(int(x))
        self.y.append(int(y))
        self.i += 1


        #self.to_file()
        #plt.show(block=False)



    def to_file(self):
        plt.clf()
        plt.cla()
        #plt.title("Performance of %s in %s node graph")
        plt.plot(self.x, self.y, label=self.name)
        plt.legend()
        file_dest = dir_path + "/output/%s/%s.%s"
        file_name = self.prefix + "_" + self.name.replace(" ", "_")

        plt.savefig(file_dest % ("figures", file_name, "png"), dpi=1200)
        plt.savefig(file_dest % ("figures", file_name, "pdf"), dpi=1200)
        plt.savefig(file_dest % ("figures", file_name, "eps"), dpi=1200)

        json.dump({
            "name": self.name,
            "x": self.x,
            "y": self.y,
        }, open(file_dest % ("plot_data", file_name, "json"), "w+"))

