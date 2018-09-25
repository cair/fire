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
        plt.show()

        file_dest = dir_path + "/output/%s/%s.%s"
        file_name = self.name.replace(" ", "_")

        plt.savefig(file_dest % ("figures", file_name, "png"))
        plt.savefig(file_dest % ("figures", file_name, "pdf"))
        plt.savefig(file_dest % ("figures", file_name, "eps"))
        json.dump({
            "name": self.name,
            "x": self.x,
            "y": self.losses
        }, open(file_dest % ("plot_data", file_name, "json"), "w+"))

class PlotPerformance:
    def __init__(self):
        self.i = 0
        self.fig = plt.figure()
        self.agents = {}
        self.name = "N/A"

        self.sx = None
        self.sy = None

    def new(self, name):
        self.name = name
        self.start()

    def start(self):
        self.agents[self.name] = [[], []]
        self.agent = self.agents[self.name]



    def log(self, x, y):
        self.agent[0].append(x)
        self.agent[1].append(y)
        self.i += 1

        if self.i % 5 != 1:
            return False
        #clear_output(wait=True)

        for agent_name, values in self.agents.items():
            plt.plot(values[0], values[1], label=agent_name)

        plt.legend()
        plt.show();
