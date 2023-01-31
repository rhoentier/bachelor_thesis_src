import warnings
from ai_framework.ts_ai import TSAI

warnings.simplefilter(action="ignore", category=UserWarning)


###
#
# Taken from:
# Johannes Alecke. Analyse und Optimierung von Angriffen auf tiefe neuronale Netze, Hochschule Bonn-Rhein-Sieg, 2020
#
###

class TrafficSignMain:

    def __init__(self, model: TSAI, epochs, image_size=299, num_classes=43) -> None:
        super().__init__()
        self.model = model
        self.epochs = epochs
        self.image_size = image_size
        self.input_shape = (3, image_size, image_size)
        self.num_classes = num_classes

    # Function for using the AI.
    def loading_ai(self):
        self.model.load()
