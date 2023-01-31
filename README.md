# HIL-Framework

### Beschreibung

Ein Framework um ein KI-basiertes Fahrassistenz-System systematisch zu analysieren.
Das Framework nimmt Bilddateien von Verkehrsschildern und bettet diese in
eine Fotografie einer Straße ein. Danach wird auf die Erkennung der Klassifikationssysteme gewartet und das Ergebnis
gespeichert.

Zusätzlich kann für jedes Schild ein adversarialer Angriff berechnet werden, um die Robustheit der
Klassifikationssysteme gegenüber Angriffen zu testen.

----

### Getting Started

#### Virtuelle Umgebung anlegen:

```conda create --name hil_framework```

```conda activate hil_framework```

#### Abhängigkeiten installieren:

Cuda Toolkit (https://developer.nvidia.com/cuda-toolkit) installieren und passende PyTorch-Version
installieren (https://pytorch.org/get-started/locally/).

Kvaser Treiber downloaden und installieren (https://www.kvaser.com/download/).

Zusätzliche Module installieren:
```pip install -r requirements.txt```

#### HIL-Framework ausführen:

1. Computer mit Klassifikationssytemen verbinden
2. ```run_hil.py``` ausführen