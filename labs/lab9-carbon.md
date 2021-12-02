Lab 9: Carbon footprint
===
The goal of this lab is for you estimate the carbon footprint of your class project.

Group name:
---
Group members present in lab today:

1: Inference
----
1. Plug your device in to the Kill-a-watt and run inference using your model to get a measurement of the energy draw. What is its baseline energy draw, and how does that compare to running inference?
2. Multiply energy draw by inference time to get an estimate of energy required per inference (you can average over input size).
3. Multiply this by the carbon intensity of the location of your device. You can use [this resource](https://www.epa.gov/egrid/power-profiler#/).
4. Please include at least this estimate in your final project report.

2: Training
----
1. Did your project require training a model? If so, compute that estimate as well. If you used cloud resources, you can use [this tool](https://mlco2.github.io/impact/#compute) to help estimate. Otherwise, try to use the methods discussed last class for estmating carbon footprint due to training. Show your work and explain.

3: Extra
----
1. Everything else: Do you have logs you can use to estimate the amount of energy + carbon that went in to all project development? Other ways of estimating? Was your device plugged in and running for the entire semester?
2. Supply chain / lifecycle analysis: Can you estimate the additional footprint due to manufacturing hardware? Lifetime energy use of your project if it was deployed in the real world?
3. If you have a Pi or Jetson 4GB: Compare Kill-a-watt measurements to command line energy measurements, and discuuss.
