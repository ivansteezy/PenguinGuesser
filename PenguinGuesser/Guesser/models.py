from django.db import models


class Classifier(models.Model):
    hiddenLayer1 = models.CharField(max_length=5)
    hiddenLayer2 = models.CharField(max_length=5)
    hiddenLayer3 = models.CharField(max_length=5)

    numberOfIterations = models.CharField(max_length=5)
    activationFunction = models.CharField(max_length=20)
    optimizationAlgorithm = models.CharField(max_length=20)

