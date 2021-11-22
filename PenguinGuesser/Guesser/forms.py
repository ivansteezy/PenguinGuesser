from django import forms
from .models import Classifier

class ClassifierForm(forms.ModelForm):
    class Meta:
        model = Classifier
        fields = ['hiddenLayer1', 'hiddenLayer3', 'hiddenLayer3', 'numberOfIterations', 'activationFunction', 'optimizationAlgorithm']