from django.shortcuts import render
from . import PenguinClassifier
import pandas as pd
import logging
import json

def MakePrediction(request):
    if request.method == "POST":
        hiddenlayers1 = int(request.POST.get('hiddenlayers1'))
        hiddenlayers2 = int(request.POST.get('hiddenlayers2'))
        hiddenlayers3 = int(request.POST.get('hiddenlayers3'))
        numIterations = int(request.POST.get('numIterations'))
        activationFunction = request.POST.get('activationFunction')
        optimizationAlgorithm = request.POST.get('optimizationAlgorithm')
        trainingDataSize = float(request.POST.get('trainingDataSize'))

        print("El request contiene")
        print (hiddenlayers1)
        print (hiddenlayers2)
        print (hiddenlayers3)
        print (numIterations)
        print (activationFunction)
        print (optimizationAlgorithm)
        print (trainingDataSize)

        pc = PenguinClassifier.PenginClassifier((hiddenlayers1, hiddenlayers2, hiddenlayers3), numIterations, activationFunction, optimizationAlgorithm, trainingDataSize)
        pc.TrainNeuralNetwork()

        trainerData = pc.GetTrainerData()
        pc.PredictData()

        ala, pico = trainerData[0].T
        trainerSpecies = trainerData[1].tolist()

        trainerDf = pd.DataFrame(list(zip(ala, pico, trainerSpecies)), columns=['Longitud del pico (mm)', 'Longitud de la aleta (mm)', 'Especie'])
        print("Datos de entrenmiento")
        print(trainerDf)

        expectedTrainer = pc.GetExpectedData()[0].tolist()
        alaRes,PicoRes = pc.GetExpectedData()[1].T
        speciesResult = pc.GetPredictionResults().tolist()
        trainerRes = pd.DataFrame(list(zip(alaRes, PicoRes, expectedTrainer, speciesResult)), columns=['billLength', 'flipperLength', 'expectedRes', 'obtainedRes'])
        print("Datos de resultados")
        print(trainerRes)

        accuracy = pc.GetAccuracyPercentage()
        print("El accuracy fue de: {:.4f}%".format(accuracy))
        
    else:
        print("No es un post")

    jsonRecord = trainerRes.reset_index().to_json(orient='records');
    data = [];
    data = json.loads(jsonRecord);
    context = {'d': data}
    return render(request, 'homepage/index.html', context)



def index(request):      
    print("Hola mundo")  
    return render(request, 'homepage/index.html')