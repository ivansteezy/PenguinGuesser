from django.shortcuts import render
import logging

def MakePrediction(request):
    if request.method == "POST":
        print("El request contiene")
        print (request.POST.get('hiddenlayers1'))
        print (request.POST.get('hiddenlayers2'))
        print (request.POST.get('hiddenlayers3'))
        print (request.POST.get('numIterations'))
        print (request.POST.get('activationFunction'))
        print (request.POST.get('optimizationAlgorithm'))
        print (request.POST.get('trainingDataSize'))

        
    else:
        print("No es un post")

    return render(request, 'homepage/index.html')



def index(request):      
    print("Hola mundo")  
    return render(request, 'homepage/index.html')