import os
import sys

import cv2
import imageUtils as iu
import cellOperations as co


class tarea3:

    def main(raw, ref):
        # creamos los directorios donde guardaremos los outputs si es que no existen
        if not os.path.exists('output/'):
            os.makedirs('output/')

        calculado = iu.readTif(raw)
        original = iu.readTif(ref)
        calcDict = {'nro': [], 'center': [], 'area': [], 'elong': [], 'orient': []}
        origDict = {'nro': [], 'center': [], 'area': [], 'elong': [], 'orient': []}
        for image in range(len(calculado)):
            currentOriginal = original[image]
            currentCalculado = calculado[image]
            # calculamos lo necesario para las imagenes raw
            calculadoMoments, calculadoMomentAreas, calculadoContours = iu.getSegmentation(currentCalculado)
            # pintamos la imagen en relacion a las areas de las celulas y guardamos la imagen en una carpeta output
            paintCalculado = iu.paintCells(currentCalculado, calculadoMomentAreas, calculadoContours)
            cv2.imwrite("output/output" + str(image) + ".jpg", paintCalculado)
            calculadoRealMoments, calculadoNroCelulas = co.getRealCells(calculadoMoments)
            # agregamos el numero de celulas y arreglos vacios en donde guardaremos los datos de las celulas
            calcDict['nro'].append(calculadoNroCelulas)
            calcDict['center'].append([])
            calcDict['area'].append([])
            calcDict['elong'].append([])
            calcDict['orient'].append([])
            for moment in range(len(calculadoRealMoments)):
                cell = calculadoRealMoments[moment]
                center = co.getCenterOfMass(cell)
                calcDict['center'][image].append(center)
                area = co.getAreaOfCell(cell)
                calcDict['area'][image].append(area)
                elong = co.getElongation(cell)
                calcDict['elong'][image].append(elong)
                orient = co.getOrientation(cell)
                calcDict['orient'][image].append(orient)

            originalMoments = iu.getOriginalMoments(currentOriginal)
            originalRealMoments, originalNroCelulas = co.getRealCells(originalMoments, area=20.0)
            origDict['nro'].append(originalNroCelulas)
            origDict['center'].append([])
            origDict['area'].append([])
            origDict['elong'].append([])
            origDict['orient'].append([])
            for moment in range(len(originalRealMoments)):
                cell = originalRealMoments[moment]
                center = co.getCenterOfMass(cell)
                origDict['center'][image].append(center)
                area = co.getAreaOfCell(cell)
                origDict['area'][image].append(area)
                elong = co.getElongation(cell)
                origDict['elong'][image].append(elong)
                orient = co.getOrientation(cell)
                origDict['orient'][image].append(orient)

    if __name__ == "__main__":
        main(sys.argv[1], sys.argv[2])
