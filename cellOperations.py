import numpy as np


# receives a list of moments of contours to determine its valid cells (area > 100)
# returns a list without the invalid cells and the number of valid cells
def getRealCells(moments, area=100.0):
    aux = []
    for i in range(len(moments)):
        if moments[i]['m00'] > area:
            aux.append(moments[i])
    return aux, len(aux)


# receives a dictionary with the moments of a cell and returns the center
# of mass of the cell as a tuple
def getCenterOfMass(moments):
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    return [cx, cy]


# receives a dictionary with the moments of a cell and returns the area of the cell
def getAreaOfCell(moments):
    area = moments['m00']
    return int(area)


# receives a dictionary with the moments of a cell and returns the elongation of a cell
def getElongation(moments):
    matrix = np.matrix([[moments['mu20'] / moments['m00'], moments['mu11'] / moments['m00']],
                        [moments['mu11'] / moments['m00'], moments['mu02'] / moments['m00']]])
    values = np.linalg.eigvals(matrix)
    elongation = 1.0 - (values.min() / values.max())
    return int(elongation)


# receives a dictionary with the moments of a cell and returns the orientation
# of the bigger axis of a cell with respect to the vertical axis
def getOrientation(moments):
    div = (moments['mu20'] - moments['mu02'])
    if div == 0:
        orientation = 0.0
    else:
        alpha = 0.5 * np.arctan(2 * moments['mu11'] / (moments['mu20'] - moments['mu02']))
        orientation = 90.0 - alpha
    return int(orientation)
