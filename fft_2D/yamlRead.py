import yaml as yl
import globalData as glob

def parseYAML(dataDir):
    paraFile = dataDir + "input/parameters.yaml"
    yamlFile = open(paraFile, 'r')
    try:
        yamlData = yl.load(yamlFile, yl.FullLoader)
    except:
        yamlData = yl.load(yamlFile)

    glob.Lx = float(yamlData["Program"]["X Length"])
    glob.Lz = float(yamlData["Program"]["Z Length"])

    glob.Nx = int(yamlData["Mesh"]["X Size"])
    glob.Nz = int(yamlData["Mesh"]["Z Size"])

    gridType = str(yamlData["Mesh"]["Mesh Type"])

    if gridType[0] == 'U':
        glob.btX = 0.0
    else:
        glob.btX = float(yamlData["Mesh"]["X Beta"])

    if gridType[2] == 'U':
        glob.btZ = 0.0
    else:
        glob.btZ = float(yamlData["Mesh"]["Z Beta"])
