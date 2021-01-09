from geetools import batch
import ee
import time

#autentica l'utente
ee.Initialize()

#imageCollection presa a caso, selezionate alcune bande
landsat = ee.ImageCollection("LANDSAT/LC08/C01/T1_SR").select(["B4","B3","B2"])

#regione casuale
Ituna_AOI = ee.Geometry.Rectangle([-51.84448, -3.92180, -52.23999, -4.38201])


landsat = landsat.filterDate("2019-07-01", "2019-08-01")
landsat_AOI = landsat.filterBounds(Ituna_AOI)

#controlla i file totali
print("Total number:", landsat_AOI.size().getInfo())

#crea una lista di task
tasks = batch.Export.imagecollection.toDrive(landsat_AOI, "cartella_python_1007", region = Ituna_AOI, scale = 100)

#prende l'ultima task
task = tasks[-1]
print(task)

#a regola dovrebbe far partire solo task, ma attiva anche le altre :\
task.start()
