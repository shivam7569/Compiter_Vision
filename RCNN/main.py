from RCNN.createFineTuneData import createFineTuneData
from RCNN.fineTune import loadBestFineTuneModel, performFineTuning

##### Preparing fine tuning data #####

# createFineTuneData()

##### ************************** #####

##### Fine Tuning #####

performFineTuning(epochs=250, debug=False)

##### *********** #####

##### Load Model #####

# model = loadBestFineTuneModel()

##### ********** #####
