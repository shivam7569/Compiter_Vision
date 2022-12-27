from RCNN.createFineTuneData import createFineTuneData
from RCNN.createClassifierData import createClassifierData
from RCNN.fineTune import loadBestFineTuneModel, performFineTuning

##### Preparing fine tuning data #####

# createFineTuneData()

##### ************************** #####

##### Fine Tuning #####

performFineTuning(epochs=500, debug=False)

##### *********** #####

##### Load Model #####

# model = loadBestFineTuneModel()

##### ********** #####

##### Prepare classifier data #####

# createClassifierData()

##### *********************** #####

# events.out.tfevents.1670179168.erd-server
