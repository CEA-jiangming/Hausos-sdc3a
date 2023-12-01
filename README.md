[![DOI](https://zenodo.org/badge/725544679.svg)](https://zenodo.org/doi/10.5281/zenodo.10229979)
![Reproducibility badge](https://uc9e62fbdf8f7783342b43806158.previews.dropboxusercontent.com/p/thumb/ACFT81THAHQgC3tmtABXCEcjCq-zQXPIetE0LWVjHhJxHJhGFLwr4DKHhF5TTG7pNXAbQTnKv42bWuZsmcnwzC147D6Ezzt0ixyZ55b39_8ifDcfgwtucK6Amws1q94uwDf-uTSA_p3WQftzDm6H43M85246IyH6ezHiI7rmRq6wmKI1_UPw6mtRnRRkIR2Zi6YPokxEmuoTkMx06mpPbZ8CIKoVILsHTtIz8br67iTA58HRR8DD5FfLj_ZQC36NI_wwWL-FKcZZxN88YErKmmqvKdGfkuVUS512018brpnhAF4wjm40gR0u8vUJbwYU1dt-hgnKozhzsMFOtg3O47e73BO2Bk_PCiACN2GsR5HUKzvC_aw1anOuhL9fk3XUFwU/p.png)
# Hausos-sdc3a
This is the repository containing codes for SKA sdc3a data challenge, our team is named Hausos, which means _The Original Dawn Goddess_, a decent name for the detection of EoR signal of the Universe.
*For more details about the data challenge, please refer to the website (https://sdc3.skao.int/overview)*

## Contents
1. [Introduction](#intro)
1. [Execution](#exec)
1. [Authors](#authors)
1. [Acknowledgments](#ack)

<a name="intro"></a>
## Introduction
The method is a deep learning method for the EoR estimation, which is based on the transformer model. The author admits that the method is not optimized at the current stage due to time limitations.
The repository includes preprocessing, methodology, postprocessing, and evaluation, one can execute them separately.
preprocessing: 
 - cutImage.py: file used to cut testing images into patches
 - cutImage_SDC3.py: file used to cut training images into patches
methodology:
 - main.py: script to run the neural network model, one can choose 'train', 'test' or 'all' mode for the training, testing or training and testing.
 - myDatasets.py: setup for the training and testing datasets
 - utils.py: some utilities
 - myModels/setr: neural network model, which is based on transformer.
postprocessing:
 - combineImage.py: combine predicted patches into images for the final evaluation
evaluation:
 - ps2d.py: function to calculate the 2d power spectrum, which is called in ps2d_script.py
 - ps2d_script.py: script to evaluate the 2d power spectrum

<a name="exec"></a>
## Exectution
4 steps to produce the final 2d power spectrum
1. prepare the data using cutImage_SDC3.py to produce format-correct training patches, and cut the test data into patches using cutImage.py to feed the neural network model
1. train the neural network by seting 'train' in main.py, and 'test' to predict the EoR patches after training
1. combine the patches into images by using combineImage.py
1. evaluate the 2d power spectrum by using ps2d_script.py. Note that the format of power spectrum is saved according to the requirements of SDC3a data challenge.

<a name="authors"></a>
## Authors
On behalf of Hausos team, one can contact the team leader Dr. Ming Jiang via
* **Ming Jiang [mingjiang@xidian.edu.cn](mailto:mingjiang@xidian.edu.cn)**
