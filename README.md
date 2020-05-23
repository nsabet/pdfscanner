# install py-opencv
conda install -c conda-forge py-opencv=4.2.0

conda install -c conda-forge imutils

conda install -c anaconda scikit-image

# exporting environment
conda env export | grep -v "^prefix: " > environment.yml

# creating environment
conda env create -f environment.yml

conda env list
