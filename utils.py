import numpy as np
import pydicom as pd



def load_dicom(ds):
    image = ds.pixel_array
