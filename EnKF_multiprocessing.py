# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 17:14:05 2021

@author: gaso001
"""
# -*- coding: utf-8 -*-
# Copyright (c) 2021 Wageningen-UR
# Deborah Gaso Melgar and Allard de Wit, Febrary 2021

#from: https://github.com/ajwdewit/pcse_notebooks/blob/master/08_data_assimilation_with_the_EnKF.ipynb
import sys, os.path
this_dir = os.getcwd()
up_dir = os.path.dirname(this_dir)
if not up_dir in sys.path:
    sys.path.append(up_dir)

import config
import sys, os
import time
from itertools import product

from multiprocessing import Pool
import rasterio
from pandas import DataFrame

from run_EnKF_2 import run_EnKF

## test mp
def gather_all_pixels():
    """Gathers the cols/rows for all pixels with data

    :return: a list of [(col, row), (col, row),...]
    """
    variable = "ci"
    flist = sorted([fname for fname in os.listdir(config.s2_dir)
                    if fname.endswith('.tif') and variable in fname])
    for fname in flist:
        im = rasterio.open(os.path.join(config.s2_dir, fname))
        imarray = im.read(1)
        break
    relevant_pixels = [] 
    i=0
    for (row, col) in product(range(im.shape[0]), range(im.shape[1])):
        i+=1
        if imarray[row, col] == config.S2_nodata:
            continue
        relevant_pixels.append((col, row))
        
    return relevant_pixels

def run_one_pixel(inputs):
    """Runs for one pixel with given inputs.

    Function is mainly here to handle exceptions and to unpack the inputs tuple.

    :param inputs: a tuple of type (year, (col, row))
    :return: the optimized parameters.
    """
    year, (col, row) = inputs
    try:
        r = run_EnKF(year, col, row, silent=True)
    except Exception as e:
        print(f"Run EnKF failed on year: {year}, row: {row}, col: {col}")
        r = None
    return r

def optimizer_with_mp(year):
    """Runs the optimization using multiprocessing.Pool
    """
    
#    pixels_for_DA  = gather_all_pixels()
#    relevant_years_pixels = list(product(years, pixels_for_DA))
    years = [2020, ]
    pixels_for_DA=gather_all_pixels()
    relevant_years_pixels = list(product(years, pixels_for_DA))

    p = Pool(6)
    start_time = time.time()
    results = p.map(run_one_pixel, relevant_years_pixels)
    df = DataFrame(results)
    df1=df.T
    df1.to_csv("Results_DROL.csv")
    
    end_time = time.time()-start_time
    print("\n")
    print(f"Processing{len(relevant_years_pixels)} numbers took {end_time} time using multiprocessing.")
    print(results)
#    
#    for result, (year, (col, row)) in zip(results, relevant_years_pixels):
#        print(f"optimized parameters for year: {year}, row: {row}, col: {col}")
#        for parname, value in zip(config.selected_parameters, result):
#            print(f" - {parname}: {value}")            
    return 

if __name__ == "__main__":
    
    #ensemble_growingseason(year=2020, col=8, row=5)
    optimizer_with_mp(year=2020)
