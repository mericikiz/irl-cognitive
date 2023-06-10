import numpy as np

def normalize_array(arr):
    #makes every array have all its elements between -1 and 1, keeping + values  +, - values - and 0s 0
    max_magnitude_value = max(arr, key=abs)
    normalized_arr = arr/abs(max_magnitude_value)
    #if max_magnitude_value<0:
    #    normalized_arr = np.where(normalized_arr < 0, normalized_arr / abs(normalized_arr), normalized_arr)
    #if max_magnitude_value > 0:


    return normalized_arr

def normalize_array_old(arr):
    #makes every array have all its elements between -1 and 1, keeping + values  +, - values - and 0s 0
    normalized_arr = arr
    negatives = arr[arr < 0]
    if len(negatives)>0:
        min_negative = np.min(negatives)
        normalized_arr = np.where(normalized_arr < 0, normalized_arr / abs(min_negative), normalized_arr)
    positives = arr[arr>0]
    if len(positives)>0:
        max_positive = np.max(positives)
        normalized_arr = np.where(normalized_arr > 0, normalized_arr / max_positive, normalized_arr)

    return normalized_arr

def normalize_array_oldd(arr):
    #makes every array have all its elements between -1 and 1, keeping + values  +, - values - and 0s 0
    normalized_arr = arr
    min_negative = np.min(normalized_arr)
    if min_negative<0:
        normalized_arr = np.where(normalized_arr < 0, normalized_arr / abs(min_negative), normalized_arr)

    max_positive = np.max(normalized_arr)
    if max_positive>0:
        normalized_arr = np.where(normalized_arr > 0, normalized_arr / max_positive, normalized_arr)

    return normalized_arr

def normalize_one_row(arr, row_no): #normalize according to all array but only one row changes
    min = np.min(arr)
    max = np.max(arr)
    arr_row = arr[row_no]
    if min<0:
        arr_row = np.where(arr_row < 0, arr_row / abs(min), arr_row)
    if max>0:
        arr_row = np.where(arr_row > 0, arr_row / max, arr_row)
    return arr_row
