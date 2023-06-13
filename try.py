import numpy as np

def extract_segments(list_all, element_list):
    extracted_segments = []
    arr_all = np.array(list_all)
    arr_elements = np.array(element_list)
    print(len(list_all))

    while True:
        mask = np.isin(arr_all, arr_elements)
        if not np.any(mask): #if none of the elements are present anymore
            extracted_segments.append(arr_all[0:]) #add the last bit
            break
        first_occurrence = np.argmax(mask)
        segment = arr_all[:first_occurrence+1]
        extracted_segments.append(segment)
        arr_all = arr_all[first_occurrence+1:]
    result = np.array([])
    for seg in extracted_segments:
        result = np.concatenate((result, other(len(seg))))


    return extracted_segments
def other(i):
    return np.ones(i)

# Example usage
list_all = [1, 5, 9, 5, 3, 2, 4, 5, 4, 6, 2, 6, 8, 4, 10, 2, 4, 12, 13]
element_list = [2, 4]

segments = extract_segments(list_all, element_list)

for segment in segments:
    print(segment)
