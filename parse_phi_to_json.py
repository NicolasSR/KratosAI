import numpy as np

def parse_np_array(arr):
    result = ""
    i=1
    for j in range(arr.shape[0]//2):
        result += f'"{i}": [\n'
        result += f'\t\t{np.array2string(arr[j*2], separator=", ")},\n'
        result += f'\t\t{np.array2string(arr[j*2+1], separator=", ")}\n'
        result += '],\n'
        i+=1
    result = result.rstrip(',\n')  # Remove the trailing comma and newline character
    return result

# Example usage
# np_array = np.array([[0.0, 0.0], [0.0, 0.0], [6.536484260689393e-05, -3.433134716808561e-05], [4.0923415159223766e-05, -1.5406483207919906e-05]])
array=np.load('my_phi.npy')[:,:20]
print(array.shape)
result_string = parse_np_array(array)
print(result_string)
print(array[-10:])

with open("my_json_phi.txt", "w") as text_file:
    text_file.write(result_string)