import re
import json

filename = "text_log_to_history.txt"

with open(filename, "r") as f:
    text = f.read()

###########   FOR KERAS TRAINED MODELS  ###############
""" # Use regular expressions to extract the values for each epoch
pattern = r"Epoch \d+/\d+\n.*loss_x: ([\d\.e+-]+).*err_r: ([\d\.e+-]+).*val_loss_x: ([\d\.e+-]+).*val_err_r: ([\d\.e+-]+)"
matches = re.findall(pattern, text)

history={
    "loss_x": [],
    "err_r": [],
    "val_loss_x": [],
    "val_err_r": []
}

# Convert the results to a dictionary
for i, match in enumerate(matches):
    history["loss_x"].append(float(match[0]))
    history["err_r"].append(float(match[1]))
    history["val_loss_x"].append(float(match[2]))
    history["val_err_r"].append(float(match[3]))

print(history)

with open("out_history.json", "w") as f:
    # Use the JSON module to write the dictionary to the file
    json.dump(history, f) """


###########   FOR BFGS TRAINED MODELS  ###############
# Use regular expressions to extract the values for each epoch
# pattern = r"At iterate    \d+    f=  [\d\.D+-]+    |proj g|=  [\d\.D+-]+\n.*loss: ([\d\.e+-]+).*err_r: ([\d\.e+-]+)"
pattern = r"\d+/\d+ \[+.*loss: ([\d\.e+-]+).*err_r: ([\d\.e+-]+).+\nAt iterate"

# At iterate    0    f=  1.10486D-05    |proj g|=  8.79314D-03
# 19999/20000 [============================>.] - ETA: 0s - loss: 1.7838e-07 - err_r: 8.4185e-06


matches = re.findall(pattern, text)

history={
    "loss_x": [],
    "err_r": [],
}

# Convert the results to a dictionary
for i, match in enumerate(matches):
    history["loss_x"].append(float(match[0]))
    history["err_r"].append(float(match[1]))

print(history)

with open("out_history.json", "w") as f:
    # Use the JSON module to write the dictionary to the file
    json.dump(history, f)