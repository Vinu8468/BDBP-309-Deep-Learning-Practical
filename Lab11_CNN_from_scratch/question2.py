# Implement maxpool operation from scratch.

import numpy as np
# Input feature map
feature_map = np.random.rand(30,30)

# pool size
pool_size = 2

# stride
stride = 2

# output size
output_size = (feature_map.shape[0]-pool_size)//stride+1

# create output
pooled =np.zeros((output_size,output_size))

# max pooling
for i in range(0,output_size-pool_size+1,stride):
    for j in range(0,output_size-pool_size+1,stride):
        # start with the first value in the pooling region
        max_value = feature_map[i,j]
        # traverse the pooling window
        for m in range(pool_size):
            for n in range(pool_size):
                # get current value
                current_value = feature_map[i+m,j+n]
                #compare with maximum value
                if current_value > max_value:
                    max_value = current_value

        # store maximum value
        pooled[i//stride,j//stride]=max_value

print("Input Shape:",feature_map.shape)
print("Output Shape:",pooled.shape)

# function

def max_pool(image,pool_size=2,stride=2):
    input_height=image.shape[0]
    input_width=image.shape[1]
    #output
    output_height = (input_height -pool_size)//stride+1
    output_width = (input_width -pool_size)//stride+1
    # empty output for now
    pooled =np.zeros((output_height,output_width))

    # traverse the feature map
    for i in range(0,output_height-pool_size+1,stride):
        for j in range(0,output_width-pool_size+1,stride):
            #start with first value
            max_value = feature_map[i,j]
            # traverse pooling window
            for m in range(pool_size):
                for n in range(pool_size):
                    # get current value
                    current_value = feature_map[i+m,j+n]
                    # compare values
                    if current_value > max_value:
                        max_value = current_value
            # store maximum value
            pooled[i//stride,j//stride]=max_value
    return pooled

image = np.random.rand(30,30)
pooled = max_pool(feature_map)

print("Input shape:",image.shape)
print("Output Shape:",pooled.shape)