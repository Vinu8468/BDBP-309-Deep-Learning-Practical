# Implement convolution operations from scratch.
# Assume a 3x3 kernel and apply it on
# an input image of 32x32.

import numpy as np
# input image of size 32x32
image = np.random.rand(32,32)

# 3x3 kernal
kernal = np.array(
    [
        [1,0,-1],
        [1,0,-1],
        [1,0,-1]
    ]
)
# kernal size
kernal_size = 3
# stride
stride = 1

#dimensions of the results
output_height = (image.shape[0] - kernal_size) // stride + 1
output_width = (image.shape[1] - kernal_size) // stride + 1

#create output feature map
output =np.zeros((output_height,output_width))

# traverse the image
for i in range(0,image.shape[0]-kernal_size+1,stride):
    for j in range(0,image.shape[1]-kernal_size+1,stride):
        #store convlution sum
        conv_sum =0
        #traverse the kernal
        for m in range(kernal_size):
            for n in range(kernal_size):
                # main op
                conv_sum +=image[i+m,j+n] +kernal[m,n]
        #store result
        output[i//stride,j//stride]=conv_sum

print("Input shape:",image.shape)
print("Kernal shape:",kernal.shape)
print("Output shape:",output.shape)


# now lets do a function for it
image = np.reshape(image,(32,32))
kernal = np.array(
    [
        [1,0,-1],
        [1,0,-1],
        [1,0,-1]
    ]
)
def Convolution(image,kernal,stride=1):
    kernal_size = len(kernal)
    image_height,image_width = image.shape[0],image.shape[1]
    output_height = (image_height - kernal_size) // stride + 1
    output_width = (image_width - kernal_size) // stride + 1

    output = np.zeros((output_height,output_width))

    for i in range(0,image.shape[0]-kernal_size+1,stride):
        for j in range(0,image.shape[1]-kernal_size+1,stride):
            conv_sum=0
            for m in range(kernal_size):
                for n in range(kernal_size):
                    conv_sum+=image[i+m,j+n] +kernal[m,n]
            output[i//stride,j//stride]=conv_sum
    return output

output = Convolution(image,kernal,1)
print("Output shape:",output.shape)
print("Image shape:",image.shape)
print("Kernal shape:",kernal.shape)


