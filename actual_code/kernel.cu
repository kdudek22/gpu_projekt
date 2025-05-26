extern "C" __global__ void process_image(unsigned char* img, int width, int height, int * cameraData, float *voxelSpace, int voxelSpaceXDim, int voxelSpaceYDim, int voxelSpaceZDim, int voxelSpaceUnit) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int image_index = blockIdx.z;

    int cameraPosX = cameraData[image_index * 7];
    int cameraPosY = cameraData[1 + image_index * 7];
    int cameraPosZ = cameraData[2 + image_index * 7];

    int cameraRotX = cameraData[3 + image_index * 7];
    int cameraRotY = cameraData[4 + image_index * 7];
    int cameraRotZ = cameraData[5 + image_index * 7];

    int cameraFov = cameraData[6 + image_index * 7];


    // ***** OPERATIONS *****
    if (x < width && y < height) {

        int idx = width * height * image_index + y * width + x;
        // example: invert pixel
        img[idx] = 255 - img[idx];
    }
}