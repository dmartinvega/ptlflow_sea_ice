import cv2 as cv
import ptlflow
from ptlflow.utils import flow_utils, flowpy_torch
from ptlflow.utils.io_adapter import IOAdapter


# ## Read jenny's flow (the one I provided them with)
# flow_loaded_jenny = flow_utils.flow_read(input_data = '20210313_154931_DVWF_HH_8bit_20m-jenny.flo', format = 'flo')
# print(f'loaded: {flow_loaded_jenny.shape}')


### version I shared with Ella and Jenny
# import numpy as np
# def read_flo(file_path):
# # Abre el archivo .flo en modo binario
#     with open(file_path, 'rb') as f:
#         magic = np.fromfile(f, np.float32, count=1)
#         #if magic != 202021.25:
#         if magic[0] != 202021.25:
#             raise Exception("Incorrect .flo file format")

#         width = np.fromfile(f, np.int32, count=1)
#         height = np.fromfile(f, np.int32, count=1)
#         print(f"Width: {width[0]}, Height: {height[0]}")

#         data = np.fromfile(f, np.float32, count=2 * width[0] * height[0])
#         print(f"Data size: {data.size}")

#         flow = np.resize(data, (height[0], width[0], 2))

#         return flow
# ##
# flo_file = '20210313_154931_DVWF_HH_8bit_20m-jenny.flo'
# flow = read_flo(flo_file)
# print(f'loaded: {flow.shape}')




# # Get an optical flow model. As as example, we will use RAFT Small
# # with the weights pretrained on the FlyingThings3D dataset
# model = ptlflow.get_model('sea_raft_l', ckpt_path='kitti')
# model.eval()

# # Load the images
# images = [
#     cv.imread('20210313_154931_DVWF_HH_8bit_20m.tif'),
#     cv.imread('20210314_032432_SCWA_HH_8bit_50m.tif')
# ]

# # A helper to manage inputs and outputs of the model
# io_adapter = IOAdapter(model, images[0].shape[:2])

# # inputs is a dict {'images': torch.Tensor}
# # The tensor is 5D with a shape BNCHW. In this case, it will have the shape:
# # (1, 2, 3, H, W)
# inputs = io_adapter.prepare_inputs(images)

# # Forward the inputs through the model
# predictions = model(inputs)

# # The output is a dict with possibly several keys,
# # but it should always store the optical flow prediction in a key called 'flows'.
# flows = predictions['flows']

# # flows will be a 5D tensor BNCHW.
# # This example should print a shape (1, 1, 2, H, W).
# print(flows.shape)
# print(type(flows))
# # Create an RGB representation of the flow to show it on the screen
# flow_rgb = flow_utils.flow_to_rgb(flows)
# # Make it a numpy array with HWC shape
# flow_rgb = flow_rgb[0, 0].permute(1, 2, 0)
# flow_rgb_npy = flow_rgb.detach().cpu().numpy()
# # OpenCV uses BGR format
# flow_bgr_npy = cv.cvtColor(flow_rgb_npy, cv.COLOR_RGB2BGR)

# # Show on the screen
# #cv.imshow('image1', images[0])
# #cv.imshow('image2', images[1])
# #cv.imshow('flow', flow_bgr_npy)
# #cv.waitKey()
# #print(flows)

# array = flows.detach().squeeze().permute(1, 2, 0).cpu().numpy()
# print(array.shape)
# flow_utils.flow_write(output_file = '20210313_154931_DVWF_HH_8bit_20m.flo', flow = array)
#flow_loaded = flow_utils.flow_read(input_data = '20210313_154931_DVWF_HH_8bit_20m.flo', format = 'flo')
# print(flow_loaded.shape)
# import torch
# flow_loaded = torch.from_numpy(flow_loaded)
# array = torch.from_numpy(array)
# print(torch.equal(array, flow_loaded))
# # cv.imwrite("image1.png", flow_rgb_npy*255)


# import numpy as np
# flow_utils.flow_write(output_file = '20210313_154931_DVWF_HH_8bit_20m.png', flow = array.astype(np.int64)*255)

# # png load to see if results are u,v, then check if they can be exported to .flo and reload them again
# png_loaded = flow_utils.flow_read(input_data = '20210313_154931_DVWF_HH_8bit_20m.png', format = 'png')
# print(png_loaded.shape)
# print(png_loaded)
# flow_rgb = flow_utils.flow_to_rgb(png_loaded)[:, :, ::-1]
# print(flow_rgb.shape)
# cv.imwrite("20210313_154931_DVWF_HH_8bit_20m-reconverted.png", flow_rgb)

# WORKING: flo load, convert to rgb and save png. This is to test if .flo results are ok. My results from official experiment were bad
# to read and write .flo files: use flow_utils.flow_read and flow_utils.flow_write.
flow_loaded = flow_utils.flow_read(input_data = 'outputs/sidex_40km/sea_raft_l_kitti/flows/images/20210313_154931_DVWF_HH_8bit_20m.flo', format = 'flo')
print(flow_loaded.shape)
# print(flow_loaded)
flow_rgb = flow_utils.flow_to_rgb(flow_loaded)[:, :, ::-1]
print(flow_rgb.shape)
cv.imwrite("20210313_154931_DVWF_HH_8bit_20m-sea_raft_l_kitti.png", flow_rgb)
# flow_utils.flow_write(output_file = '20210313_154931_DVWF_HH_8bit_20m-from-flow-write.flo', flow = flow_loaded)

