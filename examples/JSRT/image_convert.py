import os
from PIL import Image
import numpy as np

def convert_JCRT_images(input_dir, output_dir):
    file_names = os.listdir(input_dir)
    file_names = [item for item in file_names if item[:2] == "JP"]
    print("image number: {0:}".format(len(file_names)))

    for file_name in file_names:
        input_file_name = os.path.join(input_dir,file_name)
        output_file_name = file_name[:-3] + "png"
        output_file_name = os.path.join(output_dir, output_file_name)

        shape = (2048, 2048) # matrix size
        dtype = np.dtype('>u2') # big-endian unsigned integer (16bit)

        # Reading.
        fid = open(input_file_name, 'rb')
        data = np.fromfile(fid, dtype)
        data = data.reshape(shape)

        # Rescale intensity to 0-255
        data[data > 4096] = 4096
        data = data * 255.0 / 4096 #data.max()
        data = np.asarray(data, np.uint8)
        
        # Reshape to 256x256
        img = Image.fromarray(data)
        img = img.resize((256, 256), Image.BILINEAR)
        img.save(output_file_name)
        print(file_name, data.min(), data.max(), data.mean(), data.std())

def convert_JCRT_labels(input_dir, output_dir):
    organs      = ['heart'] 
    sub_folders = ['fold1', 'fold2']

    for sub_folder in sub_folders:
        image_folder = input_dir + '/' + sub_folder + '/masks/' + organs[0]
        image_names  = os.listdir(image_folder)
        image_names  = [item for item in image_names if item[:2] == 'JP']
        for image_name in image_names:
            print(image_name[:-4])
            label_list = []
            for i in range(len(organs)):
                image_folder = input_dir + '/' + sub_folder + '/masks/' + organs[i]
                image_full_name = os.path.join(image_folder, image_name)
                img = Image.open(image_full_name)
                img = img.resize((256,256), Image.NEAREST)
                img = np.asarray(img)
                label_list.append(img)
            label = np.asarray(label_list, np.uint8)
            label = np.max(label, axis = 0)
            label = Image.fromarray(label) 
            output_full_name = image_name[:-4] + ".png"
            output_full_name = os.path.join(output_dir, output_full_name)
            label.save(output_full_name)
    
if __name__ == "__main__":
    input_image_dir = "/home/guotai/data/JSRT/All247images"
    output_image_dir = "/home/guotai/data/JSRT/image"
    convert_JCRT_images(input_image_dir, output_image_dir)

    input_label_dir = "/home/guotai/data/JSRT/scratch"
    output_label_dir = "/home/guotai/data/JSRT/label"
    convert_JCRT_labels(input_label_dir, output_label_dir)