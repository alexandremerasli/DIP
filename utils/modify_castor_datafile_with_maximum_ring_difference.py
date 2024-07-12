import numpy as np
from re import split
from pathlib import Path
import struct


def fijii_np(path,shape,type_im='<f'):
    """"Transforming raw data to numpy array"""               
    file_path=(path)
    if (1 in shape):
        nb_dimensions = 2
    else:    
        nb_dimensions = 3
    dtype_np = np.dtype(type_im)
    with open(file_path, 'rb') as fid:
        data = np.fromfile(fid,dtype_np)
        if (nb_dimensions == 2): # 2D
            image = data.reshape(shape)
        else: # 3D
            image = data.reshape(shape[::-1])
    return image

def save_img(img,name):
    fp=open(name,'wb')
    img.tofile(fp)
    print('Succesfully save in:', name)

def atoi(text):
    return int(text) if text.isdigit() else text
    
def natural_keys(text):
    return [ atoi(c) for c in split(r'(\d+)', text) ] # APPGML final curves + resume computation

def define_data(LM_to_histo):
    # Define the variables to store the data
    data = []
    data_time = []
    data_atn = []
    data_random = []
    data_norm = []
    data_scatter = []
    data_ID1 = []
    data_ID2 = []
    data_event_value = []

    # Group the data in a lists
    data_ID = [data_ID1, data_ID2]
    if (LM_to_histo):
        data_float = [data_atn, data_random, data_norm, data_scatter]
    else:
        data_float = [data_atn, data_random, data_norm, data_event_value, data_scatter]

    return data, data_time, data_atn, data_random, data_norm, data_event_value, data_scatter, data_float, data_ID1, data_ID2, data_ID

def ring_difference(ID1_CASToR, ID2_CASToR):
    # Set UHR parameters
    voxel_size = 1.2
    nb_detectors_per_ring = 896

    ### Convert the CASToR ID to the true ring ID (considering rings without detectors in UHR)
    # Compute coordinates of the detectors based on CASToR ID
    CASToR_coordinate_1_axial = ID1_CASToR // nb_detectors_per_ring
    CASToR_coordinate_2_axial = ID2_CASToR // nb_detectors_per_ring
    # Add rings without detectors in axial coordinate
    real_coordinate_1_axial = CASToR_coordinate_1_axial // 8 + CASToR_coordinate_1_axial // 4 + CASToR_coordinate_1_axial
    real_coordinate_2_axial = CASToR_coordinate_2_axial // 8 + CASToR_coordinate_2_axial // 4 + CASToR_coordinate_2_axial
    


    # CASToR_coordinate_1_transaxial = ID1_CASToR % nb_detectors_per_ring
    # # Compute the real ID
    # ID1_real = real_coordinate_1_axial * nb_detectors_per_ring + CASToR_coordinate_1_transaxial
    
    # Compute the ring difference between the two real detectors
    RD = abs(real_coordinate_1_axial - real_coordinate_2_axial) * voxel_size

    return RD

def modify_MRD_histogram_from_histo_datafile(filename_read, filename_write, data, data_time, data_float, data_ID, MRD):
    with open(filename_read, 'rb') as f_read:
        with open(filename_write, 'wb') as f_write:
            nb_events = 0
            while True:
                print(nb_events)
                ### Read one event
                # Read 1 uint32 element
                bytes = f_read.read(4)  # uint32 is 4 bytes
                if not bytes:
                    return data_time
                value = struct.unpack('I', bytes)[0]  # 'H' is format code for uint16
                data_time.append(value)
                data.append(value)
                
                # Read 5 float32 elements
                for idx in range(5):
                    bytes = f_read.read(4)  # float32 is 4 bytes
                    if not bytes:
                        return data
                    value = struct.unpack('f', bytes)[0]
                    data_float[idx].append(value)
                    data.append(value)
                
                # Read 2 uint32 elements
                for idx in range(2):
                    bytes = f_read.read(4)  # uint32 is 4 bytes
                    if not bytes:
                        return data
                    value = struct.unpack('I', bytes)[0]  # 'H' is format code for uint16
                    data_ID[idx].append(value)
                    data.append(value)

                ### Write event if ring difference <= MRD
                if (ring_difference(data_ID[0][-1], data_ID[1][-1]) <= MRD):
                    # Write 1 uint32 element
                    bytes = struct.pack('I', data[8*nb_events])
                    f_write.write(bytes)
                    
                    # Write 5 float32 elements
                    for j in range(1,5+1):
                        bytes = struct.pack('f', data[8*nb_events+j])
                        f_write.write(bytes)
                    
                    # Write 2 uint32 elements
                    for j in range(6,7+1):
                        bytes = struct.pack('I', data[8*nb_events+j])
                        f_write.write(bytes)
                    
                    nb_events += 1

    return nb_events


def modify_MRD_data_from_LM_datafile(filename_read, filename_write, MRD):
    with open(filename_read, 'rb') as f_read:
        with open(filename_write, 'wb') as f_write:
            nb_events = 0
            nb_events_MRD = 0
            while True:
                print(nb_events)
                ### Read one event
                # Read 1 uint32 element
                bytes = f_read.read(4)  # uint32 is 4 bytes
                if not bytes:
                    return data_time
                value = struct.unpack('I', bytes)[0]  # 'H' is format code for uint16
                data_time = value
                
                # Read 4 float32 elements
                data_float = np.zeros(nb_data_cdf - 3)
                for idx in range(nb_data_cdf - 3):
                    bytes = f_read.read(4)  # float32 is 4 bytes
                    if not bytes:
                        return data
                    value = struct.unpack('f', bytes)[0]
                    data_float[idx] = value

                nb_events += 1
                
                # Read 2 uint32 elements
                data_ID = np.zeros(2, dtype=int)
                for idx in range(2):
                    bytes = f_read.read(4)  # uint32 is 4 bytes
                    if not bytes:
                        return data
                    value = struct.unpack('I', bytes)[0]  # 'H' is format code for uint16
                    data_ID[idx] = value

                ### Write event if ring difference <= MRD
                if (ring_difference(data_ID[0], data_ID[1]) <= MRD):
                    # Write 1 uint32 element
                    bytes = struct.pack('I', data_time)
                    f_write.write(bytes)
                    
                    # Write 4 float32 elements
                    # for j in range(1,4+1):
                    #     bytes = struct.pack('f', data[nb_data_cdf * nb_events+j])
                    #     f_write.write(bytes)
                    
                    # Write 2 uint32 elements
                    for j in range(2):
                        bytes = struct.pack('I', data_ID[j])
                        f_write.write(bytes)
                    
                    nb_events_MRD += 1

    return nb_events

############ Variables to be customized by the user
# Path to the histo or LM cdf file
subroot = "data/Algo/"
phantom = "image40_1"
phantom = "imageUHR_IEC"
phantom = "imageUHR_IEC4_8"
cdf_path = subroot + "/Data/database_v2/" + phantom + "/data" + phantom[5:] + "_1/data" + phantom[5:] + "_1.cdf"

# Number of data type (time, atn, norm, scatter, random, event value, ID1, ID2) for each event in the cdf file
nb_data_cdf = 3 # Smaller LM file (for UHR data without physics modelling)
# nb_data_cdf = 8 # Larger histo file
# nb_data_cdf = 7 # Larger LM file

# Define the number of events (from the header file) and the type of conversion (LM to histo or histo to LM)
if ("LM" in cdf_path):
    LM_to_histo = True
    nb_events = int(8308200 / 4) # LP2 data LM
    nb_events = 1499297 # simu data LM
    nb_events = 100382193 # thirdTestBis UHR data LM
else:
    LM_to_histo = False
    nb_events = 68516 # simu data histo

# Remove events related to LORs with too high Maximum Ring Difference (MRD), from the histo or LM cdf file. Let to false to not modify any data
# modify_MRD_histo = True
modify_MRD_histo = False
modify_MRD_LM = True
# modify_MRD_LM = False
MRD = 2 # Maximum Ring Difference (MRD) in millimeters

# Define variables to store the data
data, data_time, data_atn, data_random, data_norm, data_event_value, data_scatter, data_float, data_ID1, data_ID2, data_ID = define_data(LM_to_histo)

############ Modify data (norm, atn etc.) from histogram or LM cdf file
if (modify_MRD_histo or modify_MRD_LM):
    # Define the histogram types to modify and new datafile path
    histo_type_to_modify_str = "" # string to store the histogram types to modify, for the datafile name
    cdf_modified_folder = subroot + "/Data/database_v2/" + phantom + "/dataMRD_filtered_" + phantom[5:] + "_1/"
    Path(cdf_modified_folder).mkdir(parents=True, exist_ok=True) # path to store the new datafile
    cdf_modified_histo_path = cdf_modified_folder + "data" + phantom[5:] + "_1.cdf"

    # Remove the data from the histo or LM cdf file
    if (modify_MRD_histo):
        nb_events_MRD = modify_MRD_histogram_from_histo_datafile(cdf_path, cdf_modified_histo_path, data, data_time, data_float, data_ID, MRD)
    elif (modify_MRD_LM):
        nb_events_MRD = modify_MRD_data_from_LM_datafile(cdf_path, cdf_modified_histo_path, MRD)

    # Show nb_events_MRD
    print("nb_events_MRD = ", nb_events_MRD)

    # End
    print("end")
    exit()    