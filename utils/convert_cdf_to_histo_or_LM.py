import numpy as np
import struct
import pandas as pd

def read_histo_cdf(filename, data, data_time, data_float, data_ID):
    with open(filename, 'rb') as f:
        nb_events = 0
        while True:
            # Read 1 uint32 element
            bytes = f.read(4)  # uint32 is 4 bytes
            if not bytes:
                return data_time
            value = struct.unpack('I', bytes)[0]  # 'H' is format code for uint16
            data_time.append(value)
            data.append(value)
            
            # Read 5 float32 elements
            for idx in range(5):
                bytes = f.read(4)  # float32 is 4 bytes
                if not bytes:
                    return data
                value = struct.unpack('f', bytes)[0]
                data_float[idx].append(value)
                data.append(value)
            
            # Read 2 uint32 elements
            for idx in range(2):
                bytes = f.read(4)  # uint32 is 4 bytes
                if not bytes:
                    return data
                value = struct.unpack('I', bytes)[0]  # 'H' is format code for uint16
                data_ID[idx].append(value)
                data.append(value)
            nb_events += 1


def read_LM_cdf(filename, data, data_time, data_float, data_ID):
    data = []
    with open(filename, 'rb') as f:
        nb_events_local = 0
        while True:
            print(nb_events_local)
            
            # Read 1 uint32 element
            bytes = f.read(4)  # uint32 is 4 bytes
            if not bytes:
                return data_time
            value = struct.unpack('I', bytes)[0]  # 'H' is format code for uint16
            data_time.append(value)
            data.append(value)
            
            # Read 4 float32 elements
            for idx in range(4):
                bytes = f.read(4)  # float32 is 4 bytes
                if not bytes:
                    return data
                value = struct.unpack('f', bytes)[0]
                data_float[idx].append(value)
                data.append(value)
            
            # Read 2 uint32 elements
            for idx in range(2):
                bytes = f.read(4)  # uint32 is 4 bytes
                if not bytes:
                    return data
                value = struct.unpack('I', bytes)[0]  # 'H' is format code for uint16
                data_ID[idx].append(value)
                data.append(value)

            nb_events_local += 1
    
def write_binary_histo_file(data,filename):
    with open(filename, 'wb') as f:
        for i in range(0, len(data), 8):
            # Write 1 uint32 element
            bytes = struct.pack('I', data[i])
            f.write(bytes)
            
            # Write 5 float32 elements
            for j in range(1,5+1):
                bytes = struct.pack('f', data[i+j])
                f.write(bytes)
            
            # Write 2 uint32 elements
            for j in range(6,7+1):
                bytes = struct.pack('I', data[i+j])
                f.write(bytes)

def write_binary_file_from_LM_to_histo(data, data_time, data_float, data_ID,filename, nb_events):
    # Write the data in the LM file and in the LM order
    with open(filename, 'wb') as f:
        for i in range(0, nb_events):
            print(i / nb_events * 100, "%")
            
            # Write 1 uint32 element
            bytes = struct.pack('I', data_time[i])
            f.write(bytes)
            
            # Write 5 float32 elements
            bytes = struct.pack('f', data_float[0][i]) # atn
            f.write(bytes)
            bytes = struct.pack('f', data_float[1][i]) # random
            f.write(bytes)
            bytes = struct.pack('f', data_float[2][i]) # norm
            f.write(bytes)
            bytes = struct.pack('f', data_float[3][i]) # event value
            f.write(bytes)
            bytes = struct.pack('f', data_float[4][i]) # scatter
            f.write(bytes)
            
            # Write 2 uint32 elements
            bytes = struct.pack('I', data_ID[0][i])
            f.write(bytes)
            bytes = struct.pack('I', data_ID[1][i])
            f.write(bytes)

def remove_histogram_from_histo_datafile(data, data_time, data_float, data_ID,filename_full_cdf,filename_cdf_to_write, nb_events, histo_type_to_remove=[]):
    # Read histo cdf file and store data in lists
    read_histo_cdf(filename_full_cdf, data, data_time, data_float, data_ID)
    
    # Write the data in the new histo file and in the histo order
    with open(filename_cdf_to_write, 'wb') as f:
        for i in range(0, nb_events):
            print(i / nb_events * 100, "%")
            
            # Write 1 uint32 element
            bytes = struct.pack('I', data_time[i])
            f.write(bytes)
            
            # Write 5 float32 elements
            if ("atn" not in histo_type_to_remove):
                bytes = struct.pack('f', data_float[0][i]) # atn
                f.write(bytes)
            if ("random" not in histo_type_to_remove):
                bytes = struct.pack('f', data_float[1][i]) # random
                f.write(bytes)
            if ("norm" not in histo_type_to_remove):
                bytes = struct.pack('f', data_float[2][i]) # norm
                f.write(bytes)
            if ("event_value" not in histo_type_to_remove):
                bytes = struct.pack('f', data_float[3][i]) # event value
                f.write(bytes)
            if ("scatter" not in histo_type_to_remove):
                bytes = struct.pack('f', data_float[4][i]) # scatter
                f.write(bytes)
            
            # Write 2 uint32 elements
            bytes = struct.pack('I', data_ID[0][i])
            f.write(bytes)
            bytes = struct.pack('I', data_ID[1][i])
            f.write(bytes)

def remove_data_from_LM_datafile(data, data_time, data_float, data_ID,filename_full_cdf,filename_cdf_to_write, nb_events, histo_type_to_remove=[]):
    # Read histo cdf file and store data in lists
    read_LM_cdf(filename_full_cdf, data, data_time, data_float, data_ID)
    
    # Write the data in the new histo file and in the histo order
    with open(filename_cdf_to_write, 'wb') as f:
        for i in range(0, nb_events):
            print(i / nb_events * 100, "%")
            
            # Write 1 uint32 element
            bytes = struct.pack('I', data_time[i])
            f.write(bytes)
            
            # Write 4 float32 elements
            if ("atn" not in histo_type_to_remove):
                bytes = struct.pack('f', data_float[0][i])
                f.write(bytes)
            if ("scatter" not in histo_type_to_remove):
                bytes = struct.pack('f', data_float[1][i])
                f.write(bytes)
            if ("random" not in histo_type_to_remove):
                bytes = struct.pack('f', data_float[2][i])
                f.write(bytes)
            if ("norm" not in histo_type_to_remove):
                bytes = struct.pack('f', data_float[3][i])
                f.write(bytes)
            
            # Write 2 uint32 elements
            bytes = struct.pack('I', data_ID[0][i])
            f.write(bytes)
            bytes = struct.pack('I', data_ID[1][i])
            f.write(bytes)    

def write_binary_file_from_histo_to_LM(data, data_time, data_float, data_ID, data_event_value, filename, nb_events):
    # Assign the data to the corresponding variables
    data_atn = data_float[0]
    data_scatter = data_float[1]
    data_random = data_float[2]
    data_norm = data_float[3]
    data_ID1 = data_ID[0]
    data_ID2 = data_ID[1]
    
    # Count the number of events in the LM file to change .cdh header manually
    nb_LM_events = 0
    
    # Write the data in the LM file and in the LM order
    with open(filename, 'wb') as f:
        for i in range(0, nb_events):
            print(i / nb_events * 100, "%")
            for event in range(int(data_event_value[i])):
                nb_LM_events += 1
                
                # Write 1 uint32 element
                bytes = struct.pack('I', data_time[i])
                f.write(bytes)
                
                # Write 5 float32 elements
                bytes = struct.pack('f', data_atn[i])
                f.write(bytes)
                bytes = struct.pack('f', data_random[i])
                f.write(bytes)
                bytes = struct.pack('f', data_norm[i])
                f.write(bytes)
                bytes = struct.pack('f', data_event_value[i])
                f.write(bytes)
                bytes = struct.pack('f', data_scatter[i])
                f.write(bytes)
                
                # Write 2 uint32 elements
                bytes = struct.pack('I', data_ID1[i])
                f.write(bytes)
                bytes = struct.pack('I', data_ID2[i])
                f.write(bytes)

    return nb_LM_events

    
def write_binary_file_from_histo_to_norm_for_LM(data, data_float, data_ID,filename, nb_events):

    # Assign the data to the corresponding variables
    data_atn = data_float[0]
    data_norm = data_float[2]
    data_ID1 = data_ID[0]
    data_ID2 = data_ID[1]

    # Write the data in the LM file and in the LM order
    with open(filename, 'wb') as f:
        for i in range(0, nb_events):
            print(i / nb_events * 100, "%")
            
            # Write 2 float32 elements
            bytes = struct.pack('f', data_atn[i])
    
            f.write(bytes)
            bytes = struct.pack('f', data_norm[i])
            f.write(bytes)

            # Write 2 uint32 elements
            bytes = struct.pack('I', data_ID1[i])
            f.write(bytes)
            bytes = struct.pack('I', data_ID2[i])
            f.write(bytes)

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

def write_header_file(filename_to_read, filename_to_write, nb_events, data_mode):
    # Write the new header file from the original one
    with open(filename_to_read, "r") as f:
        with open(filename_to_write, "w") as f1:
            for line in f:
                if line.strip().startswith('Data mode:'):
                    f1.write('Data mode: ' + data_mode)
                    f1.write('\n')
                elif line.strip().startswith('Number of events:'):
                    f1.write('Number of events: ' + str(nb_events))
                    f1.write('\n')
                else:
                    f1.write(line)


############ Variables to be customized by the user
# Path to the histo or LM cdf file
cdf_path = "/home/MEDECINE/mera1140/sherbrooke_workspace/TestCastor/umd_h12_wRot_act_BTB_1_100_df.Cdf"
cdf_path = "data/Algo/Data/database_v2/image40_1/data40_1_1/data40_1_1.cdf"
cdf_path = "data/Algo/Data/database_v2/image40_1/dataLM40_1_1/data40_1_1.cdf"

# Define the number of events (from the header file) and the type of conversion (LM to histo or histo to LM)
if ("LM" in cdf_path):
    LM_to_histo = True
    nb_events = int(8308200 / 4) # LP2 data LM
    nb_events = 1499297 # simu data LM
else:
    LM_to_histo = False
    nb_events = 68516 # simu data histo

# Test removing one histogram type from the histo or LM cdf file. Let to false to not remove any data
# remove_histo = True
remove_histo = False
# remove_LM = True
remove_LM = False

# Define variables to store the data
data, data_time, data_atn, data_random, data_norm, data_event_value, data_scatter, data_float, data_ID1, data_ID2, data_ID = define_data(LM_to_histo)

############ Remove data (norm, atn etc.) from histogram or LM cdf file
if (remove_histo or remove_LM):
    # Define the histogram types to remove and new datafile path
    histo_type_to_remove_str = "" # string to store the histogram types to remove, for the datafile name
    histo_type_to_remove = ["norm","atn"]
    for i in range(0, len(histo_type_to_remove)):
        histo_type_to_remove_str += histo_type_to_remove[i]
    cdf_removed_histo_path = "data/Algo/Data/database_v2/image40_1/data_removed_" + histo_type_to_remove_str + "40_1_1/data40_1_1.cdf"

    # Remove the data from the histo or LM cdf file
    if (remove_histo):
        remove_histogram_from_histo_datafile(data, data_time, data_float, data_ID,cdf_path, cdf_removed_histo_path, nb_events, histo_type_to_remove)
    elif (remove_LM):
        remove_data_from_LM_datafile(data, data_time, data_float, data_ID,cdf_path, cdf_removed_histo_path, nb_events, histo_type_to_remove)

    # End
    print("end")
    exit()

############ Convert histogram cdf file to listmode cdf file
if (not LM_to_histo):
    # Read histo cdf file and store data in lists
    read_histo_cdf(cdf_path, data, data_time, data_float, data_ID)

    # Compute the number of prompt for each bin of sinogram and store them in data_event_value while writing the data in the LM file
    nb_LM_events = write_binary_file_from_histo_to_LM(data, data_time, data_float, data_ID, data_event_value,"data/Algo/Data/database_v2/image40_1/dataLM40_1_1/data40_1_1.cdf", nb_events)
    folder_path = "data/Algo/Data/database_v2/image40_1/dataLM40_1_1/"
    write_binary_file_from_histo_to_norm_for_LM(data, data_float, data_ID, folder_path + "/data_norm40_1_1.cdf", nb_events)
    
    # Write the CASToR header file related to built LM datafile
    write_header_file(cdf_path[:-1] + "h", folder_path + "data40_1_1.cdh", nb_LM_events, 'list-mode')

    # Read LM built file
    cdf_LM_path = "data/Algo/Data/database_v2/image40_1/dataLM40_1_1/data40_1_1.cdf"
    data_LM, data_time_LM, data_atn_LM, data_random_LM, data_norm_LM, data_event_value_LM, data_scatter_LM, data_float_LM, data_ID1_LM, data_ID2_LM, data_ID_LM = define_data(not LM_to_histo)
    read_LM_cdf(cdf_LM_path, data_LM, data_time_LM, data_float_LM, data_ID_LM)


############ Convert listmode cdf file to histogram cdf file
if (LM_to_histo):
    # Read LM cdf file and store data in lists. For now, LOR without events are not taken into account
    read_LM_cdf(cdf_path, data, data_time, data_float, data_ID)

    ### Remove duplicates and get unique indices
    # Create array for each pair of detectors
    data_ID_array = np.zeros((len(data_ID1),2),dtype=np.int32)
    data_ID_array[:,0] = data_ID[0]
    data_ID_array[:,1] = data_ID[1]
    
    # Drop duplicate rows without sorting with panda
    df = pd.DataFrame(data_ID_array, columns=['ID1', 'ID2'])
    df_unique = df.drop_duplicates(keep='first')

    # Assign number of events for each pair of detectors to the event_value
    data_event_value = df_unique.index.values[1:] - df_unique.index.values[:-1]
    data_event_value = np.append(data_event_value, len(data_ID1) - df_unique.index.values[-1])

    # Create a dictionary to store the filtered lists
    data_filtered = {name: np.array(lst)[df_unique.index.values] for name, lst in zip(['data_time', 'data_atn', 'data_scatter', 'data_random', 'data_norm', 'data_ID1', 'data_ID2'], [data_time, data_atn, data_scatter, data_random, data_norm, data_ID1, data_ID2])}
    data_time = data_filtered['data_time']
    data_float = [data_filtered['data_atn'], data_filtered['data_random'], data_filtered['data_norm'], data_event_value, data_filtered['data_scatter']]
    data_ID = [data_filtered['data_ID1'], data_filtered['data_ID2']]

    # Write the data in the histo file
    nb_histo_events = len(data_event_value)
    from pathlib import Path
    folder_path = "data/Algo/Data/database_v2/image40_1/datahisto40_1_1/"
    Path(folder_path).mkdir(parents=True, exist_ok=True) # path to store the histo datafile
    write_binary_file_from_LM_to_histo(data, data_time, data_float, data_ID, folder_path + "data40_1_1.cdf", nb_histo_events)
    nb_events_without_0_LOR = len(data_event_value.nonzero()[0])

    # Write the CASToR header file related to built histo datafile
    write_header_file(cdf_path[:-1] + "h", folder_path + "data40_1_1.cdh", nb_histo_events, 'histogram')

    # Read histo built file
    cdf_histo_path = folder_path + "data40_1_1.cdf"
    data_histo, data_time_histo, data_atn_histo, data_random_histo, data_norm_histo, data_event_value_histo, data_scatter_histo, data_float_histo, data_ID1_histo, data_ID2_histo, data_ID_histo = define_data(not LM_to_histo)
    read_histo_cdf(cdf_histo_path, data_histo, data_time_histo, data_float_histo, data_ID_histo)

print("end")