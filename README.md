This work is in progress and will be soon cleaned for users interested to use the DNA PET reconstruction algorithm.



## Project Subtitle

This project involves saving the 2D system matrix from the CASToR reconstruction software to be used in a Python script.

### Steps

2. **Run CASToR with Good Voxel Size to save system matrix**
    - CASToR should use this files: oProjectionLine_save_system_matrix.cc, oProjectionLine_save_system_matrix.hh, iIterativeAlgorithm_save_system_matrix.cc
    - Run the following command:
        - `castor-recon -dim 112,112,1 -df header_datafile_castor.cdh-conv gaussian,4,1,3.5::psf -fout folder_system_matrix_run -it 1:1 -th 1 -proj-comp 1 -vox desired_voxel_size, desired_voxel_size, desired_voxel_size

3. **Add System Matrix Elements for sinogram bins with zero value**
    - Run the `utils/add_system_matrix_elem.py` script.

4. **Compare to CASToR Forward Projection**
    - If needed to compare to CASToR forward projection:
        - Run CASToR reconstruction with ADMMLim optimizer, and save m2p_vectorAx[0] into .img file.
        - Run the `utils/add_zero_to_sino.py` script with initialisation x (could be the phantom for instance), with input m2p_vectorAx[0].
        - Compare it to matrix product in Python between saved system matrix and image x.

## Authors

- Alexandre Merasli, PhD thesis

## Supervisors

Simon Stute, Thomas Carlier


## Project Subtitle

Ablation study (DNA - DIPRecon)

### Steps

2. **Run with DIPRecon without ReLU**
    - Run Python script with config file all_config/Gong_without_ReLU.py (1851221 means no ReLU):

3. **Add System Matrix Elements for sinogram bins with zero value**
    - Run the `utils/add_system_matrix_elem.py` script.

4. **Compare to CASToR Forward Projection**
    - If needed to compare to CASToR forward projection:
        - Run CASToR reconstruction with ADMMLim optimizer, and save m2p_vectorAx[0] into .img file.
        - Run the `utils/add_zero_to_sino.py` script with initialisation x (could be the phantom for instance), with input m2p_vectorAx[0].
        - Compare it to matrix product in Python between saved system matrix and image x.

## Authors