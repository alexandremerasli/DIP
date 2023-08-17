import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import dicom2nifti
from pathlib import Path

### Useful functions to read and save raw ###
def fijii_np(path,shape,type_im):
    """"Transforming raw data to numpy array"""
    file_path=(path)
    dtype = np.dtype(type_im)
    fid = open(file_path, 'rb')
    data = np.fromfile(fid,dtype)
    image = data.reshape(shape[::-1])
    # image = data.reshape(shape)
    return image

def save_img(img,name):
    fp=open(name,'wb')
    img.tofile(fp)

# 1st step : Insert raw to nifti to be read by 3D slicer
step = "insert_raw_to_nifti"
# nifti_not_extracted_yet = True # Override this boolean to True when nifti from DICOM are not extracted yet
nifti_not_extracted_yet = False # Override this boolean to False when nifti from DICOM are extracted
# 2nd step : Extract raw from nifti given by 3D slicer
step = "extract_raw_from_nifti"
# 3nd step : Check registration
step = "check_registration"

step = "insert_raw_to_nifti"

# Boolean to check if orientation is the same between dicom image and raw image
check_orientation_dicom_vs_raw = True

# Define PET image shape from CASToR reconstruction (launch CASToR with same FOV as DICOM image)
PET_shape = (360,360,127)
# Define MR image shape (MR image need to be resampled with imageJ to PET CASToR pixel size !!!)
MR_resampled_shape = (256,126,176)

# Paths
short_PET_path = "pet_dicom_1mm.nii"

if (step == "insert_raw_to_nifti"):

    ### PET ###

    if (nifti_not_extracted_yet):
        # Create folder for PET nifti data to be created
        nifti_PET_to_be_created_path = "data/Algo/Data/database_v2/Alexandre_FDG_Hatem/" + short_PET_path
        Path(nifti_PET_to_be_created_path).mkdir(parents=True, exist_ok=True)
        # Convert DICOM to nifti
        dicom2nifti.convert_dir.convert_directory("/disk/workspace_reco/nested_admm/data/Algo/Data/database_v2/Alexandre_FDG_Hatem/CRANE_PETETMUMAP/patient_fdg/CRANE_MR-PET_CRANE_20230517_143125_322000/_HEAD_PETACQUISITION_AC_IMAGES_30003",nifti_PET_to_be_created_path)
    else:
        # Open nifti image with DICOM information, and put CASToR pixel size
        irm_imagej_nii = nib.load("data/Algo/Data/database_v2/Alexandre_FDG_Hatem/" + short_PET_path)
        # irm_imagej_nii.header["pixdim"] = [-1.,        1.04313*2,   1.04313*2 ,  2.031254,  1.,        1.,        1. ,       1.,      ]
        irm_imagej_nii.header["pixdim"] = [-1.,        1.,          1. ,         2.031254,  1.,        1.,        1. ,       1.,      ]
        # Open raw image from CASToR reconstruction (launch CASToR with same FOV as DICOM image)
        irm_imagej_np = fijii_np("data/Algo/Data/database_v2/Alexandre_FDG_Hatem/MLEM_it2.img",shape=PET_shape,type_im='<f4')
        irm_imagej_np = np.transpose(irm_imagej_np,axes=(2,1,0))[:,::-1,::-1]
        # Save nifti image with good DICOM information and CASToR image
        new_img = nib.Nifti1Image(irm_imagej_np, irm_imagej_nii.affine, irm_imagej_nii.header)
        nib.save(new_img,"data/Algo/Data/database_v2/Alexandre_FDG_Hatem/pet_mlem.nii")


        if (check_orientation_dicom_vs_raw):
            common_slice = 70
            plt.figure()
            plt.imshow(irm_imagej_np[:,:,common_slice])

            plt.figure()
            plt.imshow(irm_imagej_nii.dataobj[:,:,common_slice])
            # plt.show()
            print("end")

    ### MRI ###

    if (nifti_not_extracted_yet):
        # Create folder for MRI nifti data to be created
        nifti_MR_to_be_created_path = "data/Algo/Data/database_v2/Alexandre_FDG_Hatem/resampled_imagej_dicom_orientation.nii"
        Path(nifti_MR_to_be_created_path).mkdir(parents=True, exist_ok=True)
        # Convert DICOM to nifti (MR image need to be resampled with imageJ to PET CASToR pixel size !!!)
        dicom2nifti.convert_dir.convert_directory("/disk/workspace_reco/nested_admm/data/Algo/Data/database_v2/Alexandre_FDG_Hatem/crane t1/",nifti_MR_to_be_created_path)
    else:
        # Open nifti image with DICOM information, and put PET pixel size
        irm_imagej_nii = nib.load("data/Algo/Data/database_v2/Alexandre_FDG_Hatem/resampled_imagej_dicom_orientation.nii")
        # irm_imagej_nii.header["pixdim"] = [-1.,        1.04313*2,   1.04313*2 ,  2.031254,  1.,        1.,        1. ,       1.,      ]
        irm_imagej_nii.header["pixdim"] = [-1.,        1.,          1. ,         2.031254,  1.,        1.,        1. ,       1.,      ]
        # Open raw image (DICOM saved with ImageJ. MR raw could be read with ImageJ with 16-bit unsigned, and big-endian byte order)
        irm_imagej_np = fijii_np("data/Algo/Data/database_v2/Alexandre_FDG_Hatem/resampled_imagej_dicom_orientation.raw",shape=MR_resampled_shape,type_im='>u2')
        # Transpose the image and reverse each necessary dimension until having the images matching (set boolean check_orientation_dicom_vs_raw to True)
        # irm_imagej_np = np.transpose(irm_imagej_np,axes=(0,2,1))[:,::-1,::-1]
        irm_imagej_np = np.transpose(irm_imagej_np,axes=(0,2,1))[:,::-1,::-1]
        # irm_imagej_np = np.transpose(irm_imagej_np,axes=(2,1,0))[:,::-1,::-1]
        # irm_imagej_np = np.transpose(irm_imagej_np,axes=(0,2,1))[:,::-1,::-1]
        # irm_imagej_np = np.transpose(irm_imagej_np,axes=(1,2,0))[::-1,::-1,:]
        # Save nifti image with good DICOM information and reasmpled image from ImageJ
        new_img = nib.Nifti1Image(irm_imagej_np, irm_imagej_nii.affine, irm_imagej_nii.header)
        nib.save(new_img,"data/Algo/Data/database_v2/Alexandre_FDG_Hatem/resampled_imagej_good_hdr_dicom_orientation.nii")

        if (check_orientation_dicom_vs_raw):
            common_slice = 80
            plt.figure()
            plt.imshow(irm_imagej_np[:,:,common_slice])

            plt.figure()
            plt.imshow(irm_imagej_nii.dataobj[:,:,int(2.031254*common_slice)])
            plt.show()
            print("end")

elif (step == "extract_raw_from_nifti"):
    ### Extract raw from nifti (output of slicer) ###

    irm_slicer_nii = nib.load("data/Algo/Data/database_v2/Alexandre_FDG_Hatem/Output.nii")
    save_img(np.transpose(np.array(irm_slicer_nii.dataobj[:,::-1,::-1]).astype(np.float32),axes=(2,1,0)),"data/Algo/image010_3D/mri_for_pet_CASToR.img")
elif (step == "check_registration"):
    ### Check registration ###

    # Load the image
    # mri_for_pet_CASToR = fijii_np("data/Algo/Data/database_v2/Alexandre_FDG_Hatem/resampled_imagej_dicom_orientation.raw",shape=MR_resampled_shape,type_im='>u2')
    # mri_for_pet_CASToR = mri_for_pet_CASToR.astype(np.float64)
    mri_for_pet_CASToR = fijii_np("data/Algo/image010_3D/mri_for_pet_CASToR.img",shape=PET_shape,type_im='<f4')
    pet_CASToR = fijii_np("data/Algo/Data/database_v2/Alexandre_FDG_Hatem/MLEM_it2.img",shape=PET_shape,type_im='<f4')



    pet_CASToR /= np.max(pet_CASToR)
    mri_for_pet_CASToR /= np.max(mri_for_pet_CASToR)

    # Create the figure and axes
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    # Display the image
    common_slice = 60
    RBG_image = np.zeros((PET_shape[0],PET_shape[1],3))
    RBG_image[:,:,0] = mri_for_pet_CASToR[common_slice,:,:] + pet_CASToR[common_slice,:,:]
    RBG_image[:,:,1] = mri_for_pet_CASToR[common_slice,:,:]
    RBG_image[:,:,2] = mri_for_pet_CASToR[common_slice,:,:]
    ax.imshow(RBG_image / np.max(RBG_image))

    # Create the slider
    ax_slider_tradeoff = plt.axes([0.25, 0.2, 0.65, 0.03])
    slider_tradeoff = Slider(ax_slider_tradeoff, 'Tradeoff', 0, 4, valinit=1)
    ax_slider_slice = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider_slice = Slider(ax_slider_slice, 'Slice', 0, PET_shape[2], valinit=common_slice)

    # Update function to change MR/PET contrast and slice
    def update(val):
        x = slider_tradeoff.val
        slice_val = slider_slice.val
        ax.clear()
        RBG_image = np.zeros((PET_shape[0],PET_shape[1],3))
        RBG_image[:,:,0] = mri_for_pet_CASToR[int(slice_val),:,:] + x * pet_CASToR[int(slice_val),:,:]
        RBG_image[:,:,1] = mri_for_pet_CASToR[int(slice_val),:,:]
        RBG_image[:,:,2] = mri_for_pet_CASToR[int(slice_val),:,:]
        ax.imshow(RBG_image / np.max(RBG_image),vmax=0.5)
        fig.canvas.draw_idle()

    slider_tradeoff.on_changed(update)
    slider_slice.on_changed(update)
    plt.show()

if (nifti_not_extracted_yet):
    raise ValueError("The Nifti PET and MR images converted from the DICOM files need to be extracted now from the created '.nii' folders. Change the name of the extracted nifti to the name of the created folder (this folder must be removed). Boolean 'nifti_not_extracted_yet' set to False once it is done.")