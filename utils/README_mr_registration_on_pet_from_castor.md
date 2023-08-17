- PET : CASToR
  - Launch CASToR with same FOV as DICOM image
    - ex : DICOM FOV : 358.84x358.84x257.98mm, 344x344x127, 1.0431x1.0431x2.0313mm
    - CASToR FOV : if we want 1x1x2.0313mm => to have same FOV, launch CASToR with -vox 359x359x127

- MRI : ImageJ
  - Ctrl+E : Image -> Scale : resample MRI to CASToR PET
  - Save as raw data “resampled_imagej_dicom_orientation.raw”

- Python code
  - Transpose the image and reverse each necessary dimension until having the images views matching (set boolean check_orientation_dicom_vs_raw to True). The slices do not need to perfectly match, it is just to check if the view is the same.

- Slicer
  - Import saved nifti files (short_PET_path and nifti_MR_to_be_created_path)
  - Ctrl + F : Resample image (BRAINS)
    - Image to Warp : put the MR volume
    - Reference Image : put the PET volume
    - Output Image : Create new Volume
    - Click on “Apply” : a volume “output image” should have been created
  - Save the “output image” as “Output.nii”

- Python code
  - Extract raw from “Output.nii”
  - Check registration
