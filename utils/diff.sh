# ADMMLim
#: '
for outer_it in {1..1}
do
echo "outer_it"$outer_it
    for inner_it in {1..500..50}
    do
        global_path="/home/meraslia/sgld/hernan_folder/data/Algo/replicate_1/Comparison/ADMMLim/config_rho=0_sub_i=500_alpha=0.005_mlem_=True/"
        path1=$global_path"ADMM_1/0_"$outer_it"_it"$inner_it".img"
        path64=$global_path"ADMM_64/0_"$outer_it"_it"$inner_it".img"

        #path1=$global_path"ADMM_1_forward_lor/0_"$outer_it"_it"$inner_it".img"
        #path64=$global_path"ADMM_64_forward_lor/0_"$outer_it"_it"$inner_it".img"

        python3 utils/diff_images.py --img1 $path1 --img2 $path64
    done
done
#'

        
# MLEM
: '
for outer_it in {1..1000..50}
do
    echo "outer_it"$outer_it
    global_path="/home/meraslia/sgld/hernan_folder/data/Algo/replicate_1/Comparison/"
    path1=$global_path"MLEM_1/config_rho=0_mlem_=True/MLEM_beta_0_it"$outer_it".img"
    path64=$global_path"MLEM_64/config_rho=0_mlem_=True/MLEM_beta_0_it"$outer_it".img"

    python3 utils/diff_images.py --img1 $path1 --img2 $path64
done
'
