for ((phantom_type=1;phantom_type<=4;phantom_type++)); do
    for num_epoch in {1000,4000,7500,9000}; do

        # Gong
        python3 utils/diff_images.py --img2 "/home/meraslia/workspace_reco/nested_admm/data/Algo/Data/database_v2/image3_$phantom_type/image3_$phantom_type.raw" --img1 "/home/meraslia/workspace_reco/nested_admm/data/Algo/image3_$phantom_type/replicate_1/Gong/Block2/post_reco config_image=BSREM_it30_rho=0_adapt=nothing_mu_DI=120_tau_D=1.5_lr=0.01_opti_=Adam_skip_=0_scali=nothing_input=CT_nb_ou=3_mlem_=False/out_cnn/24/out_DIP-100_epoch=$num_epoch.img"
        mv "/home/meraslia/workspace_reco/nested_admm/data/Algo/diff_img.png" "/home/meraslia/Documents/Thèse/Résultats à montrer/2023_09_13/bias images ReLU exp/automatic/bias_img_Gong_3_"$phantom_type"_"$num_epoch"it.png"
        # nested ADMM
        python3 utils/diff_images.py --img2 "/home/meraslia/workspace_reco/nested_admm/data/Algo/Data/database_v2/image3_$phantom_type/image3_$phantom_type.raw" --img1 "/home/meraslia/workspace_reco/nested_admm/data/Algo/image3_$phantom_type/replicate_1/nested/Block2/post_reco config_image=BSREM_it30_rho=0_adapt=nothing_mu_DI=120_tau_D=1.5_lr=0.01_opti_=Adam_skip_=0_scali=nothing_input=CT_nb_ou=3_alpha=1_adapt=both_mu_ad=2_tau=100_tau_m=100_stopp=0_saveS=1_mlem_=False/out_cnn/24/out_DIP-100_epoch=$num_epoch.img"
        mv "/home/meraslia/workspace_reco/nested_admm/data/Algo/diff_img.png" "/home/meraslia/Documents/Thèse/Résultats à montrer/2023_09_13/bias images ReLU exp/automatic/bias_img_nested_3_"$phantom_type"_"$num_epoch"it.png"

    done
done