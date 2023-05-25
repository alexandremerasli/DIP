
for ((replicate_id=1;replicate_id<=100;replicate_id++)); do
    #mv /disk/workspace_reco/nested_admm/data/Algo/image2_0/replicate_"$replicate_id"/BSREM /disk/workspace_reco/nested_admm/data/Algo/image2_0/replicate_"$replicate_id"/BSREM_quadratic
    #mv /disk/workspace_reco/nested_admm/data/Algo/metrics/image2_0/replicate_"$replicate_id"/APGMAP_quadratic /disk/workspace_reco/nested_admm/data/Algo/metrics/image2_0/replicate_"$replicate_id"/APGMAP_Nuyts
    #rm -rf /disk/workspace_reco/nested_admm/data/Algo/metrics/image2_0/replicate_"$replicate_id"/APGMAP
    #rm -rf /disk/workspace_reco/nested_admm/data/Algo/image2_0/replicate_"$replicate_id"/APGMAP_Nuyts_bad_rho
    #mkdir -p /home/meraslia/workspace_reco/nested_admm/data/Algo/Data/initialization/ADMMLim_100it/replicate_"$replicate_id"
    #cp /disk/workspace_reco/nested_admm/data/Algo/image2_0_admm/replicate_"$replicate_id"/ADMMLim/config_rho=0_alpha=1_adapt=tau_mu_ad=2_tau=100_mlem_=False_post_=0/0_it100.img /home/meraslia/workspace_reco/nested_admm/data/Algo/Data/initialization/ADMMLim_100it/replicate_"$replicate_id"/ADMMLim_it100.img
    #mkdir -p /disk/workspace_reco/nested_admm/data/Algo/Data/initialization/BSREM_30it/replicate_"$replicate_id"
    #cp /disk/workspace_reco/nested_admm/data/Algo/image2_0/replicate_"$replicate_id"/BSREM_Nuyts/config_rho=0.01_mlem_=False/BSREM_it30.img /disk/workspace_reco/nested_admm/data/Algo/Data/initialization/BSREM_30it/replicate_"$replicate_id"/BSREM_it30.img
    #rm -rf data/Algo/image2_0/replicate_"$replicate_id"/Gong
    #cd data/Algo/image2_0/replicate_"$replicate_id"/nested/Block1/config_rho=0.003_adapt=nothing_mu_DI=10_tau_D=2_lr=0.01_sub_i=100_opti_=Adam_skip_=3_scali=standardization_input=random_nb_ou=3_alpha=1_adapt=tau_mu_ad=2_tau=100_mlem_=False/before_eq22
    #ls -1q * | wc -l

    #mv /disk/workspace_reco/nested_admm/data/Algo/image2_0_nested_and_Gong_home/ /home/meraslia/workspace_reco/nested_admm/data/Algo/image2_0/
    mkdir /home/meraslia/workspace_reco/nested_admm/data/Algo/Data/initialization/image40_0/BSREM_30it/replicate_"$replicate_id"
    cp /home/meraslia/workspace_reco/nested_admm/data/Algo/image40_0/replicate_"$replicate_id"/BSREM/config_image=1_im_value_cropped_rho=0.01_mlem_=False/BSREM_it30.img /home/meraslia/workspace_reco/nested_admm/data/Algo/Data/initialization/image40_0/BSREM_30it/replicate_"$replicate_id"/BSREM_it30.img
    #python3 main_nested_ADMMLim_stand.py
    #scp -r /home/meraslia/workspace_reco/nested_admm/data/Algo/metrics/image2_0/replicate_"$replicate_id"/Gong/ liu:/home/meraslia/workspace_reco/nested_admm/data/Algo/metrics/image2_0/replicate_"$replicate_id"/Gong_ADMMLim_norm
done