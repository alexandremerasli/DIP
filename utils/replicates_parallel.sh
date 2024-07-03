# if [ $# != 1 ]
# then
# echo "Please provide replicate number !"
# exit 1
# fi

# arr={2,12,13,19,24,25,34,36}
# nb_replicates=40
# for ((replicate_id=1;replicate_id<=$nb_replicates;replicate_id++)); do
# # for replicate_id in {2,12,13,19,24,25,34,36}; do
#     cpu_available=true
#     for rep_arr in 2 12 13 19 24 25 34 36; do
#         if [ $rep_arr -eq $replicate_id ];
#         then
#             cpu_available=false
#         fi
#     done
#     if $cpu_available;
#     then
#         taskset -c $replicate_id python3 main_nested_BSREM_stand_more_ADMMLim_it.py --replicate $replicate_id &
#         echo $replicate_id
#     fi
# done

nb_replicates=15
for ((replicate_id=1;replicate_id<=$nb_replicates;replicate_id++)); do
    cpu=$((replicate_id))
    #sleep 0.1
    echo $replicate_id
    taskset -c $cpu python3 main_nested_BSREM_stand_CT2skip.py --replicate $replicate_id & # > /dev/null &
    # taskset -c $cpu python3 main_nested_BSREM_stand_CT2skip.py --replicate $replicate_id > aout_$replicate_id.txt & #> /dev/null &
    #taskset -c $cpu python3 test.py &
    
done

# taskset -cpa 41-60 $(mv data/Algo/image40_0/* /disk/workspace_reco/nested_admm/data/Algo/image40_0)

# wait
# shutdown -P +1
