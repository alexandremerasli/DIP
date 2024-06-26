#!/bin/bash

###############################################################################################
##	Script pour simuler données MMR ou biograph et générer le fichier d'entrée .cdh de CASToR
################################################################################################

if [ $# != 1 ]
then
echo "Please provide scanner name !"
exit 1
fi

dim1=112
dim2=112
dim3=1 # Must be one in this script, otherwise use let to do mathematics computation for voxel size

#nb_counts=100000000 # high statistics
nb_counts=1500000 # low statistics

# Choosing directory to work on
if [[ $1 = 'biograph' ]]
then
mkdir -p simu_biograph
cd simu_biograph
else
mkdir -p simu_mmr
cd simu_mmr
fi

## Etape 0: Creation du fantôme (activite) et de la carte d'attenuation
# Create activity phantom
create_phantom.exe -o image4_1 -d $dim1 $dim2 $dim3 -v 4. 4. 4. -c 0. 0. 0. 150. 4. 100 -c 50. 10. 0. 20. 4. 400 -c -40. -40. 0. 40. 4. 10. -c 50. 90. 0. 20. 4. 400 -x 32 32 1

# Create attenuation map
create_phantom.exe -o image4_1_atn -d $dim1 $dim2 $dim3 -v 4. 4. 4. -c 0. 0. 0. 150. 4. 0.096 -x 32 32 1

# Create DIP input map (idea : similar to MR)
create_phantom.exe -o image4_1_mr -d $dim1 $dim2 $dim3 -v 4. 4. 4. -c 0. 0. 0. 150. 4. 30 -c -40. -40. 0. 40. 4. 60. -c 50. 90. 0. 20. 4. 60 -x 32 32 1

nb_replicates=100
for ((replicate_id=1;replicate_id<=nb_replicates;replicate_id++)); do
    echo "replicate_id"$replicate_id
    if [[ $1 = 'biograph' ]]
    then
    
    ###############################################################################################
    ##	Biograph simulation
    ################################################################################################

    CMmaker.exe -m biograph2D -u -o biograph

    # Simulation with sinograms
    simulator.exe -m biograph2D -c biograph/biograph.ecm -i image4_1.hdr -a image4_1_atn.hdr -r 0.9 -s 0.3 -p 4 -P $nb_counts -D -v 2 -o simulation1 -T 4

    # Convert in list mode
    make_castor_datafile.exe -m biograph -p simulation1/simulation1_pt.s.hdr -r simulation1/simulation1_rd.s.hdr -s simulation1/simulation1_sc.s.hdr -n simulation1/simulation1_nm.s.hdr -A simulation1/simulation1_at.s -v 2 -o data_eff10 -c biograph/biograph.ecm

    else

    ###############################################################################################
    ##	MMR simulation
    ################################################################################################

    ## Etape 1: Creation d'une carte des coordonnees des cristaux et de leur efficacite respective.
    ##          Tu peux le refaire pour chaque simu pour eviter que ce soit toujours pareil, comme tu veux.

    # Cela regle la variation d'efficacite aleatoire maximale en pourcent de chaque cristal (autour de 1).
    # Si tu mets 10, cela tire au hasard une efficacite entre 0.9 et 1.1.
    # 10 est une valeur plus ou moins realiste.
    eff=10
    CMmaker.exe -m mmr2d -r ${eff} -o cmap0 -w -v 2

    ## Etape 2: Simulation des sinogrammes a partir des images d'emission et d'attenuation (en cm-1).

    # -r regle la fraction de randoms en rapport au total des prompts
    # -l regle la fraction de randoms lies au background lso en rapport au total des randoms
    # -s regle la fraction de diffuses en rapport aux net-trues (prompts-randoms)
    # -p regle la psf dans l'image, demande a Thomas combien elle vaut pour le MMR (ici c'est la fwhm en mm d'une gausienne isotrope et invariante)
    # -D utilse une implementation du projecteur siddon qui est la meme que dans castor, important de garder cette option
    # -c le fichier "crystal map" cree dans la premiere etape
    # -m mmr2d, parametre fixe
    # -i l'image d'emission en entree
    # -a la mumap en cm-1, doit etre de la meme taille que l'image d'emission
    # -P le nombre de prompts a simuler
    SMprojector.exe -m mmr2d -c cmap0/cmap0.ecm -i image4_1.hdr -a image4_1_atn.hdr -s 0.35 -r 0.9 -l 0.8 -p 4. -v 5 -P $nb_counts -o simu0_${replicate_id} -D

    ## Etape 3: Creation du ficher castor a partir des sinogrammes simules

    # En gros tu redonnes tous les sinogrammes simules en entree. Il faut donner les header, sauf pour
    # l'attenuation -A ou il faut donner directement le sinogramme. N'oublie pas l'option -castor.
    SMmaker.exe -m mmr2d -o data4_1_${replicate_id} -p simu0_${replicate_id}/simu0_${replicate_id}_pt.s.hdr -r simu0_${replicate_id}/simu0_${replicate_id}_rd.s.hdr -s simu0_${replicate_id}/simu0_${replicate_id}_sc.s.hdr -n simu0_${replicate_id}/simu0_${replicate_id}_nm.s.hdr -A simu0_${replicate_id}/simu0_${replicate_id}_at.s -c cmap0/cmap0.ecm -castor -v 2

    fi

    ###############################################################################################
    ##	Copy to DIP used directories
    ################################################################################################
    mkdir -p ../data/Algo/Data/database_v2/image4_1

    # Copying previously computed masks 
    #cp -nr ../data/Algo/Data/database_v2/image4_1_1replicate/* ../data/Algo/Data/database_v2/image4_1

    # Copying phantoms
    #cp image4_1* ../data/Algo/Data/database_v2/image4_1
    cp image4_1.img ../data/Algo/Data/database_v2/image4_1/image4_1.raw
    cp image4_1.img ../data/Algo/Data/database_v2/image4_1/image4_1.img
    cp image4_1.hdr ../data/Algo/Data/database_v2/image4_1/image4_1.hdr
    cp image4_1_atn.img ../data/Algo/Data/database_v2/image4_1/image4_1_atn.raw
    cp image4_1_atn.img ../data/Algo/Data/database_v2/image4_1/image4_1_atn.img
    cp image4_1_atn.hdr ../data/Algo/Data/database_v2/image4_1/image4_1_atn.hdr
    cp image4_1_mr.img ../data/Algo/Data/database_v2/image4_1/image4_1_mr.raw
    cp image4_1_mr.img ../data/Algo/Data/database_v2/image4_1/image4_1_mr.img
    cp image4_1_mr.hdr ../data/Algo/Data/database_v2/image4_1/image4_1_mr.hdr
    # Copying datafile, cmap and sinograms
    cp -r data4_1_${replicate_id}/ ../data/Algo/Data/database_v2/image4_1
    cp -r cmap0/ ../data/Algo/Data/database_v2/image4_1
    cp -r simu0_${replicate_id}/ ../data/Algo/Data/database_v2/image4_1
done