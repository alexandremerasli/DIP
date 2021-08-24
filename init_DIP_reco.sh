#!/bin/bash

###############################################################################################
##	Script pour simuler et ranger data pour le DIP
################################################################################################


dim1=128
dim2=128
dim3=1

if [ $1 = 'biograph' ]
then
	
###############################################################################################
##	Biograph simulation
################################################################################################

mkdir -p simu_biograph
cd simu_biograph
export PATH=$PATH:/home/meraslia/sgld/simulation_pack/bin

# Create activity phantom
create_phantom.exe -o phantom_act -d $dim1 $dim2 $dim3 -v 4. 4. 4. -c 0. 0. 0. 150. 4. 100 -c 50. 10. 0. 20. 4. 400 -c -40. -40. 0. 40. 4. 0. -x 32 32 1

# Create attenuation map
create_phantom.exe -o phantom_atn -d $dim1 $dim2 $dim3 -v 4. 4. 4. -c 0. 0. 0. 150. 4. 0.096 -c -40. 40. 0. 25. 4. 0.02 -x 32 32 1

CMmaker.exe -m biograph2D -u -o biograph

# Simulation with sinograms
simulator.exe -m biograph2D -c biograph/biograph.ecm -i phantom_act.hdr -a phantom_atn.hdr -r 0.9 -s 0.3 -p 4 -P 50000000 -D -v 2 -o simulation1 -T 4

# Convert in list mode
make_castor_datafile.exe -m biograph -p simulation1/simulation1_pt.s.hdr -r simulation1/simulation1_rd.s.hdr -s simulation1/simulation1_sc.s.hdr -n simulation1/simulation1_nm.s.hdr -A simulation1/simulation1_at.s -v 2 -o data_eff10 -c biograph/biograph.ecm

# MLEM short reconstruction with CASToR
it=1
castor-recon -df data_eff10/data_eff10.cdh -dout castor_output -dim $dim1,$dim2,$dim3 -vox 4,4,4 -vb 3 -it $it:1 -proj incrementalSiddon -opti MLEM -th 0 -osens -oit -1

else

###############################################################################################
##	MMR simulation
################################################################################################

mkdir -p simu_mmr
cd simu_mmr
export PATH=$PATH:/home/meraslia/sgld/simulator_mmr_2d/bin

## Etape 0: Creation du fantôme (activite) et de la carte d'attenuation

create_phantom.exe -o phantom_act -d $dim1 $dim2 $dim3 -v 4. 4. 4. -c 0. 0. 0. 150. 4. 100 -c 50. 10. 0. 20. 4. 400 -c -40. -40. 0. 40. 4. 0. -x 32 32 1
create_phantom.exe -o phantom_atn -d $dim1 $dim2 $dim3 -v 4. 4. 4. -c 0. 0. 0. 150. 4. 0.096 -c -40. 40. 0. 25. 4. 0.02 -x 32 32 1

## Etape 1: Creation d'une carte des coordonnees des cristaux et de leur efficacite respective.
##          Tu peux le refaire pour chaque simu pour eviter que ce soit toujours pareil, comme tu veux.

# Cela regle la variation d'efficacite aleatoire maximale en pourcent de chaque cristal (autour de 1).
# Si tu mets 10, cela tire au hasard une efficacite entre 0.9 et 1.1.
# 10 est une valeur plus ou moins realiste.
eff=10
CMmaker.exe -m mmr2d -r ${eff} -o cmap_eff${eff} -w -v 2

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
SMprojector.exe -m mmr2d -c cmap_eff${eff}/cmap_eff${eff}.ecm -i phantom_act.hdr -a phantom_atn.hdr -s 0.35 -r 0.9 -l 0.8 -p 4. -v 5 -P 100000000 -o simu_eff${eff} -D

## Etape 3: Creation du ficher castor a partir des sinogrammes simules

# En gros tu redonnes tous les sinogrammes simules en entree. Il faut donner les header, sauf pour
# l'attenuation -A ou il faut donner directement le sinogramme. N'oublie pas l'option -castor.
SMmaker.exe -m mmr2d -o data_eff${eff} -p simu_eff${eff}/simu_eff${eff}_pt.s.hdr -r simu_eff${eff}/simu_eff${eff}_rd.s.hdr -s simu_eff${eff}/simu_eff${eff}_sc.s.hdr -n simu_eff${eff}/simu_eff${eff}_nm.s.hdr -A simu_eff${eff}/simu_eff${eff}_at.s -c cmap_eff${eff}/cmap_eff${eff}.ecm -castor -v 2


# MLEM short reconstruction with CASToR
it=6
castor-recon -df data_eff10/data_eff10.cdh -dout castor_output -dim $dim1,$dim2,$dim3 -vox 4,4,4 -vb 3 -it $it:1 -proj incrementalSiddon -opti MLEM -th 0 -osens -oit -1

fi

###############################################################################################
##	Copy to DIP used directories
################################################################################################
mkdir -p ../data/Algo/Block2/data
mkdir -p ../data/Algo/Data/data_eff10

cp phantom_* /home/meraslia/sgld/hernan_folder/data/Algo/Block2/data/
cp data_eff10/* /home/meraslia/sgld/hernan_folder/data/Algo/Data/data_eff10
cp castor_output/* /home/meraslia/sgld/hernan_folder/data/Algo/Data/