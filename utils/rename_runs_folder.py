import os

# Put this outside "utils/" folder to run it
specific_run = "do_everything_2023-05-17_14-42-55"
runs_dir = os.path.join(os.getcwd(),"runs",specific_run)
i = 0
rho_list = [0,10,20,30,40,50,60,70,80,90,100,110,120,130,300,500,1000]
for folder in sorted(os.listdir(runs_dir)):
    if not os.path.isdir(os.path.join(runs_dir,folder)):
        continue # Not a directory
    rho = rho_list[i]
    if rho <= 60:
        os.rename(os.path.join(runs_dir,folder),os.path.join(runs_dir,"rho = " + str(rho)))
    else:
        os.rename(os.path.join(runs_dir,folder),os.path.join(runs_dir,"bigrho = " + str(rho)))
    i += 1

    # print(folder)