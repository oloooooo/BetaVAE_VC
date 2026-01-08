import os
import glob2
import shutil
ori_mel_list = glob2.glob("/root/BetaVAE_VC/betaVC/*.npy")
mel_list =  [os.path.basename(i).split(".")[0] for i in ori_mel_list]
print(mel_list[50:])
with open("openvoicev2.txt","r") as f:
    for line in f:
        if "unseen" in line and "P" in line.split("_to_")[-1]:
            
            line = os.path.basename(line.strip()).split(".")[0]
            i = mel_list.index(line) # mel_list更多
            shutil.copy2(ori_mel_list[i],"/root/autodl-tmp/betaVC")