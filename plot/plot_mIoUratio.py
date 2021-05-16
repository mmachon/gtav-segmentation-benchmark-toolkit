import json
import pandas as pd
import matplotlib.pyplot as plt

# Loss, Precision, Recall, mIoU
with open('scores.json') as json_file:
    data1 = json.load(json_file)

with open('scores.json') as json_file:
    data2 = json.load(json_file)

with open('generalization_scores_c2.json') as json_file:
    data3 = json.load(json_file)
score_means1 = data3['score_means'] # precision, recall, mean_jaccard, weighted_mean_jaccard, score_means

with open('generalization_scores_dd.json') as json_file:
    data4 = json.load(json_file)
score_means2 = data4['score_means'] # precision, recall, mean_jaccard, weighted_mean_jaccard, score_means

# Groups
df = pd.DataFrame()
grp1 = 'C2Land'
grp2 = 'Drone Deploy'

# Labels
df['Class'] = ["All weighted","All","Building","Clutter","Vegetation","Water","Ground","Car"]

# Mean values and std's
df[grp1] = [score_means1["wmj_mean"] / data1["fwm_iou"],score_means1["mj_mean"] / data1["m_iou:"],score_means1["building_jc_mean"] / data1["iou"][0],score_means1["clutter_jc_mean"] / data1["iou"][1],score_means1["vegetation_jc_mean"] / data1["iou"][2],score_means1["water_jc_mean"] / data1["iou"][3],score_means1["ground_jc_mean"] / data1["iou"][4],score_means1["car_jc_mean"] / data1["iou"][5]]

df[grp1+'_std'] = [score_means1["wmj_std"],score_means1["mj_std"],score_means1["building_jc_std"],score_means1["clutter_jc_std"],score_means1["vegetation_jc_std"],score_means1["water_jc_std"],score_means1["ground_jc_std"],score_means1["car_jc_std"]]

df[grp2] = [score_means2["wmj_mean"] / data2["fwm_iou"],score_means2["mj_mean"] / data2["m_iou:"],score_means2["building_jc_mean"] / data2["iou"][0],score_means2["clutter_jc_mean"] / data2["iou"][1],score_means2["vegetation_jc_mean"] / data2["iou"][2],score_means2["water_jc_mean"] / data2["iou"][3],score_means2["ground_jc_mean"] / data2["iou"][4],score_means2["car_jc_mean"] / data2["iou"][5]]

df[grp2+'_std'] = [score_means2["wmj_std"],score_means2["mj_std"],score_means2["building_jc_std"],score_means2["clutter_jc_std"],score_means2["vegetation_jc_std"],score_means2["water_jc_std"],score_means2["ground_jc_std"],score_means2["car_jc_std"]]

# Plotting
ax = df.plot.bar(x='Class', 
                y=[grp1,grp2],
                #yerr=df[[grp1+"_std",grp2+"_std"]].T.values,
                #color=['cornflowerblue', 'mediumblue'],
                color=['orange', 'orangered'],
                figsize=(8, 6))
plt.ylim(0, 1)
plt.suptitle("mIoU_ratio per class", fontsize=18)
plt.ylabel('mIoU_ratio', fontsize=14)
plt.xlabel('Class', fontsize=14)
plt.tight_layout()
plt.savefig('plotted_mIoUratio_per_class.png')
#plt.show()
#plt.savefig(f"{self.basedir}/plotted_mIOU.png")