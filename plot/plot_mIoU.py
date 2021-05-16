import json
import pandas as pd
import matplotlib.pyplot as plt

# Loss, Precision, Recall, mIoU
with open('generalization_scores.json') as json_file:
    data1 = json.load(json_file)
score_means1 = data1['score_means'] # precision, recall, mean_jaccard, weighted_mean_jaccard, score_means

with open('generalization_scores_smooth.json') as json_file:
    data2 = json.load(json_file)
score_means2 = data2['score_means'] # precision, recall, mean_jaccard, weighted_mean_jaccard, score_means


# Groups
df = pd.DataFrame()
grp1 = 'Before postprocessing'
grp2 = 'After postprocessing'

# Labels
df['Class'] = ["All weighted","All","Building","Clutter","Vegetation","Water","Ground","Car"]

# Mean values and std's
df[grp1] = [score_means1["wmj_mean"],score_means1["mj_mean"],score_means1["building_jc_mean"],score_means1["clutter_jc_mean"],score_means1["vegetation_jc_mean"],score_means1["water_jc_mean"],score_means1["ground_jc_mean"],score_means1["car_jc_mean"]]

df[grp1+'_std'] = [score_means1["wmj_std"],score_means1["mj_std"],score_means1["building_jc_std"],score_means1["clutter_jc_std"],score_means1["vegetation_jc_std"],score_means1["water_jc_std"],score_means1["ground_jc_std"],score_means1["car_jc_std"]]

df[grp2] = [score_means2["wmj_mean"],score_means2["mj_mean"],score_means2["building_jc_mean"],score_means2["clutter_jc_mean"],score_means2["vegetation_jc_mean"],score_means2["water_jc_mean"],score_means2["ground_jc_mean"],score_means2["car_jc_mean"]]

df[grp2+'_std'] = [score_means2["wmj_std"],score_means2["mj_std"],score_means2["building_jc_std"],score_means2["clutter_jc_std"],score_means2["vegetation_jc_std"],score_means2["water_jc_std"],score_means2["ground_jc_std"],score_means2["car_jc_std"]]

# Printing LaTeX table syntax
print(" & " + str(round(score_means1["pr_mean"], 3)) + " & " + str(round(score_means1["re_mean"], 3)) + " & " + str(round(score_means1["wmj_mean"], 3)) + " & " + str(round(score_means1["mj_mean"], 3)) + " & " + str(round(score_means1["building_jc_mean"], 3)) + " & " + str(round(score_means1["clutter_jc_mean"], 3)) + " & " + str(round(score_means1["vegetation_jc_mean"], 3)) + " & " + str(round(score_means1["water_jc_mean"], 3)) + " & " + str(round(score_means1["ground_jc_mean"], 3)) + " & " + str(round(score_means1["car_jc_mean"], 3)) + " \\\\")

print(" & " + str(round(score_means2["pr_mean"], 3)) + " & " + str(round(score_means2["re_mean"], 3)) + " & " + str(round(score_means2["wmj_mean"], 3)) + " & " + str(round(score_means2["mj_mean"], 3)) + " & " + str(round(score_means2["building_jc_mean"], 3)) + " & " + str(round(score_means2["clutter_jc_mean"], 3)) + " & " + str(round(score_means2["vegetation_jc_mean"], 3)) + " & " + str(round(score_means2["water_jc_mean"], 3)) + " & " + str(round(score_means2["ground_jc_mean"], 3)) + " & " + str(round(score_means2["car_jc_mean"], 3)) + " \\\\")

# Plotting
ax = df.plot.bar(x='Class', 
                y=[grp1,grp2],
                yerr=df[[grp1+"_std",grp2+"_std"]].T.values,
                color=['cornflowerblue', 'mediumblue'],
                #color=['orange', 'orangered'],
                figsize=(8, 6))
plt.ylim(0, 1)
plt.suptitle("U-Net: mIoU per class", fontsize=18)
plt.ylabel('mIoU', fontsize=14)
plt.xlabel('Class', fontsize=14)
plt.tight_layout()
plt.savefig('plotted_mIoU_per_class.png')
#plt.show()
#plt.savefig(f"{self.basedir}/plotted_mIOU.png")