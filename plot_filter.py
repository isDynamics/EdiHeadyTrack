import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- Matplotlib style & colors --------------------------------------------
plt.style.use(['tableau-colorblind10', 'ediheadytrack.mplstyle'])
_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

df = pd.read_csv("../../../Data/filter_results.csv")

fig, axs = plt.subplots(1, 3, figsize=(8, 3), sharey=True, sharex=True)
models = df['model'].unique()

for i, model in enumerate(models):
    if i == 0:
        color = _colors[0]
    elif i == 1:
        color = _colors[1]
    else:
        color = "tab:green"
    model_data = df[df['model'] == model]
    axs[i].plot(model_data['sigma'], np.rad2deg(model_data['mean_d']), label='Mean GE', color=color, lw=1)
    axs[i].plot(model_data['sigma'], np.rad2deg(model_data['med_d']), label='Median GE', color=color, linestyle='dashed', lw=1)
    axs[i].plot(model_data['sigma'], np.rad2deg(model_data['mean_d_deltat']), label='Mean IGE', color=color, alpha=0.4, lw=1)
    axs[i].plot(model_data['sigma'], np.rad2deg(model_data['med_d_deltat']), label='Median IGE', color=color, alpha=0.4, linestyle='dashed', lw=1)
    # axs[i].scatter(model_data['sigma'], np.rad2deg(model_data['mean_d']), label='Mean GE', color=color, marker='o')
    # axs[i].scatter(model_data['sigma'], np.rad2deg(model_data['med_d']), label='Median GE', color=color, marker='.')
    # axs[i].scatter(model_data['sigma'], np.rad2deg(model_data['mean_d_deltat']), label='Mean IGE', color=color, marker='s')
    # axs[i].scatter(model_data['sigma'], np.rad2deg(model_data['med_d_deltat']), label='Median IGE', color=color, marker='s')

    axs[i].set_title(model)
    axs[i].set_xlim(0, 5)
    axs[i].set_ylim(0, 17)
    axs[i].legend()

axs[1].set_xlabel(r'Gaussian filter standard deviation, $\sigma$')
axs[0].set_ylabel('Error (deg)')
plt.tight_layout()
plt.savefig("../FigureDev/outputs/filter_sensitivity.pdf")
plt.show()