import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

def hrv_plot(data):
    plt.rcParams.update({'font.size': 18, 'text.latex.preamble':r'\usepackage{sfmath} \boldmath'})

    fig, axs = plt.subplots(1, 1, sharex=True, figsize=(24, 6))

    axs.plot(data['hrv_rmssd'], color = "red")
    axs.set_title('HRV - RMSSD')
    axs.set_ylabel('ms', weight = 'bold')
    plt.show()

def radar_plot(loader):
    conditions = loader.protocol_conditions
    df = pd.DataFrame(loader.get_questionnaire_responses('PANAS'))
    categories = df.columns
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    plt.xticks(angles[:-1], categories, color='grey', size=10)
    
    ax.set_rlabel_position(30)
    plt.yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"], color="grey", size=7)
    plt.ylim(0, 5)
    
    for i, row in df.iterrows():
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'{conditions[i]}')
        ax.fill(angles, values, alpha=0.25)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), ncol = 2, fontsize = 10)
    
    plt.title('Subject 3 PANAS Responses')
    plt.show()


def line_plot_stressed(loader):
    conditions = loader.protocol_conditions
    intervals = loader.get_condition_intervals()
    df = pd.DataFrame(loader.get_questionnaire_responses('PANAS'))
    
    stressed_values = df['Stressed']

    plt.figure(figsize=(8, 6))
    sns.lineplot(x=range(len(stressed_values)), y=stressed_values, marker='s', markersize = 15, linewidth=3, linestyle="--")
    plt.title('Subject 3 Stress Levels Over Condition Periods\n')
    plt.xlabel('\nCondition (Time)')
    plt.ylabel('Self-Reported Stress')
    plt.grid(True)
    plt.ylim(0, 5)  
    labels = [f"{conditions[i]}\n({intervals[conditions[i]]['start'].strftime('%H:%M:%S')})" for i in range(len(stressed_values))]
    plt.xticks(range(len(stressed_values)), labels=labels,
               ha = 'center')
    plt.tight_layout()
    plt.show()