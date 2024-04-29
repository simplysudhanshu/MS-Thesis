import sys
import os
import json
import copy
import math
import pickle
import traceback

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import supermarq_metrics

# Experiments dict skeleton
experiments = []
experiment_dict = {
    "name": None,
    "size": [],
    "shots": [],
    "runtimes":{
        "Encoder": [],
        "Invert + Measurement": [],
        "Transpile": [],
        "Simulate": [],
        "Decoder": [],
        "Algorithm Runtime": [],
        "Noisy Encoder": [],
        "Noisy Invert + Measurement": [],
        "Noisy Transpile": [],
        "Noisy Simulate": [],
        "Noisy Decoder": [],
        "Noisy Algorithm Runtime": []
    },
    "depths": {
        "Encoder": [],
        "Invert + Measurement": [],
        "Transpile": [],
        "Simulate": []
    },
    "widths": [],
    "accuracy": [],
    "noisy_accuracy": [],
    "supermarq_metrics": [],
    "count_ops": [],
    "data_points": [],
    "noisy_data_points": []
}

shots_dict = { 
    "shots": [], 
    "runtimes": [],
    "accuracy": [] 
    }

backend_comparison_dict = {
    "name": "FRQI",
    "backend": ["StateVec", "Pure Sim", "Noisy Sim", "IBMQ_Manilla"],
    "size": [4, 16, 64, 256],
    "runtimes": [[5, 10, 15, 20],
                [27, 28, 29, 36],
                [27, 29, 32, 38],
                [25, 28, 33, 43]],
    "accuracy": [[100, 100, 100, 100],
                [35, 33, 33, 30],
                [33, 32, 32, 30],
                [30, 28, 28, 29]]
}

# diffs and their names
global_p_diffs = {
    "Qubit Lattice": [[], []],
    "Phase": [[], []],
    "FRQI": [[], []] 
}

global_n_diffs = {
    "Qubit Lattice": [[], []],
    "Phase": [[], []],
    "FRQI": [[], []] 
}

# setup directory for storing images
os.makedirs("./experiment_data_vis", exist_ok=True)
# color_map = ['tab:blue','tab:cyan','tab:green','yellow', 'tab:orange', 'tab:red']

'''
Utils
'''
#__________________________________
def get_dict(dict_type="exp"):
    if dict_type == "exp":
        return copy.deepcopy(experiment_dict)
    
    elif dict_type == "shots":
        return copy.deepcopy(shots_dict)
    
    elif dict_type == "backend":
        return copy.deepcopy(backend_comparison_dict)


#__________________________________
# highlight a cell in imshow
def highlight_cell(x,y, ax=None, **kwargs):
    rect = plt.Rectangle((x-.5, y-.5), 1, 1, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect 

#__________________________________
def parse_log(filename):
    try:
        logs = open(filename, "r")
    except:
        print(f"Log file {filename} not found.\tTerminating.")
        exit()

    logs = logs.readlines()

    current_experiment = {}

    # parse logs and fill up experiments dictionaries
    for line in logs:
        if "INFO" in line:
            profiler = json.loads(line[line.index("= ")+1:])
            # print(profiler)

            if profiler['Profiler'] == 'Encoder':            
                name, size, shots = profiler['Exp'].split(",")

                if 'name' in current_experiment and name != current_experiment['name']: 
                    experiments.append(current_experiment)
                    current_experiment = copy.deepcopy(experiment_dict)
                elif not current_experiment:
                    current_experiment = copy.deepcopy(experiment_dict)

                current_experiment["name"] = name
                current_experiment["size"].append(float(size))
                current_experiment["shots"].append(float(shots))
                current_experiment["widths"].append(float(profiler["width"]))
                current_experiment["runtimes"]["Encoder"].append(float(profiler["runtime"]))
                current_experiment["depths"]["Encoder"].append(float(profiler["depth"]))

            elif profiler['Profiler'] == "Invert + Measurement":
                current_experiment["runtimes"]["Invert + Measurement"].append(float(profiler["runtime"]))
                current_experiment["depths"]["Invert + Measurement"].append(float(profiler["depth"]))

            elif profiler['Profiler'] == "Transpile":
                current_experiment["runtimes"]["Transpile"].append(float(profiler["runtime"]))
                current_experiment["depths"]["Transpile"].append(float(profiler["depth"]))

            elif profiler['Profiler'] == "Simulate":
                current_experiment["runtimes"]["Simulate"].append(float(profiler["runtime"]))
                current_experiment["depths"]["Simulate"].append(float(profiler["depth"]))
            
            elif profiler['Profiler'] == "Decoder":
                current_experiment["runtimes"]["Decoder"].append(float(profiler["runtime"]))

            elif profiler['Profiler'] == "Accuracy":
                current_experiment["accuracy"].append(float(profiler["value"]))
            
            elif profiler['Profiler'] == "Algorithm Runtime":
                current_experiment["runtimes"]['Algorithm Runtime'].append(float(profiler["runtime"]))
            
            elif profiler['Profiler'] == "Data Points":
                current_experiment["data_points"].append([profiler['original_values'], profiler['reconstructed_values']])
            
    return current_experiment


'''
Runtime bars
'''
#__________________________________
def plot_runtimes(exp):
    print("\033[K", f"Plotting Runtimes", end='\r')

    sizeLables = [str(x) for x in exp['size']]
    sizeValues = np.arange(len(sizeLables))

    # colors = plt.colormaps['RdYlGn'](np.linspace(0.15, 0.85, 3))
    
    # encode_perc = [x*100/y for x,y in zip(exp['runtimes']['Encoder'], exp['runtimes']['Algorithm Runtime'])]

    # simulate_sum = [sum(x) for x in zip(exp['runtimes']['Invert + Measurement'], exp['runtimes']['Transpile'], exp['runtimes']['Simulate'])]
    # simulate_perc = [x*100/y for x,y in zip(simulate_sum, exp['runtimes']['Algorithm Runtime'])]
    
    # decode_perc = [x*100/y for x,y in zip(exp['runtimes']['Decoder'], exp['runtimes']['Algorithm Runtime'])]

    fig, ax = plt.subplots()

    b = ax.bar(sizeValues, exp['runtimes']['Encoder'], color="dodgerblue", zorder=3)
    ax.bar_label(b, ["%.2f" % x for x in exp['runtimes']['Encoder']], color="k")

    ax.set_xticks(sizeValues)
    ax.set_xticklabels(sizeLables)
    ax.set_xlabel("Problem Size")

    ax.set_ylim([0, 1.1*max(exp['runtimes']['Encoder'])])
    ax.set_ylabel("Runtime (s)")

    ax.grid(axis='y', alpha=0.5, linestyle="dotted", zorder=0)

    fig.tight_layout()
    
    plt.savefig(os.path.join("experiment_data_vis", f"{exp['name']}_encoding_runtimes"))
    plt.close()


    #-------
    fig, ax = plt.subplots()

    b = ax.bar(sizeValues, exp['runtimes']['Algorithm Runtime'], color="deeppink", zorder=3)
    ax.bar_label(b, ["%.2f" % x for x in exp['runtimes']['Algorithm Runtime']], color="k")

    ax.set_xticks(sizeValues)
    ax.set_xticklabels(sizeLables)
    ax.set_xlabel("Problem Size")
    
    ax.set_ylabel("Runtime (s)")
    ax.set_ylim([0.9*min(exp['runtimes']['Algorithm Runtime']), 1.1*max(exp['runtimes']['Algorithm Runtime'])])
    # ax.set_yticks(np.arange(20, 41, 2.5))
    # ax.set_yticklabels([20,'',25,'',30,'',35,'',40])

    ax.grid(axis='y', alpha=0.5, linestyle="dotted", zorder=0)

    fig.tight_layout()
    
    plt.savefig(os.path.join("experiment_data_vis", f"{exp['name']}_algorithm_runtimes"))
    plt.close()

'''
Circuit Barh
'''
#__________________________________
def plot_circuit_stats(exp):
    print("\033[K", f"Plotting Circuits", end='\r')

    xLables = [str(x) for x in exp['size']]
    xValues = np.arange(len(xLables))

    fig, (ax_width, ax_depth) = plt.subplots(1, 2)

    #-------
    width_bar = ax_width.barh(xValues, exp['widths'], height=0.4, label="Width", color="blueviolet", zorder=3)
    ax_width.bar_label(width_bar, exp['widths'], fontsize=9, padding=-12, color="k")
    
    ax_width.invert_xaxis()
    ax_width.set_xlim([max(exp['widths'])+3, 0])
    
    ax_width.tick_params('y', labelleft=False)
    ax_width.yaxis.tick_right()
    ax_width.set_ylabel("Problem Size")
    
    ax_width.legend(loc="lower left")
    ax_width.grid(axis='y', alpha=0.5, linestyle="dotted", zorder=0)

    #-------
    depth_bar = ax_depth.barh(xValues, exp['depths']['Invert + Measurement'], height=0.4, label="Depth", color="springgreen", zorder=3)
    ax_depth.bar_label(depth_bar, exp['depths']['Invert + Measurement'], fontsize=9, padding=3, color="k")
    
    xMaxLimPow = math.ceil(math.log(max(exp['depths']['Invert + Measurement']), 10))

    ax_depth.set_xscale('log')
    # ax_depth.set_xlim([1, 10**xMaxLimPow])
    ax_depth.set_xticks([10**x for x in range(xMaxLimPow+1)])
    ax_depth.set_xticklabels([0] + [f"$10^{x}$" for x in range(1, xMaxLimPow+1)])
    
    ax_depth.set_yticks(xValues)
    ax_depth.set_yticklabels(xLables)

    ax_depth.legend(loc="lower right")
    ax_depth.grid(axis='x', alpha=0.5, linestyle="dotted", zorder=0)

    #-------
    # fig.suptitle(exp['name'] + " Experiment: Circuits")
    
    plt.subplots_adjust(wspace=0.2, bottom=0.1, top=0.95, left=0.075, right=0.95)    
    # plt.tight_layout()
        
    plt.savefig(os.path.join("experiment_data_vis", f"{exp['name']}_depths"))
    plt.close()

'''
Data imshows
'''
#__________________________________
def plot_data(exp):        
    for i in range(len(exp['data_points'])):
        print("\033[K", f"Plotting Data {i+1}/{len(exp['data_points'])}", end='\r')

        input_data, output_pure = exp['data_points'][i]
        output_noisy = exp['noisy_data_points'][i][1]
        output_expected = [255-x for x in input_data]

        # dimensions of the image
        side = int(math.sqrt(len(input_data)))        

        # Differences
        p_diffs = np.reshape([abs(255-x-y) for x,y in zip(input_data, output_pure)], (side, side))
        n_diffs = np.reshape([abs(255-x-y) for x,y in zip(input_data, output_noisy)], (side, side))

        global_p_diffs[exp['name']][0].append(p_diffs.flatten())
        global_p_diffs[exp['name']][1].append(f"{side}x{side}")

        global_n_diffs[exp['name']][0].append(n_diffs.flatten())
        global_n_diffs[exp['name']][1].append(f"{side}x{side}")
        
        # for axis limits
        global_min_diff, global_max_diff = min(np.min(p_diffs), np.min(n_diffs)), max(np.max(p_diffs), np.max(n_diffs))

        data_plots = {
            'input': input_data,
            'expected': output_expected,
            'pure': output_pure,
            'noisy': output_noisy,
            'pure_err': p_diffs,
            'noisy_err': n_diffs
                   }

        # For Pure and Noisy: input-expected-output-error distribution
        for plot, data in data_plots.items():
            print("\033[K", f"Plotting Data {i+1}: {plot}", end='\r')

            fig, ax = plt.subplots()

            if "err" in plot:
                im = ax.imshow(data, vmin=global_min_diff, vmax=global_max_diff, cmap="OrRd")

                # if np.max(data) > 0:
                #     max_diffs = list(zip(*np.where(data == np.max(data))))
                #     for index in max_diffs:
                #         highlight_cell(index[1], index[0], ax=ax, color="k", linewidth=1)

                # annotate
                for x in range(len(data)):
                    for y in range(len(data[x])):
                        if (side < 16 or (side >= 16 and data[x, y] == np.max(data))) and data[x, y] != 0:
                            text = ax.text(y, x, data[x, y], ha="center", va="center", color="w")

                title_string = f"{'Pure' if 'pure' in plot else 'Noisy'} Simulator: Absolute Diff\n~{round(np.count_nonzero(data==0)*100/len(input_data), 2)}% accuracy"
            
            else:
                im = ax.imshow(np.reshape(data, (side, side)), cmap='gray')
                
                # print("\t\t\t\t", data[-4:])

                if plot == "input":
                    title_string = "Input data"
                elif plot == "expected":
                    title_string = "Expected output (inverted pixels)"
                elif plot == "pure":
                    title_string = "Pure Simulator output"
                elif plot == "noisy":
                    title_string = "Noisy Simulator output"

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("bottom", size="5%", pad=0.05)
            cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
            cbar.ax.tick_params(labelsize=6)
            # # cbar.set_ticks(range(0, 255, 50))
            # # cbar.set_ticklabels(range(0, 255, 50))

            #-------
            # Major ticks
            ax.set_xticks(np.arange(0, side))
            ax.set_yticks(np.arange(0, side))

            # Labels for major ticks
            ax.set_xticklabels(np.arange(1, side+1))
            ax.set_yticklabels(np.arange(1, side+1))

            # Minor ticks
            ax.set_xticks(np.arange(-.5, side), minor=True)
            ax.set_yticks(np.arange(-.5, side), minor=True)

            # Gridlines based on minor ticks
            ax.grid(which='minor', color='w', linestyle='-', linewidth=1)

            # Remove minor ticks
            ax.tick_params(which='minor', bottom=False, left=False, labelsize=8)
            ax.tick_params(which='major', bottom=False, top=True, labeltop=True, labelsize=8)

            # fig.colorbar(im, orientation="horizontal", pad=0.2)

            # plt.suptitle(exp['name'] + f" Analysis - {side} x {side} image ({exp['widths'][i]} qubits)")
            # plt.title(title_string)
    
            fig.tight_layout()
            fig.subplots_adjust()

            plt.savefig(os.path.join("experiment_data_vis", f"{exp['name']}_{plot}_{side}x{side}"), bbox_inches='tight')
            plt.close()

    # plot error violins
    plot_errors(exp=exp)
    #-------
    #
    #  diff = diff.flatten()
    # ax_error.bar(range(side*side), diff, color=np.where(diff == np.max(diff), "r", "royalblue"))
    # ax_error.axes.get_yaxis().set_ticks([1, 2, 3, 4])
    # ax_error.axhline(np.mean(diff), color="orange", linestyle=":", label=f"Avg: {round(np.mean(diff), 2)}")
    # ax_error.legend(loc="upper right")
    # ax_error.set_ylim([0, global_max_diff+.1])
    # ax_error.set_title(f"MaxErr: {np.max(diff)} ({round(np.max(diff)*100/255, 2)}%), AvgErr: {round(np.mean(diff), 2)} ({round(np.mean(diff)*100/255, 2)}%)")

'''
Error violins
'''
#__________________________________
def plot_errors(exp):
    print("\033[K", f"Plotting Errors", end='\r')
    
    if not global_p_diffs[exp['name']][0]: return
    bbox = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)

    fig, ax = plt.subplots()

    ax.violinplot(global_p_diffs[exp['name']][0], showmeans=True)
    ax.plot(range(1, len(global_p_diffs[exp['name']][0])+1), [np.mean(x) for x in global_p_diffs[exp['name']][0]], marker=".", color="r", label="Avg Error", linestyle="dashed", alpha=0.5)
    
    for x,y in zip(range(1, len(global_p_diffs[exp['name']][0])+1), [np.mean(x) for x in global_p_diffs[exp['name']][0]]):
        ax.annotate("%.2f" % y, xy=(x+0.15,y-0.2), color="r", bbox=bbox)
    
    ax.axes.get_xaxis().set_ticks(range(1, len(global_n_diffs[exp['name']][0])+1), global_p_diffs[exp['name']][1])
    ax.set_xlabel('Input size')

    ticks_list = range(0, np.max([np.max(x) for x in global_p_diffs[exp['name']][0]]), 5)
    ax.axes.get_yaxis().set_ticks(ticks_list, ticks_list)
    ax.set_ylabel('Absolute error values')
    
    ax.grid(axis='y', which='both', alpha=0.5, linestyle="dotted", clip_on=False)
    
    plt.legend(loc="upper left")
    
    plt.savefig(os.path.join("experiment_data_vis", f"{exp['name']}_pure_error_violins"), bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots()

    #-------
    ax.violinplot(global_n_diffs[exp['name']][0], showmeans=True)
    ax.plot(range(1, len(global_n_diffs[exp['name']][0])+1), [np.mean(x) for x in global_n_diffs[exp['name']][0]], marker=".", color="r", label="Avg Error", linestyle="dashed", alpha=0.5)

    for x,y in zip(range(1, len(global_n_diffs[exp['name']][0])+1), [np.mean(x) for x in global_n_diffs[exp['name']][0]]):
        ax.annotate("%.2f" % y, xy=(x+0.1,y-0.2), color="r", bbox=bbox)

    ax.axes.get_xaxis().set_ticks(range(1, len(global_n_diffs[exp['name']][0])+1), global_n_diffs[exp['name']][1])
    ax.set_xlabel('Input size')

    ticks_list = range(0, np.max([np.max(x) for x in global_p_diffs[exp['name']][0]]), 5)
    ax.axes.get_yaxis().set_ticks(ticks_list, ticks_list)
    ax.set_ylabel('Absolute error values')
    ax.grid(axis='y', alpha=0.5, linestyle="dotted", zorder=0)
    
    plt.legend(loc="upper left")
    
    plt.savefig(os.path.join("experiment_data_vis", f"{exp['name']}_noisy_error_violins"), bbox_inches='tight')
    plt.close()


'''
Backend comparative imshow
'''
#__________________________________
def plot_backends(backend_comparison_dict):
    print("\033[K", f"Plotting Backends", end='\r')
    for plot, color in [('accuracy', 'plasma_r'), ('runtimes', 'viridis')]:
        fig, ax = plt.subplots()

        cmap = plt.get_cmap(color)
        cmap.set_under(color='k')

        data = np.asarray(backend_comparison_dict[plot])
        data_min = np.min(data)

        data = np.where(data == 100 if plot== "accuracy" else 0, -1, data)
        data_max = np.max(data)

        im = ax.imshow(data, cmap=cmap, vmin=data_min, vmax=data_max, aspect=0.5)

        axins = inset_axes(ax,
                    width="100%",  
                    height="10%",
                    loc='upper center',
                    borderpad=-4
                   )

        cbar = fig.colorbar(im, cax=axins, fraction=0.046, orientation="horizontal", extend="max")
        
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        # cbar.set_over()
        # # cbar.set_ticks(range(0, 255, 50))
        # # cbar.set_ticklabels(range(0, 255, 50))
        
        # annotate
        for x in range(len(data)):
            for y in range(len(data[x])):
                textVal = data[x][y]
                if textVal == -1:
                    textVal = 100 if plot == "accuracy" else 0
                
                text = ax.text(y, x, textVal, ha="center", va="center", color="w",  weight='bold')

        #-------
        # Major ticks
        ax.set_xticks(np.arange(0, len(backend_comparison_dict['size'])))
        ax.set_yticks(np.arange(0, len(backend_comparison_dict['backend'])))

        # Labels for major ticks
        ax.set_xticklabels(backend_comparison_dict['size'])
        ax.set_yticklabels(backend_comparison_dict['backend'])

        # Minor ticks
        ax.set_xticks(np.arange(-.5, len(backend_comparison_dict['size'])), minor=True)
        ax.set_yticks(np.arange(-.5, len(backend_comparison_dict['backend'])), minor=True)

        # Gridlines based on minor ticks
        ax.grid(which='minor', color='w', linestyle='-', linewidth=1)

        # Remove minor ticks
        ax.tick_params(which='minor', bottom=False, right=False, labelsize=0)
        ax.tick_params(which='major', bottom=True, top=False, labelbottom=True, labelsize=10)

        ax.yaxis.tick_right()

        # fig.tight_layout()
        # fig.subplots_adjust()

        plt.savefig(os.path.join("experiment_data_vis", f"{backend_comparison_dict['name']}_backend_{plot}"), bbox_inches='tight')
        plt.close()

'''
Shots Trends (FRQI - 256)
'''
#__________________________________
def plot_shots(shots_dict):
    print("\033[K", f"Plotting Shots", end='\r')
    bbox = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)

    fig, (ax_runtime, ax_accuracy) = plt.subplots(2, 1, sharex=True)

    ax_runtime.plot(np.arange(0, len(shots_dict['shots'])), shots_dict['runtimes'], color="orangered", marker=".")
    
    ax_runtime.set_ylabel("Runtime (s)")
    
    ax_runtime.grid(axis='y', alpha=0.5, linestyle="dotted", clip_on=False)
    for x,y in zip(np.arange(0, len(shots_dict['shots'])), shots_dict['runtimes']):
        ax_runtime.annotate("%.2f" % y, xy=(x+0.1,y), color="orangered", bbox=bbox)

    #-------
    ax_accuracy.plot(np.arange(0, len(shots_dict['shots'])), shots_dict['accuracy'], color="yellowgreen", marker=".")
    
    ax_accuracy.set_ylabel("Accuracy")

    ax_accuracy.set_xlabel("No. of shots")
    ax_accuracy.set_xticks(np.arange(0, len(shots_dict['shots'])))
    ax_accuracy.set_xticklabels(shots_dict['shots'])

    ax_accuracy.grid(axis='y', alpha=0.5, linestyle="dotted", clip_on=False)
    for x,y in zip(np.arange(0, len(shots_dict['shots'])), shots_dict['accuracy']):
        ax_accuracy.annotate("%.2f" % y, xy=(x+0.1,y), color="yellowgreen", bbox=bbox)
    
    #-------
    acc_to_run = [x/y for x,y in zip(shots_dict['accuracy'], shots_dict['runtimes'])]
    
    for i in np.where(acc_to_run == np.max(acc_to_run))[0]:
        ax_runtime.fill_betweenx([math.floor(ax_runtime.get_ylim()[0]), math.ceil(ax_runtime.get_ylim()[1])+1], i-0.3, i+0.3, alpha=0.3, color="orangered", hatch="/")
        
        ax_accuracy.fill_betweenx([ax_accuracy.get_ylim()[0]-0.02, ax_accuracy.get_ylim()[1]+0.02], i-0.3, i+0.3, alpha=0.3, color="yellowgreen", hatch="/")

    plt.tight_layout()
    plt.savefig(os.path.join("experiment_data_vis", "FRQI_shots"), bbox_inches='tight')
    plt.close()

'''
Error violins
'''
#__________________________________
def plot_supermarq(exp):
    print("\033[K", f"Plotting SupermarQ", end='\r')
    supermarq_metrics.plot_benchmark(data=[exp['widths'], exp['supermarq_metrics']], show=False, savefn=os.path.join("experiment_data_vis", "Qubit Lattice_supermarq"))

#__________________________________
def plot(exp_dict=None, shots_dict=None):
    try:
        if exp_dict:
            plot_runtimes(exp=exp_dict)
            plot_circuit_stats(exp_dict)
            plot_data(exp=exp_dict)
            plot_supermarq(exp=exp_dict)
        if shots_dict:
            plot_shots(shots_dict=shots_dict)
    except Exception as e:
        print(f"! ERROR while plotting !\n\t{traceback.format_exc()}\nDict:\n\t{exp_dict or shots_dict}\n")

#__________________________________
if __name__ == "__main__":
    if sys.argv[1][-3:] == "log":
        log_file = os.path.join("experiment_data", f"btq_{sys.argv[1]}")
        exp = parse_log(log_file)
        plot(exp)

    
    elif sys.argv[1][-3:] == "pkl":
        with open(os.path.join("experiment_data", f"{sys.argv[1]}"), 'rb') as f:
           exp = pickle.load(f)
        # print(exp)
        plot(exp)

    elif sys.argv[1] == "backend":
        plot_backends(backend_comparison_dict=backend_comparison_dict)
    
    elif sys.argv[1].startswith("frqi_shots"):
        with open(os.path.join("experiment_data", f"{sys.argv[1]}.pkl"), 'rb') as f:
           exp = pickle.load(f)

        plot(shots_dict=exp)
