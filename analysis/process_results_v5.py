from statslib.ExperimentCollection import (
    ExperimentResultCollection,
    PrivateCorePrefetcher,
    ExperimentDescription,
    StatSets,
    l1d_controllers,
    l2_controllers,
    l3_controllers,
    pickle_cache_controllers,
    memory_controllers
)

import glob
import multiprocessing as mp
from pathlib import Path

import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pds
import seaborn as sns

PATH_TO_DATA = Path("/workdir/ARTIFACTS/results_v5/")
PATH_TO_ASSETS = Path("/workdir/pickle-resources/analysis/figures/")

def getStatSetsFromPath(file_path):
    file_path = Path(file_path)
    parts = file_path.parents[0].name.split('-')
    application = parts[0]
    graph_name = parts[1]
    with_pdev = False 
    prefetch_distance = "0"
    pdev_cache_size = "32KiB" 
    private_core_prefetcher = PrivateCorePrefetcher.NONE
    ideal_l1 = False
    ideal_l3 = False
    if parts[2] == "baseline":
        if len(parts) > 3:
            private_core_prefetcher = PrivateCorePrefetcher(parts[3])
    elif parts[2].startswith("pdev"):
        with_pdev = True
        pdev_info = parts[2].split("_")
        prefetch_distance = str(int(pdev_info[2]) - int(pdev_info[4]))
        if len(parts) > 3:
            private_core_prefetcher = PrivateCorePrefetcher(parts[3])
    elif parts[2] == "ideal_l3":
        ideal_l3 = True
    experiment_description = ExperimentDescription(
        filepath = file_path,
        application = application,
        with_pdev = with_pdev,
        graph_name = graph_name,
        prefetch_distance = prefetch_distance,
        pdev_cache_size = pdev_cache_size,
        private_core_prefetcher = private_core_prefetcher,
        ideal_l1 = ideal_l1,
        ideal_l3 = ideal_l3,
    )
    return StatSets.fromStatFile(experiment_description, file_path)

def discoverStats():
    stats = []
    print(PATH_TO_DATA)
    task_list = []
    for file_path in glob.glob(str(PATH_TO_DATA / "*"/ "stats.txt")):
        task_list.append(file_path)
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(getStatSetsFromPath, task_list)
    stats = results
    print(f"Discovered {len(stats)} stat files")
    return stats

result_collection = ExperimentResultCollection.get_from_discover_stats(discoverStats)

def plot_cat_plot(field_name, normalize, normalize_factor, graphs, y_label, metadata):
    baseline_per_graph_name = {}
    baseline_stats = {}
    for baseline_prefetcher in baseline_prefetchers:
        baseline_criteria = lambda description: all([
            description.application == application,
            description.with_pdev == False,
            description.graph_name in graphs,
            description.private_core_prefetcher == PrivateCorePrefetcher(baseline_prefetcher),
            description.ideal_l1 == False,
            description.ideal_l3 == False,
        ])
        baseline_stats[baseline_prefetcher] = result_collection.get_matches(baseline_criteria)

    pickle_prefetcher_criteria = lambda description: all([
        description.application == application,
        description.with_pdev == True,
        description.prefetch_distance == DEFAULT_PREFETCH_DISTANCE,
        description.graph_name in graphs,
        description.pdev_cache_size == "32KiB",
        #description.private_core_prefetcher == PrivateCorePrefetcher.NONE,
        description.ideal_l1 == False,
        description.ideal_l3 == False,
    ])
    pickle_prefetcher_stats = result_collection.get_matches(pickle_prefetcher_criteria)
    data = {
        "graph_name": [],
        "prefetcher": [],
        field_name: [],
    }

    for stats in baseline_stats["None"]:
        data["graph_name"].append(stats.description.graph_name)
        data["prefetcher"].append("Baseline (No Prefetchers)")
        data[field_name].append(stats.getField(field_name, metadata))
        if normalize:
            data[field_name][-1] /= stats.getField("simSeconds") * 1e12 / 250
            data[field_name][-1] *= normalize_factor

    for baseline_prefetcher in baseline_prefetchers[1:]:
        for stats in baseline_stats[baseline_prefetcher]:
            data["graph_name"].append(stats.description.graph_name)
            data["prefetcher"].append(baseline_prefetcher)
            data[field_name].append(stats.getField(field_name, metadata))
            if normalize:
                data[field_name][-1] /= stats.getField("simSeconds") * 1e12 / 250
            data[field_name][-1] *= normalize_factor

    for stats in pickle_prefetcher_stats:
        data["graph_name"].append(stats.description.graph_name)
        if stats.description.private_core_prefetcher == PrivateCorePrefetcher.NONE:
            data["prefetcher"].append("picklepf-distance-"+stats.description.prefetch_distance)
        else:
            data["prefetcher"].append("picklepf-distance-"+stats.description.prefetch_distance+"-"+stats.description.private_core_prefetcher)
        data[field_name].append(stats.getField(field_name, metadata))
        if normalize:
            data[field_name][-1] /= stats.getField("simSeconds") * 1e12 / 250
            data[field_name][-1] *= normalize_factor

    df = pds.DataFrame(data)
    df.reset_index();
    x_order = graphs
    hue_order = ["Baseline (No Prefetchers)", "stride", "ampm", "imp", "multiv1", ] \
        + [f"picklepf-distance-{DEFAULT_PREFETCH_DISTANCE}"] \
        + [f"picklepf-distance-{DEFAULT_PREFETCH_DISTANCE}-{pf}" for pf in baseline_prefetchers[1:]]
    plt.rcParams.update({'font.size': 18})
    ax = sns.catplot(data=df, kind="bar", x="graph_name", y=field_name, hue="prefetcher", palette="tab10", order = x_order, hue_order=hue_order, aspect=3.5,)\
            .set(ylabel=y_label, xlabel="Graph Names")

def plot_src_distribution(df, graph_name):
    df.reset_index();
    x_order = [f"{graph_name}-{prefetcher}"
               for prefetcher in ["baseline", f"picklepf-distance-{DEFAULT_PREFETCH_DISTANCE}"]]
               #for prefetcher in ["baseline", "stride", "ampm", "imp", "multiv1", "picklepf-distance-32"]]
    x_order = reversed(x_order)
    ax = df.set_index("graph_name").loc[x_order].plot(kind="barh", stacked=True, figsize=(12,3), xlabel="Count")
    handles, previous_labels = ax.get_legend_handles_labels()
    new_labels = []
    label_map = {
        "local_l1d": "Local L1D",
        "other_l1d": "Other L1D",
        "local_l2": "Local L2",
        "other_l2": "Other L2",
        "l3": "LLC",
        "pickle_cache": "Pickle Cache",
        "memory": "DRAM",
    }
    for handle in handles:
        new_labels.append(label_map[handle._label])
    ax.legend(labels=new_labels, bbox_to_anchor=(1,0.5), loc='center left')
    ax.set_ylabel("Graph & Prefetchers")
    ax.set_xlabel("Data Source Count")

def plot_src_distribution_of_specific_core(stats_list, core_id, with_l1d):
    assert(isinstance(stats_list, list))
    source_count = {
        "graph_name": [],
        "local_l1d": [],
        "other_l1d": [],
        "local_l2": [],
        "other_l2": [],
        "l3": [],
        "pickle_cache": [],
        "memory": []
    }
    for i, stats in enumerate(stats_list):
        counts = stats.getDataMovementCountToCoreFromAllDataSources(f"board.cache_hierarchy.core_tiles{core_id}.l1d_cache", core_id)
        mean_latency = stats.getDataMovementMeanLatencyToCoreFromAllDataSources(f"board.cache_hierarchy.core_tiles{core_id}.l1d_cache", core_id)
        graph_name = [stats.description.graph_name]
        if stats.description.with_pdev:
            graph_name.append("picklepf-distance-" + str(stats.description.prefetch_distance))
        if stats.description.private_core_prefetcher != PrivateCorePrefetcher.NONE:
            graph_name.append(str(stats.description.private_core_prefetcher))
        elif stats.description.with_pdev == False:
            graph_name.append("baseline")
        source_count["graph_name"].append("-".join(graph_name))
        source_count["local_l1d"].append(counts["local_l1d"])
        source_count["other_l1d"].append(counts["other_l1d"])
        source_count["local_l2"].append(counts["local_l2"])
        source_count["other_l2"].append(counts["other_l2"])
        source_count["l3"].append(counts["l3"])
        source_count["pickle_cache"].append(counts["pickle_cache"])
        source_count["memory"].append(counts["memory"])
    if not with_l1d:
        #del source_count["local_l1d"]
        source_count["local_l1d"] = [0] * len(source_count["memory"])
    #del source_count["other_l1d"]
    #del source_count["local_l2"]
    #del source_count["other_l2"]
    #del source_count["l3"]
    #del source_count["pickle_cache"]
    df = pds.DataFrame(source_count)
    plot_src_distribution(df, stats.description.graph_name)

def plot_src_distribution_all_cores(stats_list, with_l1d):
    assert(isinstance(stats_list, list))
    source_count = {
        "graph_name": [],
        "local_l1d": [],
        "local_l2": [],
        "other_l1d": [],
        "local_l2": [],
        "other_l2": [],
        "l3": [],
        "pickle_cache": [],
        "memory": []
    }
    for i, stats in enumerate(stats_list):
        counts = {
            "local_l1d": 0,
            "local_l2": 0,
            "other_l1d": 0,
            "other_l2": 0,
            "l3": 0,
            "pickle_cache": 0,
            "memory": 0
        }
        for core_id in range(len(l1d_controllers)):
            local_counts = stats.getDataMovementCountToCoreFromAllDataSources(f"board.cache_hierarchy.core_tiles{core_id}.l1d_cache", core_id)
            mean_latency = stats.getDataMovementMeanLatencyToCoreFromAllDataSources(f"board.cache_hierarchy.core_tiles{core_id}.l1d_cache", core_id)
            for field in counts.keys():
                counts[field] += local_counts[field]
        graph_name = [stats.description.graph_name]
        if stats.description.with_pdev:
            graph_name.append("picklepf-distance-" + str(stats.description.prefetch_distance))
        if stats.description.private_core_prefetcher != PrivateCorePrefetcher.NONE:
            graph_name.append(str(stats.description.private_core_prefetcher))
        elif stats.description.with_pdev == False:
            graph_name.append("baseline")
        source_count["graph_name"].append("-".join(graph_name))
        source_count["local_l1d"].append(counts["local_l1d"])
        source_count["other_l1d"].append(counts["other_l1d"])
        source_count["local_l2"].append(counts["local_l2"])
        source_count["other_l2"].append(counts["other_l2"])
        source_count["l3"].append(counts["l3"])
        source_count["pickle_cache"].append(counts["pickle_cache"])
        source_count["memory"].append(counts["memory"])
    if not with_l1d:
        #del source_count["local_l1d"]
        source_count["local_l1d"] = [0] * len(source_count["memory"])
    #del source_count["other_l1d"]
    #del source_count["local_l2"]
    #del source_count["other_l2"]
    #del source_count["l3"]
    #del source_count["pickle_cache"]
    df = pds.DataFrame(source_count)
    plot_src_distribution(df, stats.description.graph_name)

def plot_mem_accesses(graph_name, with_l1d):
    baseline_roadNetCA_criteria = lambda description: all([
        description.application == application,
        description.with_pdev == False,
        description.graph_name == graph_name,
        #description.private_core_prefetcher == PrivateCorePrefetcher.NONE,
        description.ideal_l1 == False,
        description.ideal_l3 == False,
    ])
    baseline_stats = result_collection.get_matches(baseline_roadNetCA_criteria)
    assert(len(baseline_stats) == 5)

    pickle_roadNetCA_criteria = lambda description: all([
        description.application == application,
        description.with_pdev == True,
        description.prefetch_distance == DEFAULT_PREFETCH_DISTANCE,
        description.graph_name == graph_name,
        description.pdev_cache_size == "32KiB",
        description.private_core_prefetcher == PrivateCorePrefetcher.NONE,
        description.ideal_l1 == False,
        description.ideal_l3 == False,
    ])
    pickle_prefetcher_stats = result_collection.get_matches(pickle_roadNetCA_criteria)
    assert(len(pickle_prefetcher_stats) == 1)

    plot_src_distribution_all_cores(
        stats_list = baseline_stats + pickle_prefetcher_stats,
        with_l1d = with_l1d
    )

application = "bfs"
baseline_prefetchers = ["None", "stride", "ampm", "imp", "multiv1"]
graphs = ["amazon", "youtube", "web_google", "web_berkstan", "roadNetCA", "wiki_talk", "higgs", "wiki_topcats", "pokec", "livejournal"]
color_palette = sns.color_palette("tab10")[:4] + sns.color_palette("tab10")[5:]
freq = 4e9
oneTick = 1e-12
tickPerCycle = 1/freq/oneTick
tickToCycle = lambda t: t / tickPerCycle
DEFAULT_PREFETCH_DISTANCE="32"

baseline_stats = {}
for baseline_prefetcher in baseline_prefetchers:
    baseline_criteria = lambda description: all([
        description.application == application,
        description.with_pdev == False,
        description.graph_name in graphs,
        description.private_core_prefetcher == PrivateCorePrefetcher(baseline_prefetcher),
        description.ideal_l1 == False,
        description.ideal_l3 == False,
    ])
    baseline_stats[baseline_prefetcher] = result_collection.get_matches(baseline_criteria)

ideal_l3_criteria = lambda description: all([
    description.application == application,
    description.with_pdev == False,
    description.graph_name in graphs,
    description.private_core_prefetcher == PrivateCorePrefetcher.NONE,
    description.ideal_l1 == False,
    description.ideal_l3 == True,
])
ideal_l3_stats = result_collection.get_matches(ideal_l3_criteria)

pickle_prefetcher_criteria = lambda description: all([
    description.application == application,
    description.with_pdev == True,
    description.prefetch_distance == DEFAULT_PREFETCH_DISTANCE,
    description.graph_name in graphs,
    description.pdev_cache_size == "32KiB",
    #description.private_core_prefetcher == PrivateCorePrefetcher.NONE,
    description.ideal_l1 == False,
    description.ideal_l3 == False,
])
pickle_prefetcher_stats = result_collection.get_matches(pickle_prefetcher_criteria)

def plot_speedup_figure():
    # plot
    baseline_per_graph_name = {}
    speedup_data = {
        "graph_name": [],
        "prefetcher": [],
        "speedup": [],
    }

    s = []
    s2 = []

    for stats in baseline_stats["None"]:
        speedup_data["graph_name"].append(stats.description.graph_name)
        speedup_data["prefetcher"].append("Baseline (No Prefetchers)")
        speedup_data["speedup"].append(1.0)
        baseline_per_graph_name[stats.description.graph_name] = stats.getField("simSeconds")

    for stats in ideal_l3_stats:
        speedup_data["graph_name"].append(stats.description.graph_name)
        speedup_data["prefetcher"].append("Ideal L3")
        baseline_time = baseline_per_graph_name[stats.description.graph_name]
        ideal_l3_time = stats.getField("simSeconds")
        speedup_data["speedup"].append(baseline_time / ideal_l3_time)

    for baseline_prefetcher in baseline_prefetchers[1:]:
        for stats in baseline_stats[baseline_prefetcher]:
            speedup_data["graph_name"].append(stats.description.graph_name)
            speedup_data["prefetcher"].append(baseline_prefetcher)
            baseline_time = baseline_per_graph_name[stats.description.graph_name]
            baseline_prefetcher_time = stats.getField("simSeconds")
            speedup_data["speedup"].append(baseline_time / baseline_prefetcher_time)

    for stats in pickle_prefetcher_stats:
        speedup_data["graph_name"].append(stats.description.graph_name)
        if stats.description.private_core_prefetcher == PrivateCorePrefetcher.NONE:
            speedup_data["prefetcher"].append("picklepf-distance-"+stats.description.prefetch_distance)
        else:
            speedup_data["prefetcher"].append("picklepf-distance-"+stats.description.prefetch_distance+"-"+stats.description.private_core_prefetcher)
        baseline_time = baseline_per_graph_name[stats.description.graph_name]
        prefetcher_time = stats.getField("simSeconds")
        speedup_data["speedup"].append(baseline_time / prefetcher_time)
        s2.append(speedup_data["speedup"][-1])
        if stats.description.private_core_prefetcher == PrivateCorePrefetcher.NONE:
            s.append(speedup_data["speedup"][-1])

    pdev = {}
    for i in range(len(speedup_data["speedup"])):
        graph_name = speedup_data["graph_name"][i]
        prefetcher = speedup_data["prefetcher"][i]
        speedup = speedup_data["speedup"][i]
        pdev[(graph_name, prefetcher)] = speedup

    speedups = {"stride": [], "ampm": [], "imp": [], "multiv1": []}
    for i in range(len(speedup_data["speedup"])):
        graph_name = speedup_data["graph_name"][i]
        prefetcher = speedup_data["prefetcher"][i]
        speedup = speedup_data["speedup"][i]
        prefetcher_parts = prefetcher.split("-")
        if len(prefetcher_parts) > 3:
            prefetcher = prefetcher_parts[-1]
        else:
            continue
        speedups[prefetcher].append(speedup / pdev[(graph_name, prefetcher)])

    df = pds.DataFrame(speedup_data)
    df.reset_index();
    x_order = graphs
    hue_order = ["Baseline (No Prefetchers)", "stride", "ampm", "imp", "multiv1", ] \
            + [f"picklepf-distance-{DEFAULT_PREFETCH_DISTANCE}"] \
            + [f"picklepf-distance-{DEFAULT_PREFETCH_DISTANCE}-{pf}" for pf in baseline_prefetchers[1:]]
    plt.rcParams.update({'font.size': 18})
    ax = sns.catplot(data=df, kind="bar", x="graph_name", y="speedup", hue="prefetcher", palette="tab10", order = x_order, hue_order=hue_order, aspect=3.5,)\
            .set(ylabel="Speedups", xlabel="Graph Names")
    plt.axhline(y=1, color='black', linestyle='-')
    plt.axhline(y=2, color='gray', linestyle='--')
    plt.savefig(PATH_TO_ASSETS / "01-prefetcher_comparison.pdf", bbox_inches='tight')
    print("Generated plot:", PATH_TO_ASSETS / "01-prefetcher_comparison.pdf")

def plot_load_to_use_figure():
    # plot
    baseline_per_graph_name = {}
    load_to_use_data = {
        "graph_name": [],
        "prefetcher": [],
        "load_to_use": [],
    }

    s = []

    for stats in baseline_stats["None"]:
        load_to_use_data["graph_name"].append(stats.description.graph_name)
        load_to_use_data["prefetcher"].append("Baseline (No Prefetchers)")
        #load_to_use_data["load_to_use"].append(1.0)
        baseline_per_graph_name[stats.description.graph_name] = stats.getField("loadToUse")
        load_to_use_data["load_to_use"].append(stats.getField("loadToUse"))
    for baseline_prefetcher in baseline_prefetchers[1:]:
        for stats in baseline_stats[baseline_prefetcher]:
            load_to_use_data["graph_name"].append(stats.description.graph_name)
            load_to_use_data["prefetcher"].append(baseline_prefetcher)
            baseline_time = baseline_per_graph_name[stats.description.graph_name]
            baseline_prefetcher_time = stats.getField("loadToUse")
            load_to_use_data["load_to_use"].append(baseline_prefetcher_time)

    for stats in pickle_prefetcher_stats:
        load_to_use_data["graph_name"].append(stats.description.graph_name)
        if stats.description.private_core_prefetcher == PrivateCorePrefetcher.NONE:
            load_to_use_data["prefetcher"].append("picklepf-distance-"+stats.description.prefetch_distance)
        else:
            load_to_use_data["prefetcher"].append("picklepf-distance-"+stats.description.prefetch_distance+"-"+stats.description.private_core_prefetcher)
        baseline_time = baseline_per_graph_name[stats.description.graph_name]
        prefetcher_time = stats.getField("loadToUse")
        load_to_use_data["load_to_use"].append(prefetcher_time)

    df = pds.DataFrame(load_to_use_data)
    df.reset_index();
    x_order = graphs
    hue_order = ["Baseline (No Prefetchers)", "stride", "ampm", "imp", "multiv1", ] \
            + [f"picklepf-distance-{DEFAULT_PREFETCH_DISTANCE}"] \
            + [f"picklepf-distance-{DEFAULT_PREFETCH_DISTANCE}-{pf}" for pf in baseline_prefetchers[1:]]
    plt.rcParams.update({'font.size': 18})
    ax = sns.catplot(data=df, kind="bar", x="graph_name", y="load_to_use", hue="prefetcher", palette="tab10", order = x_order, hue_order=hue_order, aspect=3.5,)\
            .set(ylabel="Load To Use (Cycles)", xlabel="Graph Names")
    plt.savefig(PATH_TO_ASSETS / "01-load_to_use.pdf", bbox_inches='tight')
    print("Generated plot:", PATH_TO_ASSETS / "01-load_to_use.pdf")

def plot_ideal_l3_comparison():
    # plot
    baseline_per_graph_name = {}
    speedup_data = {
        "graph_name": [],
        "prefetcher": [],
        "speedup": [],
    }

    ideal_l3 = {}
    pdev = {}

    for stats in baseline_stats["None"]:
        speedup_data["graph_name"].append(stats.description.graph_name)
        speedup_data["prefetcher"].append("Baseline (No Prefetchers)")
        speedup_data["speedup"].append(1.0)
        baseline_per_graph_name[stats.description.graph_name] = stats.getField("simSeconds")

    for stats in ideal_l3_stats:
        speedup_data["graph_name"].append(stats.description.graph_name)
        speedup_data["prefetcher"].append("Ideal LLC")
        baseline_time = baseline_per_graph_name[stats.description.graph_name]
        ideal_l3_time = stats.getField("simSeconds")
        speedup_data["speedup"].append(baseline_time / ideal_l3_time)
        ideal_l3[speedup_data["graph_name"][-1]] = speedup_data["speedup"][-1]

    for baseline_prefetcher in baseline_prefetchers[1:]:
        for stats in baseline_stats[baseline_prefetcher]:
            speedup_data["graph_name"].append(stats.description.graph_name)
            speedup_data["prefetcher"].append(baseline_prefetcher)
            baseline_time = baseline_per_graph_name[stats.description.graph_name]
            baseline_prefetcher_time = stats.getField("simSeconds")
            speedup_data["speedup"].append(baseline_time / baseline_prefetcher_time)

    for stats in pickle_prefetcher_stats:
        speedup_data["graph_name"].append(stats.description.graph_name)
        if stats.description.private_core_prefetcher == PrivateCorePrefetcher.NONE:
            speedup_data["prefetcher"].append("picklepf-distance-"+stats.description.prefetch_distance)
        else:
            speedup_data["prefetcher"].append("picklepf-distance-"+stats.description.prefetch_distance+"-"+stats.description.private_core_prefetcher)
        baseline_time = baseline_per_graph_name[stats.description.graph_name]
        prefetcher_time = stats.getField("simSeconds")
        speedup_data["speedup"].append(baseline_time / prefetcher_time)
        if speedup_data["prefetcher"][-1] == f"picklepf-distance-{DEFAULT_PREFETCH_DISTANCE}":
            pdev[speedup_data["graph_name"][-1]] = speedup_data["speedup"][-1]

    df = pds.DataFrame(speedup_data)
    df.reset_index();
    x_order = graphs
    hue_order = ["Baseline (No Prefetchers)", "Ideal LLC", f"picklepf-distance-{DEFAULT_PREFETCH_DISTANCE}"]
    plt.rcParams.update({'font.size': 18})
    ax = sns.catplot(data=df, kind="bar", x="graph_name", y="speedup", hue="prefetcher", palette="tab10", order = x_order, hue_order=hue_order, aspect=3.5,)\
            .set(ylabel="Speedups", xlabel="Graph Names")
    plt.axhline(y=1, color='black', linestyle='-')
    plt.axhline(y=2, color='gray', linestyle='--')
    plt.axhline(y=3, color='gray', linestyle='--')
    plt.axhline(y=4, color='gray', linestyle='--')
    plt.savefig(PATH_TO_ASSETS / "02-ideal_l3.pdf", bbox_inches='tight')
    print("Generated plot:", PATH_TO_ASSETS / "02-ideal_l3.pdf")

def plot_timeliness():
    baseline = {}
    for stats in baseline_stats["None"]:
        baseline[stats.description.graph_name] = stats.getField("simSeconds")
    pickle = {}
    for stats in pickle_prefetcher_stats:
        pickle_time = stats.getField("simSeconds")
        pickle[stats.description.graph_name] = baseline[stats.description.graph_name] / pickle_time
    ideal_l3 = {}
    for stats in ideal_l3_stats:
        ideal_l3_time = stats.getField("simSeconds")
        ideal_l3[stats.description.graph_name] = baseline[stats.description.graph_name] / ideal_l3_time

    timeliness_data = {
        "graph_name": [],
        "prefetcher": [],
        "upside_percentage": [],
        "timely_prefetch_percentage": []
    }

    for stats in pickle_prefetcher_stats:
        timeliness_data["graph_name"].append(stats.description.graph_name)
        if stats.description.private_core_prefetcher == PrivateCorePrefetcher.NONE:
            timeliness_data["prefetcher"].append("picklepf-distance-"+stats.description.prefetch_distance)
        else:
            timeliness_data["prefetcher"].append("picklepf-distance-"+stats.description.prefetch_distance+"-"+stats.description.private_core_prefetcher)
        timeliness_data["upside_percentage"].append((pickle[stats.description.graph_name]-1)/(ideal_l3[stats.description.graph_name]-1)*100)
        timeliness_data["timely_prefetch_percentage"].append(stats.getField("timely prefetches count") / stats.getField("task count") * 100)
        
    df = pds.DataFrame(timeliness_data)
    df.reset_index();
    x_order = graphs
    hue_order = [f"picklepf-distance-{DEFAULT_PREFETCH_DISTANCE}"]
    plt.rcParams.update({'font.size': 18})
    #ax = sns.catplot(data=df, kind="bar", x="graph_name", y="timely_prefetch_percentage", hue="prefetcher", palette="tab10", order = x_order, hue_order=hue_order, aspect=3.5,)\
    #        .set(ylabel="Timely Prefetch %", xlabel="Graph Names")

    ax1 = sns.set_style(style=None, rc=None)

    fig, ax1 = plt.subplots(figsize=(20,5))

    plt.rcParams.update({'font.size': 18})
    sns.barplot(df, x="graph_name", y="upside_percentage", legend=True, ax=ax1, order=graphs, label="Upside Percentage")
    ax2 = ax1.twinx()
    sns.pointplot(df, x="graph_name", y="timely_prefetch_percentage", ax=ax2, marker='o', color="orange", markersize = 15, order=graphs, label = "Timely Prefetch Percentage")
    ax2.set_ylim((0,100))
    ax1.set_ylim((0,100))
    ax1.legend(loc=(0, 0.88))
    ax2.legend(loc=(0, 0.76))
    ax2.set_ylabel("Timely Prefetch Percentage (%)")
    ax1.set_xlabel("Graph Names")
    ax1.set_ylabel("Upside Captured (%)")
    plt.savefig(PATH_TO_ASSETS / "03-timeliness.pdf", bbox_inches='tight')
    print("Generated plot:", PATH_TO_ASSETS / "03-timeliness.pdf")

def plot_dram_bandwidth():
    # plot
    baseline_per_graph_name = {}
    bw_total_data = {
        "graph_name": [],
        "prefetcher": [],
        "bw_total": [],
    }

    for stats in baseline_stats["None"]:
        bw_total_data["graph_name"].append(stats.description.graph_name)
        bw_total_data["prefetcher"].append("Baseline (No Prefetchers)")
        bw_total_data["bw_total"].append(stats.getField("bwTotal") / 1024 / 1024 / 1024)

    for baseline_prefetcher in baseline_prefetchers[1:]:
        for stats in baseline_stats[baseline_prefetcher]:
            bw_total_data["graph_name"].append(stats.description.graph_name)
            bw_total_data["prefetcher"].append(baseline_prefetcher)
            bw_total_data["bw_total"].append(stats.getField("bwTotal") / 1024 / 1024 / 1024)

    for stats in pickle_prefetcher_stats:
        bw_total_data["graph_name"].append(stats.description.graph_name)
        if stats.description.private_core_prefetcher == PrivateCorePrefetcher.NONE:
            bw_total_data["prefetcher"].append("picklepf-distance-"+stats.description.prefetch_distance)
        else:
            bw_total_data["prefetcher"].append("picklepf-distance-"+stats.description.prefetch_distance+"-"+stats.description.private_core_prefetcher)
        bw_total_data["bw_total"].append(stats.getField("bwTotal") / 1024 / 1024 / 1024)

    df = pds.DataFrame(bw_total_data)
    df.reset_index();
    x_order = graphs
    hue_order = ["Baseline (No Prefetchers)", "stride", "ampm", "imp", "multiv1", ] \
            + [f"picklepf-distance-{DEFAULT_PREFETCH_DISTANCE}"] \
            + [f"picklepf-distance-{DEFAULT_PREFETCH_DISTANCE}-{pf}" for pf in baseline_prefetchers[1:]]
    plt.rcParams.update({'font.size': 18})
    ax = sns.catplot(data=df, kind="bar", x="graph_name", y="bw_total", hue="prefetcher", palette="tab10", order = x_order, hue_order=hue_order, aspect=3.5,)\
            .set(ylabel="DRAM Bandwidth (GiB/s)", xlabel="Graph Names")
    plt.savefig(PATH_TO_ASSETS / "05-bwTotal.pdf", bbox_inches='tight')
    print("Generated plot:", PATH_TO_ASSETS / "05-bwTotal.pdf")

plot_speedup_figure()
plot_load_to_use_figure()
plot_ideal_l3_comparison()
plot_timeliness()
plot_dram_bandwidth()
plot_mem_accesses("livejournal", with_l1d=True)
plt.savefig(PATH_TO_ASSETS / "06-livejournal-data_src-withl1d.pdf", bbox_inches='tight')
print("Generated plot:", PATH_TO_ASSETS / "06-livejournal-data_src-withl1d.pdf")
plot_mem_accesses("livejournal", with_l1d=False)
plt.savefig(PATH_TO_ASSETS / "06-livejournal-data_src.pdf", bbox_inches='tight')
print("Generated plot:", PATH_TO_ASSETS / "06-livejournal-data_src.pdf")
plot_mem_accesses("web_berkstan", with_l1d=True)
plt.savefig(PATH_TO_ASSETS / "06-web_berkstan-data_src-withl1d.pdf", bbox_inches='tight')
print("Generated plot:", PATH_TO_ASSETS / "06-web_berkstan-data_src-withl1d.pdf")
plot_mem_accesses("web_berkstan", with_l1d=False)
plt.savefig(PATH_TO_ASSETS / "06-web_berkstan-data_src.pdf", bbox_inches='tight')
print("Generated plot:", PATH_TO_ASSETS / "06-web_berkstan-data_src.pdf")
plot_mem_accesses("roadNetCA", with_l1d=True)
plt.savefig(PATH_TO_ASSETS / "06-roadNetCA-data_src-withl1d.pdf", bbox_inches='tight')
print("Generated plot:", PATH_TO_ASSETS / "06-roadNetCA-data_src-withl1d.pdf")
plot_mem_accesses("roadNetCA", with_l1d=False)
plt.savefig(PATH_TO_ASSETS / "06-roadNetCA-data_src.pdf", bbox_inches='tight')
print("Generated plot:", PATH_TO_ASSETS / "06-roadNetCA-data_src.pdf")
plot_mem_accesses("amazon", with_l1d=True)
plt.savefig(PATH_TO_ASSETS / "06-amazon-data_src-withl1d.pdf", bbox_inches='tight')
print("Generated plot:", PATH_TO_ASSETS / "06-amazon-data_src-withl1d.pdf")
plot_mem_accesses("amazon", with_l1d=False)
plt.savefig(PATH_TO_ASSETS / "06-amazon-data_src.pdf", bbox_inches='tight')
print("Generated plot:", PATH_TO_ASSETS / "06-amazon-data_src.pdf")
plot_cat_plot("acc_link_utilization", normalize=True, metadata="l3_router", normalize_factor=1/2/8, graphs=graphs, y_label="Avg LLC Link Utilization per Cycle")
plt.savefig(PATH_TO_ASSETS / f"04-l3_utilization.pdf", bbox_inches='tight')
print("Generated plot:", PATH_TO_ASSETS / "04-l3_utilization.pdf")
