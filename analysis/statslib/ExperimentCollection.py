from enum import Enum

l1d_controllers = [
    "Cache_1",
    "Cache_5",
    "Cache_9",
    "Cache_13",
    "Cache_17",
    "Cache_21",
    "Cache_25",
    "Cache_29",
]

l2_controllers = [
    "Cache_2",
    "Cache_6",
    "Cache_10",
    "Cache_14",
    "Cache_18",
    "Cache_22",
    "Cache_26",
    "Cache_30",
]

l3_controllers = [
    "Cache_3",
    "Cache_7",
    "Cache_11",
    "Cache_15",
    "Cache_19",
    "Cache_23",
    "Cache_27",
    "Cache_31",
]

pickle_cache_controllers = [
    "Cache_33"
]

memory_controllers = [
    "Memory_0",
    "Memory_1",
    "Memory_2",
    "Memory_3",
]

all_controllers = l1d_controllers + l2_controllers + l3_controllers + pickle_cache_controllers + memory_controllers

assert(len(l1d_controllers) == 8)
assert(len(l2_controllers) == 8)
assert(len(l3_controllers) == 8)
assert(len(pickle_cache_controllers) == 1)
assert(len(memory_controllers) == 4)

class PrivateCorePrefetcher(str, Enum):
    NONE = "None"
    IMP = "imp"
    AMPM = "ampm"
    STRIDE = "stride"
    MULTIV1 = "multiv1"

class StatSet:
    def __init__(self):
        self.stats = {}
    def addField(self, field_name, field_value):
        #assert(field_name not in self.stats)
        #if field_name in self.stats:
        #    print("warn:", field_name, "is repeated")
        field_value = float(field_value)
        self.stats[field_name] = field_value
    def getField(self, field_name):
        return self.stats[field_name]
    def getFieldFromFieldMatcher(self, matcher):
        return [self.stats[field_name] for field_name in self.stats.keys() if matcher(field_name)]
    def getFieldNameAndFieldFromFieldMatcher(self, matcher):
        return {field_name: self.stats[field_name] for field_name in self.stats.keys() if matcher(field_name)}
def parseStatFile(file_path):
    stats_set = []
    with open(file_path, "r") as f:
        curr_stats = StatSet()
        for line in f.readlines():
            if "---------- Begin Simulation Statistics ----------" in line:
                continue
            if "---------- End Simulation Statistics   ----------" in line:
                stats_set.append(curr_stats)
                curr_stats = StatSet()
                continue
            parts = line.split()
            if len(parts) <= 1:
                continue
            #if parts[0] in stats_of_interests:
            #print(parts)
            if parts[1] == "nan":
                parts[1] = 0
            if parts[1] == "|":
                continue
            curr_stats.addField(parts[0], parts[1])
    return stats_set
    
class ExperimentDescription:
    def __init__(self,
                 filepath,
                 application,
                 with_pdev,
                 graph_name,
                 prefetch_distance,
                 pdev_cache_size,
                 private_core_prefetcher,
                 ideal_l1,
                 ideal_l3):
        self.filepath = filepath
        self.application = application
        self.with_pdev = with_pdev
        # Legacy support
        if graph_name[0].isdigit():
            self.graph_name = "Graph-" + graph_name
        else:
            self.graph_name = graph_name
        self.prefetch_distance = prefetch_distance
        self.pdev_cache_size = pdev_cache_size
        self.private_core_prefetcher = private_core_prefetcher
        self.ideal_l1 = ideal_l1
        self.ideal_l3 = ideal_l3
    def get_hash(self):
        parts = [self.application,
                 "pdev" if self.with_pdev else "nopdev",
                 self.graph_name,
                 self.prefetch_distance,
                 self.pdev_cache_size,
                 self.private_core_prefetcher,
                 str(self.ideal_l1),
                 str(self.ideal_l3)
                ]
        return "-".join(parts)
    @classmethod
    def from_hash(cls, h):
        parts = h.split("-")
        return ExperimentDescription(
            "",
            application = parts[0],
            with_pdev = parts[1] == "pdev",
            graph_name = parts[2],
            prefetch_distance = parts[3],
            pdev_cache_size = parts[4],
            private_core_prefetcher = part[5],
            ideal_l1 = part[6],
            ideal_l3 = part[7],
        )
    def __eq__(self, other):
        return self.get_hash() == other.get_hash()
    def __str__(self):
        return self.get_hash()
    
class StatSets:
    def __init__(self, description, stats_set):
        self.description = description
        self.stats_set = stats_set
    @classmethod
    def fromStatFile(cls, description, file_path):
        return StatSets(description, parseStatFile(file_path))
    def is_valid(self): # TODO
        return len(self.stats_set) >= 2
    def match_description(self, match_f):
        return match_f(self.description)
    def getDataMovementCount(self, destination_controller, source_controller):
        field_name = f"{destination_controller}.data_tracker.data_movement_tracker.{source_controller}::samples"
        old_count = self.stats_set[-2].getField(field_name)
        new_count = self.stats_set[-1].getField(field_name)
        return new_count - old_count
    def getDataMovementCountNewOld(self, destination_controller, source_controller):
        field_name = f"{destination_controller}.data_tracker.data_movement_tracker.{source_controller}::samples"
        old_count = self.stats_set[-2].getField(field_name)
        new_count = self.stats_set[-1].getField(field_name)
        return (new_count, old_count)
    def getDataMovementTotalLatency(self, destination_controller, source_controller):
        field_name = f"{destination_controller}.data_tracker.data_movement_tracker.{source_controller}::mean"
        old_mean = self.stats_set[-2].getField(field_name)
        new_mean = self.stats_set[-1].getField(field_name)
        new_count, old_count = self.getDataMovementCountNewOld(destination_controller, source_controller)
        return new_count * new_mean - old_count * old_mean
    def getDataMovementMeanLatency(self, destination_controller, source_controller):
        field_name = f"{destination_controller}.data_tracker.data_movement_tracker.{source_controller}::mean"
        old_mean = self.stats_set[-2].getField(field_name)
        new_mean = self.stats_set[-1].getField(field_name)
        new_count, old_count = self.getDataMovementCountNewOld(destination_controller, source_controller)
        return (new_count * new_mean - old_count * old_mean) / (new_count - old_count)
    def getL1DHitCount(self, core_id):
        field_name = f"board.cache_hierarchy.core_tiles{core_id}.l1d_cache.cache.m_demand_hits"
        return self.stats_set[-1].getField(field_name) - self.stats_set[-2].getField(field_name)
    def getL1DMissCount(self, core_id):
        field_name = f"board.cache_hierarchy.core_tiles{core_id}.l1d_cache.cache.m_demand_misses"
        return self.stats_set[-1].getField(field_name) - self.stats_set[-2].getField(field_name)
    def getDataMovementCountToCoreFromAllDataSources(self, destination_controller, core_id):
        local_l1d = l1d_controllers[core_id]
        local_l2 = l2_controllers[core_id]
        other_l1d = set(l1d_controllers) - { local_l1d }
        other_l2 = set(l2_controllers) - { local_l2 }
        count = {
            "local_l1d": 0,
            "local_l2": 0,
            "other_l1d": 0,
            "other_l2": 0,
            "l3": 0,
            "pickle_cache": 0,
            "memory": 0
        }
        count["local_l1d"] = self.getL1DHitCount(core_id)
        count["local_l2"] = self.getDataMovementCount(destination_controller, local_l2)
        for l1 in other_l1d:
            count["other_l1d"] += self.getDataMovementCount(destination_controller, l1)
        for l2 in other_l2:
            count["other_l2"] += self.getDataMovementCount(destination_controller, l2)
        for l3 in l3_controllers:
            count["l3"] += self.getDataMovementCount(destination_controller, l3)
        for pickle in pickle_cache_controllers:
            count["pickle_cache"] += self.getDataMovementCount(destination_controller, pickle)
        for memory in memory_controllers:
            count["memory"] += self.getDataMovementCount(destination_controller, memory)
        return count
    def getDataMovementMeanLatencyToCoreFromAllDataSources(self, destination_controller, core_id):
        local_l1d = l1d_controllers[core_id]
        local_l2 = l2_controllers[core_id]
        other_l1d = set(l1d_controllers) - { local_l1d }
        other_l2 = set(l2_controllers) - { local_l2 }
        count = {
            "local_l1d": 0,
            "local_l2": 0,
            "other_l1d": 0,
            "other_l2": 0,
            "l3": 0,
            "pickle_cache": 0,
            "memory": 0
        }
        total_latency = {
            "local_l2": 0,
            "other_l1d": 0,
            "other_l2": 0,
            "l3": 0,
            "pickle_cache": 0,
            "memory": 0
        }
        count["local_l2"] = self.getDataMovementCount(destination_controller, local_l2)
        total_latency["local_l2"] = self.getDataMovementTotalLatency(destination_controller, local_l2)
        for l1 in other_l1d:
            count["other_l1d"] += self.getDataMovementCount(destination_controller, l1)
            total_latency["other_l1d"] += self.getDataMovementTotalLatency(destination_controller, l1)
        for l2 in other_l2:
            count["other_l2"] += self.getDataMovementCount(destination_controller, l2)
            total_latency["other_l2"] += self.getDataMovementTotalLatency(destination_controller, l2)
        for l3 in l3_controllers:
            count["l3"] += self.getDataMovementCount(destination_controller, l3)
            total_latency["l3"] += self.getDataMovementTotalLatency(destination_controller, l3)
        for pickle in pickle_cache_controllers:
            count["pickle_cache"] += self.getDataMovementCount(destination_controller, pickle)
            total_latency["pickle_cache"] += self.getDataMovementTotalLatency(destination_controller, pickle)
        for memory in memory_controllers:
            count["memory"] += self.getDataMovementCount(destination_controller, memory)
            total_latency["memory"] += self.getDataMovementTotalLatency(destination_controller, memory)
        avg_latency = {}
        for cat in count.keys():
            if count[cat] == 0:
                avg_latency[cat] = 0
            else:
                avg_latency[cat] = total_latency[cat] / count[cat]
        return avg_latency
    def getField(self, field_name, metadata = ""):
        if field_name == "simSeconds":
            return self.stats_set[-1].getField("simSeconds") - self.stats_set[-2].getField("simSeconds")
        if field_name == "timely prefetches count":
            matcher = lambda field_name: "timelyPrefetches" in field_name and "samples" in field_name
            return sum(self.stats_set[-1].getFieldFromFieldMatcher(matcher))
        if field_name == "late prefetches count":
            matcher = lambda field_name: "latePrefetches" in field_name and "samples" in field_name
            return sum(self.stats_set[-1].getFieldFromFieldMatcher(matcher))
        if field_name == "task count":
            matcher = lambda field_name: "taskCount" in field_name
            result = self.stats_set[-1].getFieldFromFieldMatcher(matcher)
            return sum(result)
        if field_name == "loadToUse":
            mean_matcher = lambda field_name: "loadToUse" in field_name and "mean" in field_name
            count_matcher = lambda field_name: "loadToUse" in field_name and "samples" in field_name
            old_mean_result = self.stats_set[-2].getFieldNameAndFieldFromFieldMatcher(mean_matcher)
            old_count_result = self.stats_set[-2].getFieldNameAndFieldFromFieldMatcher(count_matcher)
            new_mean_result = self.stats_set[-1].getFieldNameAndFieldFromFieldMatcher(mean_matcher)
            new_count_result = self.stats_set[-1].getFieldNameAndFieldFromFieldMatcher(count_matcher)
            total_diff = {}
            count_diff = {}
            field_prefixes = []
            for field_name, new_field in new_mean_result.items():
                field_name_prefix = field_name.split("::")[0]
                field_prefixes.append(field_name_prefix)
            for field_prefix in field_prefixes:
                old_mean = old_mean_result[field_prefix+"::mean"]
                new_mean = new_mean_result[field_prefix+"::mean"]
                old_count = old_count_result[field_prefix+"::samples"]
                new_count = new_count_result[field_prefix+"::samples"]
                total_diff[field_prefix] = new_count * new_mean - old_count * old_mean
                count_diff[field_prefix] = new_count - old_count
                if count_diff[field_prefix] == 0:
                    print(old_mean, new_mean, old_count, new_count)
                    assert(total_diff[field_prefix] == 0)
            if sum(count_diff.values()) == 0:
                avg = 0
            else:
                avg = sum(total_diff.values()) / sum(count_diff.values())
            return avg
        if field_name == "bwTotal":
            bytes_read_matcher = lambda field_name: "dram.bytesRead::total" in field_name
            bytes_written_matcher = lambda field_name: "dram.bytesWritten::total" in field_name
            old_bytes_read = self.stats_set[-2].getFieldFromFieldMatcher(bytes_read_matcher)
            old_bytes_written = self.stats_set[-2].getFieldFromFieldMatcher(bytes_written_matcher)
            new_bytes_read = self.stats_set[-1].getFieldFromFieldMatcher(bytes_read_matcher)
            new_bytes_written = self.stats_set[-1].getFieldFromFieldMatcher(bytes_written_matcher)
            sim_seconds = self.stats_set[-1].getField("simSeconds") - self.stats_set[-2].getField("simSeconds")
            bytes_diff = sum(new_bytes_read) + sum(new_bytes_written) - sum(old_bytes_read) - sum(old_bytes_written)
            return bytes_diff / sim_seconds
        if field_name == "pageHitRate":
            read_bursts_matcher = lambda field_name: "dram.readBursts" in field_name
            read_row_hit_matcher = lambda field_name: "dram.readRowHits" in field_name
            old_read_bursts_result = self.stats_set[-2].getFieldNameAndFieldFromFieldMatcher(read_bursts_matcher)
            old_read_row_hit_result = self.stats_set[-2].getFieldNameAndFieldFromFieldMatcher(read_row_hit_matcher)
            new_read_bursts_result = self.stats_set[-1].getFieldNameAndFieldFromFieldMatcher(read_bursts_matcher)
            new_read_row_hit_result = self.stats_set[-1].getFieldNameAndFieldFromFieldMatcher(read_row_hit_matcher)
            read_bursts_diff = {}
            read_row_hit_diff = {}
            field_prefixes = []
            for field_name, new_field in new_read_bursts_result.items():
                field_name_prefix = ".".join(field_name.split(".")[:-1])
                field_prefixes.append(field_name_prefix)
            for field_prefix in field_prefixes:
                old_read_bursts = old_read_bursts_result[field_prefix+".readBursts"]
                old_read_row_hits = old_read_row_hit_result[field_prefix+".readRowHits"]
                new_read_bursts = new_read_bursts_result[field_prefix+".readBursts"]
                new_read_row_hits = new_read_row_hit_result[field_prefix+".readRowHits"]
                read_bursts_diff[field_prefix] = new_read_bursts - old_read_bursts
                read_row_hit_diff[field_prefix] = new_read_row_hits - old_read_row_hits
            hitRate = sum(read_row_hit_diff.values()) / sum(read_bursts_diff.values())
            return hitRate
        if field_name in {"total_msg_wait_time", "total_bw_sat_cy", "acc_link_utilization"}:
            if metadata:
                matcher = lambda line: field_name in line and metadata in line
            else:
                matcher = lambda line: field_name in line
            old_vals = self.stats_set[-2].getFieldNameAndFieldFromFieldMatcher(matcher)
            new_vals = self.stats_set[-1].getFieldNameAndFieldFromFieldMatcher(matcher)
            diff = {}
            field_prefixes = []
            for name, new_field in new_vals.items():
                name_prefix = ".".join(name.split(".")[:-1])
                field_prefixes.append(name_prefix)
            for field_prefix in field_prefixes:
                old_val = old_vals[field_prefix+"."+field_name]
                new_val = new_vals[field_prefix+"."+field_name]
                diff[field_prefix] = new_val - old_val
            return sum(diff.values())
        assert(False and "Not supported field name")
        
class ExperimentResultCollection:
    def __init__(self, stat_sets):
        self.stat_sets = stat_sets
    @classmethod
    def get_from_discover_stats(cls, discoverStats):
        return ExperimentResultCollection(discoverStats())
    def get_matches(self, match_f):
        matched_stats = []
        for stats in self.stat_sets:
            if stats.is_valid() and stats.match_description(match_f):
                matched_stats.append(stats)
        return matched_stats
