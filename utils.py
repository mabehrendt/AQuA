from transformers.adapters.composition import Parallel

def get_dynamic_parallel(adapter_number):
    if adapter_number == 1:
        return Parallel("adapter0")
    elif adapter_number == 2:
        return Parallel("adapter0", "adapter1")
    elif adapter_number == 3:
        return Parallel("adapter0", "adapter1", "adapter2")
    elif adapter_number == 4:
        return Parallel("adapter0", "adapter1", "adapter2", "adapter3")
    elif adapter_number == 5:
        return Parallel("adapter0", "adapter1", "adapter2", "adapter3", "adapter4")
    elif adapter_number == 6:
        return Parallel("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5")
    elif adapter_number == 7:
        return Parallel("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6")
    elif adapter_number == 8:
        return Parallel("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6", "adapter7")
    elif adapter_number == 9:
        return Parallel("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6", "adapter7",
                        "adapter8")
    elif adapter_number == 10:
        return Parallel("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6", "adapter7",
                        "adapter8", "adapter9")
    elif adapter_number == 11:
        return Parallel("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6", "adapter7",
                        "adapter8", "adapter9", "adapter10")
    elif adapter_number == 12:
        return Parallel("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6", "adapter7",
                        "adapter8", "adapter9", "adapter10", "adapter11")
    elif adapter_number == 13:
        return Parallel("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6", "adapter7",
                        "adapter8", "adapter9", "adapter10", "adapter11", "adapter12")
    elif adapter_number == 14:
        return Parallel("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6", "adapter7",
                        "adapter8", "adapter9", "adapter10", "adapter11", "adapter12", "adapter13")
    elif adapter_number == 15:
        return Parallel("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6", "adapter7",
                        "adapter8", "adapter9", "adapter10", "adapter11", "adapter12", "adapter13", "adapter14")
    elif adapter_number == 16:
        return Parallel("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6", "adapter7",
                        "adapter8", "adapter9", "adapter10", "adapter11", "adapter12", "adapter13", "adapter14",
                        "adapter15")
    elif adapter_number == 17:
        return Parallel("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6", "adapter7",
                        "adapter8", "adapter9", "adapter10", "adapter11", "adapter12", "adapter13", "adapter14",
                        "adapter15",
                        "adapter16")
    elif adapter_number == 18:
        return Parallel("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6", "adapter7",
                        "adapter8", "adapter9", "adapter10", "adapter11", "adapter12", "adapter13", "adapter14",
                        "adapter15",
                        "adapter16", "adapter17")
    elif adapter_number == 19:
        return Parallel("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6", "adapter7",
                        "adapter8", "adapter9", "adapter10", "adapter11", "adapter12", "adapter13", "adapter14",
                        "adapter15",
                        "adapter16", "adapter17", "adapter18")
    elif adapter_number == 20:
        return Parallel("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6",
                        "adapter7",
                        "adapter8", "adapter9", "adapter10", "adapter11", "adapter12", "adapter13",
                        "adapter14",
                        "adapter15",
                        "adapter16", "adapter17", "adapter18", "adapter19")
    elif adapter_number == 21:
        return Parallel("adapter0", "adapter1", "adapter2", "adapter3", "adapter4", "adapter5", "adapter6",
                 "adapter7", "adapter8", "adapter9", "adapter10", "adapter11", "adapter12", "adapter13",
                 "adapter14",
                 "adapter15",
                 "adapter16", "adapter17", "adapter18", "adapter19", "adapter20")
    else:
        print("specified adapter number has no setup yet: %d" % adapter_number)
