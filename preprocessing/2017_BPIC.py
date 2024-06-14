from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter
import pandas as pd
from create_benchmarks import remainTimeOrClassifBenchmark, nextEventBenchmark

NAME = "BPI Challenge 2017"    
PATH = "."
START_DATE = None 
END_DATE = "2017-01"
MAX_DAYS = 47.81
TEST_LEN_SHARE = .2
OUTPUT_TYPE = "xes"

KEYWORDS_DICT = {}
KEYWORDS_DICT["approved"] = ["O_Accepted"]
KEYWORDS_DICT["declined"] = ["O_Refused"]
KEYWORDS_DICT["canceled"] = ["O_Cancelled"]

log = xes_importer.apply(PATH + "/" + NAME + ".xes")
dataset = converter.apply(log,variant=converter.Variants.TO_DATA_FRAME)

# remaining time and classification
dataset["classif_target"] = dataset["concept:name"] 
remainTimeOrClassifBenchmark(dataset, PATH, NAME, START_DATE, END_DATE, MAX_DAYS, TEST_LEN_SHARE, OUTPUT_TYPE, KEYWORDS_DICT)

# next event
dataset["activity"] = dataset["concept:name"] + "_" + dataset["lifecycle:transition"]
nextEventBenchmark(dataset, PATH, NAME, START_DATE, END_DATE, MAX_DAYS, TEST_LEN_SHARE, OUTPUT_TYPE, "activity")





