from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter
import pandas as pd
from create_benchmarks import remainTimeOrClassifBenchmark, nextEventBenchmark

NAME = "PrepaidTravelCost" 
PATH = "."
START_DATE = None 
END_DATE = "2019-01"
MAX_DAYS = 114.26
TEST_LEN_SHARE = .2

log = xes_importer.apply("./" + NAME +".xes")
dataset = converter.apply(log,variant=converter.Variants.TO_DATA_FRAME)
# remaining time
remainTimeOrClassifBenchmark(dataset, PATH, NAME, START_DATE, END_DATE, MAX_DAYS, TEST_LEN_SHARE)

