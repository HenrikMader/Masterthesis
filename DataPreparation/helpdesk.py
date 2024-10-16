from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter
import pandas as pd
from create_benchmarks import remainTimeOrClassifBenchmark, nextEventBenchmark
import pm4py
import time

NAME = "../RawData/helpdesk"
PATH = "."
START_DATE = "2010-02"
END_DATE = "2013-12"
MAX_DAYS = 58
TEST_LEN_SHARE = .2

dataFrame = pd.read_csv('helpdesk.csv', sep=',')
dataset = pm4py.format_dataframe(dataFrame, case_id='Case ID', activity_key='Activity', timestamp_key='Complete Timestamp')

print("dataset before")
print(dataset)
#log = xes_importer.apply(PATH + "/" + NAME +".xes")
#dataset = converter.apply(log,variant=converter.Variants.TO_DATA_FRAME)
# remaining time
remainTimeOrClassifBenchmark(dataset, PATH, NAME, START_DATE, END_DATE, MAX_DAYS, TEST_LEN_SHARE)

