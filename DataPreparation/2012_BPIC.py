from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter
import pandas as pd
from create_benchmarks import remainTimeOrClassifBenchmark, nextEventBenchmark

NAME = "BPI_Challenge_2012"
PATH = "./"
START_DATE = None 
END_DATE = "2012-02"
MAX_DAYS = 32.28
TEST_LEN_SHARE = .05

KEYWORDS_DICT = {}
KEYWORDS_DICT["approved"] = ["A_REGISTERED_COMPLETE", "A_APPROVED_COMPLETE", "O_ACCEPTED_COMPLETE", "A_ACTIVATED_COMPLETE"]
KEYWORDS_DICT["declined"] = ["A_DECLINED_COMPLETE", "O_DECLINED_COMPLETE"]
KEYWORDS_DICT["canceled"] = ["A_CANCELLED_COMPLETE"]


log = xes_importer.apply(PATH + "/" + NAME + ".xes")
dataset = converter.apply(log,variant=converter.Variants.TO_DATA_FRAME)

# remaining time and classification
dataset["classif_target"] = dataset["concept:name"] + "_" + dataset["lifecycle:transition"]
remainTimeOrClassifBenchmark(dataset, PATH, NAME, START_DATE, END_DATE, MAX_DAYS, TEST_LEN_SHARE, "xes", KEYWORDS_DICT)


