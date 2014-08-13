import csv
import json


x = open('test.json')


x = json.load(x)

f = csv.writer(open("test.csv", "wb+"))

# Write CSV Header, If you dont need that, remove this line
f.writerow(["SENSOR", "RESOLUTION", "POWER", "NAME", "VERSION", "TYPE",
            "MAXIMUM_RANGE","VENDOR","PROBE","ACCURACY", "TIMESTAMP",
            "LUX", "EVENT_TIMESTAMP",])

for x in x:
    f.writerow([x["SENSOR"]["RESOLUTION"], 
                x["SENSOR"]["POWER"], 
                x["SENSOR"]["NAME"], 
                x["SENSOR"][ "VERSION"],
                x["SENSOR"]["TYPE"],
                x["PROBE"],
                x["ACCURACY"],
                x["TIMESTAMP"],
                x["LUX"],
                x["EVENT_TIMESTAMP"]])
