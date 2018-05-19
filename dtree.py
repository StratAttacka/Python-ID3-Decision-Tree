# Ian Santillano
# COEN166
# 5/17/18
# Implementation and Testing of ID3 decision tree:
# Tree is created using ID3 algorithm and finding out  
# which attribute will yield highest gain. Used to train 
# and be tested on multiple classification datasets.

from math import log
import random

# For the bonus drawing tree functions at the bottom
# import pydot

# DATA: The actual data sets that I train/test on
dataGolf = [
  ["Sunny","Hot","High","Weak","No"],
  ["Sunny","Hot","High","Strong","No"],
  ["Overcast","Hot","High","Weak","Yes"],
  ["Rain","Mild","High","Weak","Yes"],
  ["Rain","Cool","Normal","Weak","Yes"],
  ["Rain","Cool","Normal","Strong","No"],
  ["Overcast","Cool","Normal","Strong","Yes"],
  ["Sunny","Mild","High","Weak","No"],
  ["Sunny","Cool","Normal","Weak","Yes"],
  ["Rain","Mild","Normal","Weak","Yes"],
  ["Sunny","Mild","Normal","Strong","Yes"],
  ["Overcast","Mild","High","Strong","Yes"],
  ["Overcast","Hot","Normal","Weak","Yes"],
  ["Rain","Mild","High","Strong","No"]
]

# Used to process large datasets stored on file and store them in
# in a 2d list
def processData(data):
    dataList = []
    with open(data,"r") as f:
        for line in f:
            if data == "data/mush.data" or data == "data/bal.data":
                a = line.strip().split(',')
                b = a[0]
                a = a[1:]
                a.extend(b)
                dataList.append(a)
            else:
                dataList.append(line.strip().split(','))
    return dataList

# Process data and split into train and test sets
dataTic = processData("data/tic.data")
dataCar = processData("data/car.data")
dataMush = processData("data/mush.data")

# Process data and split into train and test sets


# EXTRA DATASETS 
# dataIris = processData("data/iris.data")
# dataBal = processData("data/bal.data")
# 
# lensTrain = [
#         [1,1,1,1,3],
#         [2,1,1,2,2],
#         [3,1,1,1,3],
#         [1,1,1,2,2],
#         [2,1,2,1,3],
#         [3,1,1,2,3],
#         [1,1,2,1,3],
#         [2,1,2,2,1],
#         [3,1,2,1,3],
#         [1,1,2,2,1],
#         [2,2,1,1,3],
#         [3,1,2,2,1],
#         [1,2,1,1,3],
#         [2,2,1,2,2],
#         [3,2,1,1,3],
#         [1,2,1,2,2],
#         [2,2,2,1,3],
#         [3,2,1,1,3],
#         [1,2,2,1,3],
        
#     ]

# lensTest = [
#         [2,2,2,2,3],
#         [3,2,1,2,2],
#         [1,2,2,2,1],
#         [2,1,1,1,3],
#         [3,2,2,1,3],
#         [3,1,1,1,3],  
# ]

# for i in range(len(lensTest)):
#     for j in range(len(lensTest[i])):
#         lensTest[i][j] = str(lensTest[i][j])

# for i in range(len(lensTrain)):
#     for j in range(len(lensTrain[i])):
#         lensTrain[i][j] = str(lensTrain[i][j])

# cryoTrain = [
#         ["1","35","12","5","1","100","0"  ],
#         ["1","29","7","5","1","96","1"  ],
#         ["1","50","8","1","3","132","0"  ],
#         ["1","32","11.75","7","3","750","0"  ],
#         ["1","67","9.25","1","1","42","0"  ],
#         ["1","41","8","2","2","20","1"  ],
#         ["1","36","11","2","1","8","0"  ],
#         ["1","59","3.5","3","3","20","0"  ],
#         ["1","20","4.5","12","1","6","1"  ],
#         ["2","34","11.25","3","3","150","0"  ],
#         ["2","21","10.75","5","1","35","0"  ],
#         ["2","15","6","2","1","30","1"  ],
#         ["2","15","2","3","1","4","1"  ],
#         ["2","15","3.75","2","3","70","1"  ],
#         ["2","17","11","2","1","10","0"  ],
#         ["2","17","5.25","3","1","63","1"  ],
#         ["2","23","11.75","12","3","72","0"  ],
#         ["2","27","8.75","2","1","6","0"  ],
#         ["2","15","4.25","1","1","6","1"  ],
#         ["2","18","5.75","1","1","80","1"  ],
#         ["1","22","5.5","2","1","70","1"  ],
#         ["2","16","8.5","1","2","60","1"  ],
#         ["1","28","4.75","3","1","100","1"  ],
#         ["2","40","9.75","1","2","80","0"  ],
#         ["1","30","2.5","2","1","115","1"  ],
#         ["2","34","12","3","3","95","0"  ],
#         ["1","20","0.5","2","1","75","1"  ],
#         ["2","35","12","5","3","100","0"  ],
#         ["2","24","9.5","3","3","20","0"  ],
#         ["2","19","8.75","6","1","160","1"  ],
#         ["1","35","9.25","9","1","100","1"  ],
#         ["1","29","7.25","6","1","96","1"  ],
#         ["1","50","8.75","11","3","132","0"  ],
#         ["2","32","12","4","3","750","0"  ],
#         ["2","67","12","12","3","42","0"  ],
#         ["2","41","10.5","2","2","20","1"  ],
#         ["2","36","11","6","1","8","0"  ],
#         ["1","63","2.75","3","3","20","0"  ],
#         ["1","20","5","3","1","6","1"  ],
#         ["1","34","12","1","3","150","0"  ],
#         ["2","21","10.5","5","1","35","0"  ],
#         ["2","15","8","12","1","30","1"  ],
#         ["1","15","3.5","2","1","4","1"  ],
#         ["2","15","1.5","12","3","70","1"  ],
#         ["1","17","11.5","2","1","10","0"  ],
#         ["1","17","5.25","4","1","63","1"  ],
#         ["2","23","9.5","5","3","72","0"  ],
#         ["1","27","10","5","1","6","0"  ],
#         ["1","15","4","7","1","6","1"  ],
#         ["2","18","4.5","8","1","80","1"  ],
#         ["2","22","5","9","1","70","1"  ],
#         ["1","16","10.25","3","2","60","1"  ],
#         ["2","28","4","11","1","100","1"  ],
#         ["2","40","8.75","6","2","80","0"  ],
#         ["2","30","0.5","8","3","115","1"  ],
#         ["1","34","10.75","1","3","95","0"  ],
#         ["1","20","3.75","11","1","75","1"  ],
#         ["2","35","8.5","6","3","100","0"  ],
#         ["1","24","9.5","8","1","20","1"  ],
#         ["2","19","8","9","1","160","1"  ],
#         ["1","35","7.25","2","1","100","1"  ],
#         ["1","29","11.75","5","1","96","0"  ],
#         ["2","50","9.5","4","3","132","0"  ],
#         ["2","32","12","12","3","750","0"  ],
#         ["1","67","10","7","1","42","0"  ],
#         ["2","41","7.75","5","2","20","1"  ],
#         ["2","36","10.5","4","1","8","0"  ],
#         ["1","67","3.75","11","3","20","0"  ],
#         ["1","20","4","3","1","6","1"  ],
#         ["1","34","11.25","1","3","150","0"  ],
#         ["2","21","10.75","7","1","35","0"  ],
#         ["1","15","10.5","11","1","30","1"  ],
#         ["1","15","2","11","1","4","1"  ],
#         ["2","15","2","10","3","70","1"  ],
#         ["1","17","9.25","12","1","10","0"  ],
#         ["1","17","5.75","10","1","63","1"  ],  
#     ]

# cryoTest = [
# ["1","23","10.25","7","3","72","0"  ],
# ["1","27","10.5","7","1","6","0"  ],
# ["1","15","5.5","5","1","6","1"  ],
# ["1","18","4","1","1","80","1"  ],
# ["2","22","4.5","2","1","70","1"  ],
# ["1","16","11","3","2","60","1"  ],
# ["2","28","5","9","1","100","1"  ],
# ["1","40","11.5","9","2","80","0"  ],
# ["1","30","0.25","10","1","115","1"  ],
# ["2","34","12","3","3","95","0"  ],
# ["2","20","3.5","6","1","75","1"  ],
# ["2","35","8.25","8","3","100","0"  ],
# ["1","24","10.75","10","1","20","1"  ],
# ["1","19","8","8","1","160","1"  ]
# ]

# imunoTrain = [
# ["1","22","2.25","14","3","51","50","1"  ],
# ["1","15","3","2","3","900","70","1"  ],
# ["1","16","10.5","2","1","100","25","1"  ],
# ["1","27","4.5","9","3","80","30","1"  ],
# ["1","20","8","6","1","45","8","1"  ],
# ["1","15","5","3","3","84","7","1"  ],
# ["1","35","9.75","2","2","8","6","1"  ],
# ["2","28","7.5","4","1","9","2","1"  ],
# ["2","19","6","2","1","225","8","1"  ],
# ["2","32","12","6","3","35","5","0"  ],
# ["2","33","6.25","2","1","30","3","1"  ],
# ["2","17","5.75","12","3","25","7","1"  ],
# ["2","15","1.75","1","2","49","7","0"  ],
# ["2","15","5.5","12","1","48","7","1"  ],
# ["2","16","10","7","1","143","6","1"  ],
# ["2","33","9.25","2","2","150","8","1"  ],
# ["2","26","7.75","6","2","6","5","1"  ],
# ["2","23","7.5","10","2","43","3","1"  ],
# ["2","15","6.5","19","1","56","7","1"  ],
# ["2","26","6.75","2","1","6","6","1"  ],
# ["1","22","1.25","3","3","47","3","1"  ],
# ["2","19","2.25","2","1","60","7","1"  ],
# ["2","26","10.5","6","1","50","9","0"  ],
# ["1","25","5.75","2","1","300","7","1"  ],
# ["2","17","11.25","4","3","70","7","1"  ],
# ["1","27","5","2","1","20","5","1"  ],
# ["2","24","4.75","10","3","30","45","1"  ],
# ["1","15","11","6","1","30","25","0"  ],
# ["2","34","11.5","12","1","25","50","0"  ],
# ["2","20","7.75","18","3","45","2","1"  ],
# ["2","38","2.5","1","3","43","50","1"  ],
# ["1","23","3","2","3","87","70","1"  ],
# ["2","48","10.25","7","1","50","25","1"  ],
# ["2","24","4.25","1","1","174","30","1"  ],
# ["2","33","8","3","1","502","8","1"  ],
# ["1","34","5","7","3","64","7","0"  ],
# ["2","41","11","11","2","21","6","0"  ],
# ["1","29","8.75","3","1","504","2","1"  ],
# ["2","22","8.5","5","1","99","8","1"  ],
# ["1","45","11.25","4","1","72","5","0"  ],
# ["2","22","8.25","9","1","352","3","1"  ],
# ["1","35","8.75","10","2","69","7","1"  ],
# ["2","34","8.5","1","2","163","7","0"  ],
# ["1","49","4.5","2","1","33","7","0"  ],
# ["2","19","11","5","2","51","6","1"  ],
# ["1","21","8","3","1","17","8","1"  ],
# ["1","26","7.75","13","2","13","5","1"  ],
# ["1","51","8.75","2","2","57","3","1"  ],
# ["1","19","7.75","6","1","32","7","1"  ],
# ["1","38","12","14","1","87","6","0"  ],
# ["2","36","1.75","10","3","45","3","1"  ],
# ["2","52","2.25","5","1","63","7","1"  ],
# ["2","49","9","4","2","14","9","1"  ],
# ["1","23","5.75","2","1","43","7","1"  ],
# ["1","45","10","8","1","58","7","1"  ],
# ["1","54","7.5","13","3","43","5","1"  ],
# ["2","47","5.25","3","3","23","45","1"  ],
# ["2","53","10","1","2","30","25","1"  ],
# ["2","56","11.75","7","1","31","50","0"  ],
# ["1","27","11.25","3","2","37","2","1"  ],
# ["2","47","3.75","14","2","67","50","1"  ],
# ["2","19","2.25","8","2","42","70","1"  ],
# ["2","33","8","5","1","63","25","1"  ],
# ["2","15","4","12","1","72","30","1"  ],
# ["1","17","8.5","2","1","44","8","1"  ],
# ["1","29","5","12","3","75","7","1"  ],
# ["1","27","11.75","8","1","208","6","0"  ],
# ["2","51","6","6","1","80","2","1"  ],
# ["1","35","6.75","4","3","41","8","1"  ],
# ["2","47","10.75","8","1","57","5","0"  ],
# ["1","43","8","1","1","59","3","1"  ],
# ["1","15","4","4","3","25","7","1"  ],
# ["1","33","1.75","7","2","379","7","0"  ],
# ["2","51","4","1","1","65","7","1"  ],
# ["1","45","6.5","9","2","49","6","1"  ],
# ["1","18","11.75","5","2","13","5","1"  ],
# ["2","46","7.75","8","1","40","3","1"  ],
# ["1","43","11","7","1","507","7","1"  ],
# ["2","28","11","3","3","91","6","0"  ],
# ]

# imunoTest = [
# ["2","28","11","3","3","91","6","0"  ],
# ["1","30","1","2","1","88","3","1"  ],
# ["2","16","2","11","1","47","7","1"  ],
# ["2","42","8.75","8","2","73","9","0"  ],
# ["2","15","8","1","1","55","7","1"  ],
# ["2","53","7.25","6","1","81","7","1"  ],
# ["1","40","5.5","8","3","69","5","1"  ],
# ["1","38","7.5","8","2","56","45","1"  ],
# ["1","38","7.5","8","2","56","45","1"  ],
# ["1","38","7.5","8","2","56","45","1"  ],
# ["1","46","11.5","4","1","91","25","0"  ],
# ["1","32","12","9","1","43","50","0"  ],
# ["2","23","6.75","6","1","19","2","1"  ]
# ]

# postopTest = [
#   ["mid","high","excellent","high","stable","stable","stable","10","S"  ],
#   ["high","low","excellent","high","stable","stable","mod-stable","10","A"  ],
#   ["mid","low","good","high","stable","unstable","mod-stable","15","A "  ],
#   ["mid","mid","excellent","high","stable","stable","stable","10","A"  ],
#   ["high","low","good","mid","stable","stable","unstable","15","S"  ],
#   ["mid","low","excellent","high","stable","stable","mod-stable","05","S"  ],
#   ["high","mid","excellent","mid","unstable","unstable","stable","10","S"  ],
#   ["mid","high","good","mid","stable","stable","stable","10","S"  ],
#   ["mid","low","excellent","mid","unstable","stable","mod-stable","10","S"  ],
#   ["mid","mid","good","mid","stable","stable","stable","15","A"  ],
#   ["mid","low","good","high","stable","stable","mod-stable","10","A"  ],
#   ["high","high","excellent","high","unstable","stable","unstable","15","A"  ],
#   ["mid","high","good","mid","unstable","stable","mod-stable","10","A"  ],
#   ["mid","low","good","high","unstable","unstable","stable","15","S"  ],
#   ["high","high","excellent","high","unstable","stable","unstable","10","A"  ],
#   ["low","high","good","high","unstable","stable","mod-stable","15","A"  ],
#   ["mid","low","good","high","unstable","stable","stable","10","A"  ],
#   ["mid","high","good","mid","unstable","stable","unstable","15","A"  ],
#   ["mid","mid","good","mid","stable","stable","stable","10","A"  ],
#   ["low","high","good","mid","unstable","stable","stable","15","A"  ],
#   ["low","mid","excellent","high","unstable","stable","unstable","10","S"  ],
#   ["mid","mid","good","mid","unstable","stable","unstable","15","A"  ],
#   ["mid","mid","good","mid","unstable","stable","stable","10","A"  ],
#   ["high","high","good","mid","stable","stable","mod-stable","10","A"  ],
#   ["low","mid","good","mid","unstable","stable","stable","10","A"  ],
#   ["high","mid","good","low","stable","stable","mod-stable","10","A"  ],
#   ["low","mid","excellent","high","stable","stable","mod-stable","10","A"  ],
#   ["mid","mid","excellent","mid","stable","stable","unstable","15","A"  ],
#   ["mid","mid","good","mid","unstable","stable","unstable","10","S"  ],
#   ["mid","mid","good","high","unstable","stable","stable","10","A"  ],
#   ["low","low","good","mid","unstable","stable","unstable","10","A"  ],
#   ["mid","mid","excellent","high","unstable","stable","mod-stable","10","A"  ],
#   ["mid","low","good","mid","stable","stable","stable","10","A"  ],
#   ["low","mid","excellent","high","stable","stable","mod-stable","10","A"  ],
#   ["mid","mid","good","mid","stable","stable","stable","10","A"  ],
#   ["low","mid","excellent","mid","stable","stable","stable","10","S"  ],
#   ["low","low","good","mid","unstable","stable","unstable","10","S"  ],
#   ["low","low","good","mid","stable","stable","stable","07","S"  ],
#   ["mid","mid","good","high","unstable","stable","mod-stable","10","A"  ],
#   ["low","low","good","mid","unstable","stable","stable","10","A"  ],
#   ["low","mid","good","mid","stable","stable","stable","15","S"  ],
#   ["high","high","good","high","unstable","stable","stable","15","S"  ],
#   ["mid","mid","good","mid","stable","stable","stable","10","S"  ],
#   ["low","low","excellent","mid","stable","stable","stable","10","A"  ],
#   ["low","mid","good","mid","unstable","stable","stable","10","S"  ],
#   ["low","mid","good","high","unstable","stable","stable","?","I"  ],
#   ["mid","mid","excellent","mid","unstable","stable","stable","10","A"  ],
#   ["high","high","excellent","high","stable","stable","unstable","?","A"  ],
#   ["mid","high","good","low","unstable","stable","stable","10","A"  ],
#   ["mid","high","good","mid","unstable","mod-stable","mod-stable","10","A"  ],
#   ["low","high","excellent","mid","unstable","stable","stable","10","A"  ],
#   ["mid","low","excellent","high","unstable","stable","unstable","10","A"  ],
#   ["mid","mid","good","mid","unstable","stable","mod-stable","10","S"  ],
#   ["high","high","excellent","mid","unstable","stable","mod-stable","10","A"  ],
#   ["mid","mid","good","mid","unstable","stable","stable","15","A"  ],
#   ["high","mid","good","high","stable","stable","unstable","15","A"  ],
#   ["mid","low","good","high","unstable","stable","mod-stable","10","A"  ],
#   ["low","low","good","high","stable","stable","stable","10","A"  ],
#   ["mid","high","good","mid","stable","stable","mod-stable","10","A"  ],
#   ["mid","high","good","mid","unstable","stable","unstable","10","A"  ],
#   ["mid","low","excellent","high","stable","stable","stable","10","A"  ],
#   ["mid","mid","good","mid","stable","stable","unstable","10","A"  ],
#   ["mid","low","excellent","mid","stable","stable","unstable","10","S"  ],
#   ["high","mid","excellent","mid","unstable","unstable","unstable","10","A"  ],
#   ["mid","mid","good","high","stable","stable","stable","10","S"  ],
#   ["mid","low","excellent","mid","unstable","stable","stable","10","A"  ],
#   ["mid","mid","excellent","mid","unstable","stable","stable","10","A"  ],
#   ["mid","mid","excellent","high","stable","stable","stable","10","A"  ],
#   ["mid","mid","excellent","low","stable","stable","stable","10","A"  ],
#   ["mid","low","excellent","mid","unstable","unstable","unstable","?","A"  ],
#   ["low","low","excellent","mid","stable","stable","stable","10","A"  ],
#   ["mid","mid","excellent","mid","stable","stable","mod-stable","10","S"  ],
#   ["mid","mid","excellent","high","stable","stable","stable","10","A"  ],
#   ["mid","low","excellent","high","stable","stable","mod-stable","10","A"  ],
#   ["low","mid","good","mid","stable","stable","unstable","10","A"  ],
#   ["mid","mid","excellent","mid","stable","stable","mod-stable","10","A"  ],
#   ["mid","mid","excellent","mid","stable","stable","unstable","10","A"  ],
#   ["mid","mid","excellent","mid","unstable","unstable","stable","10","S"  ],
#   ["mid","mid","good","high","stable","stable","stable","10","A"  ],
#   ["mid","mid","excellent","mid","stable","stable","stable","15","A"  ],
#   ["mid","mid","excellent","mid","stable","stable","stable","10","S"  ],
#   ["mid","low","good","mid","stable","stable","unstable","10","I"  ],
#   ["high","mid","excellent","mid","unstable","stable","unstable","05","A"  ],
#   ["mid","mid","excellent","mid","stable","stable","unstable","10","A"  ],
#   ["mid","mid","excellent","mid","unstable","stable","stable","10","A"  ],
#   ["mid","mid","excellent","mid","unstable","stable","stable","15","S"  ],
#   ["mid","mid","good","mid","unstable","stable","stable","15","A"  ],
#   ["mid","mid","excellent","mid","unstable","stable","stable","10","A"  ],
#   ["mid","mid","good","mid","unstable","stable","stable","15","S"  ],
#   ["mid","low","excellent","mid","unstable","unstable","unstable","?","A"  ],
#   ["low","low","excellent","mid","stable","stable","stable","10","A"  ],
#   ["mid","mid","excellent","mid","stable","stable","mod-stable","10","S"  ],
#   ["mid","mid","excellent","high","stable","stable","stable","10","A"  ],
#   ["mid","low","excellent","high","stable","stable","mod-stable","10","A"  ],
#   ["low","mid","good","mid","stable","stable","unstable","10","A"  ],
#   ["mid","mid","excellent","mid","stable","stable","mod-stable","10","A"  ],
#   ["mid","mid","excellent","mid","stable","stable","unstable","10","A"  ],
#   ["mid","mid","excellent","mid","unstable","unstable","stable","10","S"  ],
#   ["mid","mid","good","high","stable","stable","stable","10","A"  ],
#   ["mid","mid","excellent","mid","stable","stable","stable","15","A"  ],
#   ["mid","mid","excellent","mid","stable","stable","stable","10","S"  ],
#   ["mid","low","good","mid","stable","stable","unstable","10","I"  ],
#   ["high","mid","excellent","mid","unstable","stable","unstable","05","A"  ],
#   ["mid","mid","excellent","mid","stable","stable","unstable","10","A"  ],
#   ["mid","mid","excellent","mid","unstable","stable","stable","10","A"  ],
#   ["mid","mid","excellent","mid","unstable","stable","stable","15","S"  ],
#   ["mid","mid","good","mid","unstable","stable","stable","15","A"  ],
#   ["mid","mid","excellent","mid","unstable","stable","stable","10","A"  ],
#   ["mid","mid","good","mid","unstable","stable","stable","15","S"  ]
# ]


# ATTRIBUTES: Attribute Lists for the different datasets I train/test on
attributesGolf = ["Outlook","Temperature","Humidity","Wind"]
attributesTic = ["TL", "TM", "TR", "ML", "MM", "MR", "BL", "BM", "BR"]
attributesCar = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
attributesMush = ["cap-shape", "cap-surface", "cap-color", "bruises?", "odor", "gill-attachment", "gill-spacing",
                    "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
                    "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color", "ring-number", "ring-type",
                    "spore-print-color", "population", "habitat"]

# EXTRA ATTRIBUTES
# attributesIris = ["sepal-length", "sepal-width", "petal-length", "petal-width"]
# attributesBal = ["Left-Weight", "Left-Distance", "Right-Weight", "Right-Distance"]
# attributesLens = ["age", "sp", "ast", "tpr"]
# attributesCryo = ["sex","age","Time","Number_of_Warts","Type","Area","Result_of_Treatment" ]
# attributesImuno = ["sex","age","Time","Number_of_Warts","Type","Area","induration_diameter","Result_of_Treatment"  ]
# attributesPost = ["L-CORE", "L-SURF", "L-02", "L-BP", "SURF-STBL", "CORE-STBL", "BP-STBL", "COMFORT"]

# get the different class values for a datasets class attribute
def getClasses(data):
    classes = {}
    for entry in data:
        if entry[-1] not in classes:
            classes[entry[-1]] = 1
        else:
            classes[entry[-1]] += 1
    return classes

# calculate entropy of a given dataset
def calcEntropy(data):
    classes = getClasses(data)
    entropy = 0.0
    for key in classes:
        entropy += -float((classes[key]/(len(data)))) * log((classes[key]/(len(data))), 2)
    return entropy

# get different values for a given attribute
def getAttributeValsF(data, attribute):
    attributeValues = {}
    for entry in data:
        e = entry[attribute]
        if e not in attributeValues:
            attributeValues[e] = 1
        else:
            attributeValues[e] += 1
    return attributeValues

# Get all the unique attribute values for a given attribute, will be used when building the tree
def getUniqueAttributesVals(data, attribute):
    values = {}
    for entry in data:
        if entry[attribute] not in values:
            values[entry[attribute]] = 1
    return values

# calculate information gain relative to an attribute, will be used in determining the best attribute to split on
def calcInfoGain(data, attribute):
    newData = []
    count = 0
    systemEntropy = calcEntropy(data)
    attributeValues = getAttributeValsF(data, attribute)
    infoGain = 0.0
    for key in attributeValues:
        for entry in data:
            if entry[attribute] == key:
                count += 1
                newData.append([entry[attribute], entry[-1]])
        infoGain += count/len(data) * calcEntropy(newData)
        count = 0
        del newData[:] 
    infoGain = systemEntropy - infoGain
    return infoGain

# split the data into subdata for a unique value in a given attribute
# ALDO for instance, if split on OUTLOOK, could split on OUTLOOK, SUNNY
# which would produce a dataset with all instances of SUNNY
def split(data, attribute, attributeValue):
    newData = []
    for entry in data:
        if entry[attribute] == attributeValue:
            toBeAdded = entry[:attribute]
            toBeAdded.extend(entry[attribute+1:])
            newData.append(toBeAdded)
    # for i in newData:
    #   print(newData)
    return newData

# choose the best attribute according to which has the highest information gain
def chooseBestAttribute(data):
    numAttributes = len(data[0]) - 1
    bestInfoGain = 0.0
    bestAttribute = -1
    for i in range(numAttributes):
        currInfoGain = calcInfoGain(data, i)
        if currInfoGain > bestInfoGain:
            bestInfoGain = currInfoGain
            bestAttribute = i
    # print(bestAttribute, bestInfoGain)
    return bestAttribute

# Actual algorithm to create the decision tree
# Tree is represented as a nested dictionary where the key
# is a node and its value are branches coming out of that 
# node.
def createTree(data, attributes):
    classes = getClasses(data)
    for key in classes:                             # base case: if all examples are the same class
        if classes[key] == len(data):
            return key
    bestAtt = chooseBestAttribute(data)
    bestAttLabel = attributes[bestAtt]
    tree = {bestAttLabel:{}}
    del(attributes[bestAtt])                        # determined where to split the data, will not use this attribute in the future
    vals = getUniqueAttributesVals(data, bestAtt)
    for key in vals:                                # for distinct value for the best attribute to split on, create a new tree
        subTrees = attributes[:]
        tree[bestAttLabel][key] = createTree(split(data, bestAtt, key),subTrees) 
    return tree

# Used to get the branch values for a given attribute node in a tree
# For instance in aldo example, this could return:
# {"sunny":{...}, "overcast":{...}, "rain":{...}}
def getBranches(tree):
    for key, value in tree.items():
        return value

# Used for classifying some test data entry using a decision tree
def classify(testData, attributes, tree):
    data = testData
    if isinstance(tree, dict):
        for key, value in tree.items():
            if testData[attributes.index(key)] in getBranches(tree):
                return classify(testData, attributes, getBranches(tree)[testData[attributes.index(key)]])
    else:
        return tree

# Used to actually test the accuracy of the ID3 decision tree that was generated on the training data
def dataTest(f, tree, attributes):
    correct = 0
    total = 0
    treeOrig = tree
    for entry in f:
        
        total += 1
        actual = entry[-1]
        testing = entry[:-1]
        result = classify(testing, attributes, tree)
        if result == actual:
            correct += 1
        # if total == 2:
        #   break
        tree = treeOrig

    return correct/total

# 1. Splits data into train and test data
# 2. Creates tree on trainset
# 3. Tests tree on trainset (to validate 100% accuracy)
# 4. Tests tree on testset
# 5. Calculates error
# Other notes: seed is used to seed the data so that
# a certain train and test combination can be used.
def trainTest(name, data, attributes, seed, testPercentage):
    dataset = trainTestSplitter(data, testPercentage, seed)
    ogTree = createTree(dataset["train"][:], attributes[:])
    Tree = ogTree
    trainAcc = dataTest(dataset["train"], Tree, attributes)
    testAcc = dataTest(dataset["test"], Tree, attributes)
    print("Seed \"" + str(seed) + "\" used to seed \"" + name + "\" Train and Test Set")
    print("Train results for {}: ".format(name) + str(trainAcc))
    print("Test results for {}: ".format(name) + str(testAcc))
    print("Error between train and test =", "{0:.2f}".format(trainAcc-testAcc))
    return ogTree

# Test over multiple times and different train and test combinations 
# (random seed will give us different train and test sets from same
# data) and get an average accuracy over those times.
def getAvgAccuracy(data, attributes, times, seed, testPercentage):
    total = 0.0
    for i in range(times):
        dataset = trainTestSplitter(data, testPercentage, seed)
        tree = createTree(dataset["train"][:], attributes[:])
        total += dataTest(dataset["test"], tree, attributes)
        dataset.clear()
    return total/times

# Splits dataset into shuffled train and test subdatasets based on a test percentage amount
# Can seed the data so that you get a certain combination of test and data. 
def trainTestSplitter(data, testPercentage, seed):
    random.seed(seed)
    random.shuffle(data)
    dataset = {}
    testBound = len(data) - int(testPercentage * len(data))
    dataset["train"] = data[:testBound]
    dataset["test"] = data[testBound:]
    return dataset

# PREDICTIONS

# Used to seed the random sequence incase I find a good train/test combination since
# the splitting up of data is random (shuffled and then split via a percentage)
# Essentially I can "save" a trained tree by remembering its seed
seed = random.randint(0,100000)
print("\n*****TESTING*****")
# print("SEED:", seed)

# First just a quick train and test and then a test with shuffled/random
# train and test data over TIMES times. TESTPERCENTAGE is how much of the shuffled data
# is taken away from the raw data and used for testing on a tree
TIMES = 1000
TESTPERCENTAGE = .24

print("Raw Data broken into " + str("{0:.0f}".format((1-TESTPERCENTAGE) * 100)) + "% training set and " + str("{0:.0f}".format(TESTPERCENTAGE * 100)) + "% testing set")
print()

# TIC: "best" seed 4157
print("TIC TAC TOE:")
trainTest("tic", dataTic, attributesTic, 4157, TESTPERCENTAGE)
print("Avg Accuracy for tic test: over {} times".format(TIMES), getAvgAccuracy(dataTic, attributesTic, TIMES, seed, TESTPERCENTAGE))
print()

# CAR: "best" seed 87620
print("CAR:")
trainTest("car", dataCar, attributesCar, 87620, TESTPERCENTAGE)
print("Avg Accuracy for car test: over {} times".format(TIMES), getAvgAccuracy(dataCar, attributesCar, TIMES, seed, TESTPERCENTAGE))
print()

TIMES = 20
# MUSH:
print("MUSHROOM:")
trainTest("mush", dataMush, attributesMush, seed, TESTPERCENTAGE)
print("Avg Accuracy for mushroom test: over {} times".format(TIMES), getAvgAccuracy(dataMush, attributesMush, TIMES, seed, TESTPERCENTAGE))
print()


# Extra functions for drawing a visual representation of the tree and exporting it to a png file
# def draw(parent_name, child_name):
#     edge = pydot.Edge(parent_name, child_name)
#     graph.add_edge(edge)

# def visit(node, parent=None):
#     for k,v in node.items():
#         if isinstance(v, dict):
#             # We start with the root node whose parent is None
#             # we don't want to graph the None node
#             if parent:
#                 draw(parent, k)
#             visit(v, k)
#         else:
#             draw(parent, k)
#             # drawing the label using a distinct name
#             draw(k, v)

# graph = pydot.Dot(graph_type='graph')
# visit(treeGolf)
# graph.write_png('golf.png')

# # graph = pydot.Dot(graph_type='graph')
# # visit(treeLens)
# # graph.write_png('lens.png')

# # graph = pydot.Dot(graph_type='graph')
# # visit(treeCryo)
# # graph.write_png('cryo.png')

# # graph = pydot.Dot(graph_type='graph')
# # visit(treeImuno)
# # graph.write_png('imuno.png')


