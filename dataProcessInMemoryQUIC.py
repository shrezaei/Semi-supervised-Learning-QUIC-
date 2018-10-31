#It first move the whole file to memory and then work with memory
#Good for sampling methods than jumps alot
import pdb
import os
import numpy as np
import random

# Defined by Shahbaz
timestep = 45
ComputeInterArrival = True
DescretesizeLength = False
DirectionLengthCombined = True  #If true, direction will be represented by positive and negative sign in packet length
NormalizeLength = True
NormalizeInterArrival = True
MaxLength = 1434
MaxInterArrival = 1
Starting_point = 0     #Indicates how many packets from the begining of the entire flow you want to skip
StartingPointMultiply = 13  #start from multiply of this number
Num_of_extracted_subflow = 100   #Number of subflows that are goining to be extracted from each flow if possible
PaddingEnable = True
PadAtTheBegining = True
PaddingThreshold = 8    #The flow must have at least this amount of packets to be padded


IncrementalSampling = False  #Cannot be both True
RandomSampling = False

CompureStatisticsInThisScript = True

TrainingSetSize = 10
NumOfCrossValidationFolds = 5

np.random.seed(20)

input_size = 2
Channel_size = input_size


#CS5-3 (Modified)
#timestep = 75
#SkipPacketsForSampling = 13
#IncrementalSampling = True
#NumberOfSamplesUntiIncrement = 10
#IncrementalStepMultiplier = 1 

#For our QUIC dataset

#Cover up to 1000 packets
#CS5-3 (Simple Sampling)
timestep = 45
SkipPacketsForSampling = 22
IncrementalSampling = True
NumberOfSamplesUntiIncrement = 10
IncrementalStepMultiplier = 1 

#CS5-3-0.15 (Modified) (Random Sampling)
#timestep = 45
#Starting_point = 0     #Indicates how many packets from the begining of the entire flow you want to skip
#StartingPointMultiply = 0  #start from multiply of this number
#SkipPacketsForSampling = 1
#SamplingProbability = 1/22
#IncrementalSampling = False
#RandomSampling = True
#NumberOfSamplesUntiIncrement = 10
#IncrementalStepMultiplier = 1 

#CS5-4-8-1.2	(Incremental Sampling)
#timestep = 45
#SkipPacketsForSampling = 22
#IncrementalSampling = True
#RandomSampling=False
#NumberOfSamplesUntiIncrement = 10
#IncrementalStepMultiplier = 1.6


##### For Waikato Dataset #########

#CS2-3	(Simple Sampling)
#timestep = 45
#SkipPacketsForSampling = 10
#IncrementalSampling = True
#NumberOfSamplesUntiIncrement = 10
#IncrementalStepMultiplier = 1 

#CS5-4-8-1.2	(Incremental Sampling)
#timestep = 45
#SkipPacketsForSampling = 8
#IncrementalSampling = True
#NumberOfSamplesUntiIncrement = 10
#IncrementalStepMultiplier = 1.2

##CS5-3-0.15 (Random Sampling)
#timestep = 45
#Starting_point = 0     #Indicates how many packets from the begining of the entire flow you want to skip
#StartingPointMultiply = 0  #start from multiply of this number
#SkipPacketsForSampling = 1
#SamplingProbability = 0.15
#IncrementalSampling = False
#RandomSampling = True
#NumberOfSamplesUntiIncrement = 10
#IncrementalStepMultiplier = 1 

def convertDataUnsupervised(data, labels, statlabels):
    # shuffle the data
    num_samples = labels.shape[0]
    assert num_samples == data.shape[0]
    perm = np.arange(num_samples)
    np.random.shuffle(perm)

    labels = labels[perm]
    data = data[perm]
    statlabels = statlabels[perm]

    #Convert flat features into 2D input for CNN
    temp = np.zeros((num_samples,Channel_size,timestep))
    for i in range(num_samples):
        t = data[i].reshape((timestep,input_size))
        t = np.swapaxes(t,0,1)
        temp[i] = t
    data = temp

    unsupervised_train = np.arange(num_samples)
    unsupervised_train = perm[unsupervised_train]

    unsuper_trainData = data[unsupervised_train, :, :timestep]
    unsuper_trainLabel = statlabels[unsupervised_train]

    return (unsuper_trainData, unsuper_trainLabel)


def convertDataSupervised(superdata, superlabels, supertestdata, supertestlabels):
    # shuffle the data
    num_samples2 = superlabels.shape[0]
    num_samples3 = supertestlabels.shape[0]
    assert num_samples2 == superdata.shape[0]
    perm2 = np.arange(num_samples2)
    perm3 = np.arange(num_samples3)
    np.random.shuffle(perm2)
    np.random.shuffle(perm3)

    superlabels = superlabels[perm2]
    superdata = superdata[perm2]
    supertestlabels = supertestlabels[perm3]
    supertestdata = supertestdata[perm3]

    #Convert flat features into 2D input for CNN
    temp = np.zeros((num_samples2,Channel_size,timestep))
    for i in range(num_samples2):
        t = superdata[i].reshape((timestep,input_size))
        t = np.swapaxes(t,0,1)
        temp[i] = t
    superdata = temp

    temp = np.zeros((num_samples3,Channel_size,timestep))
    for i in range(num_samples3):
        t = supertestdata[i].reshape((timestep,input_size))
        t = np.swapaxes(t,0,1)
        temp[i] = t
    supertestdata = temp
    #pdb.set_trace()

    super_trainData = superdata[:, :, :timestep]
    super_trainLabel = superlabels[:]

    super_testData = supertestdata[:, :, :timestep]
    super_testLabel = supertestlabels[:]

    return (super_trainData, super_trainLabel, super_testData, super_testLabel)

def loadData(dirPath, extractedFlows = 0):
    #If it is not set, use the global value
    if extractedFlows == 0:
        extractedFlows = Num_of_extracted_subflow



    pathDir = os.listdir(dirPath)

    train_datalist = []
    train_labellist = []
    test_datalist = []
    test_labellist = []
    FileCounter = 0
    FlowCounter = 0
    SubflowCounter = 0

    # added by Shahbaz
    custom_features = [
        # 0,    #timestamp
        1,  # RelativeTime
        2,  # length
        # 3    #Direction
    ]

    # added by Shahbaz
    custom_labels = [
       12,    #minLenForward
        13,    #maxLenForward
        14,    #avgLenForward
        15,    #SDLenForward
        16,    #minLenBackward
        17,    #maxLenBackward
        18,    #avgLenBackward
        19,    #SDLenBackward
         20,    #minLenAll
          21,    #maxLenAll
          22,    #avgLenAll
          23,    #SDLenAll
        24,    #minIATForward
        25,    #maxIATForward
        26,    #avgIATForward
        27,    #SDIATForward
        28,    #minIATBackward
        29,    #maxIATBackward
        30,    #avgIATBackward
        31,    #SDIATBackward
         32,    #minIATAll
         33,    #maxIATAll
         34,    #avgIATAll
         35,    #SDIATAll

        #Current version cannot compute these ones:
#         36,    #PPSForward
#         37,    #PPSBackward
#         38,    #PPSAll
#         39,    #BPSForward
#         40,    #BPSBackward
#         41,    #BPSAll
    ]


    for folder, subs, files in os.walk(dirPath):

        #To make sure we get a different random set for each validation cross fold
        files.sort()
        np.random.shuffle(files)

        for file in files:
            filename = folder + "/" + file

            with open(filename) as f:
                
                FileCounter += 1
                statFeatures = np.zeros(42)

                EntireFile = []
                for line in f:
                    data = line.split()
                    try:
                        EntireFile.append(data)
                    except:
                        print(EntireFile)
                        pdb.set_trace()
                try:
                    EntireFile = np.array(EntireFile).astype(np.float)
                except:
                    print(EntireFile)
                    pdb.set_trace()    
                if CompureStatisticsInThisScript and EntireFile.shape[1]==4:
                    EntireFile[-1,0] =-1
                    temparray = []
                    for i in range(0,len(EntireFile)-1):
                        if EntireFile[i,3] == 0:    #Direction
                            temparray.append(EntireFile[i,2])
                    if len(temparray) == 0:
                        statFeatures[12] = 0
                        statFeatures[13] = 0
                        statFeatures[14] = 0
                        statFeatures[15] = 0
                    else:                       
                        statFeatures[12] = np.min(temparray)
                        statFeatures[13] = np.max(temparray)
                        statFeatures[14] = np.mean(temparray)
                        statFeatures[15] = np.std(temparray)
                    
                    temparray = []
                    for i in range(0,len(EntireFile)-1):
                        if EntireFile[i,3] == 1:    #Direction
                            temparray.append(EntireFile[i,2])
                    if len(temparray) == 0:
                        statFeatures[16] = 0
                        statFeatures[17] = 0
                        statFeatures[18] = 0
                        statFeatures[19] = 0
                    else:
                        statFeatures[16] = np.min(temparray)
                        statFeatures[17] = np.max(temparray)
                        statFeatures[18] = np.mean(temparray)
                        statFeatures[19] = np.std(temparray)

                    temparray = []
                    for i in range(0,len(EntireFile)-1):
                        temparray.append(EntireFile[i,2])
                    if len(temparray) == 0:
                        statFeatures[20] = 0
                        statFeatures[21] = 0
                        statFeatures[22] = 0
                        statFeatures[23] = 0
                    else:
                        statFeatures[20] = np.min(temparray)
                        statFeatures[21] = np.max(temparray)
                        statFeatures[22] = np.mean(temparray)
                        statFeatures[23] = np.std(temparray)                   

                    temparray = []
                    old_time = 0
                    for i in range(0,len(EntireFile)-1):
                        if EntireFile[i,3] == 0:    #Direction
                            temparray.append(EntireFile[i,1] - old_time)
                            old_time = EntireFile[i,1]
                    if len(temparray) == 0:
                        statFeatures[24] = 0
                        statFeatures[25] = 0
                        statFeatures[26] = 0
                        statFeatures[27] = 0
                    else:
                        statFeatures[24] = np.min(temparray)
                        statFeatures[25] = np.max(temparray)
                        statFeatures[26] = np.mean(temparray)
                        statFeatures[27] = np.std(temparray)
                    
                    temparray = []
                    old_time = 0
                    for i in range(0,len(EntireFile)-1):
                        if EntireFile[i,3] ==1:    #Direction
                            temparray.append(EntireFile[i,1] - old_time)
                            old_time = EntireFile[i,1]
                    if len(temparray) == 0:
                        statFeatures[28] = 0
                        statFeatures[29] = 0
                        statFeatures[30] = 0
                        statFeatures[31] = 0
                    else:
                        statFeatures[28] = np.min(temparray)
                        statFeatures[29] = np.max(temparray)
                        statFeatures[30] = np.mean(temparray)
                        statFeatures[31] = np.std(temparray)

                    temparray = []
                    old_time = 0
                    for i in range(0,len(EntireFile)-1):
                        temparray.append(EntireFile[i,1] - old_time)
                        old_time = EntireFile[i,1]
                    if len(temparray) == 0:
                        statFeatures[32] = 0
                        statFeatures[33] = 0
                        statFeatures[34] = 0
                        statFeatures[35] = 0
                    else:
                        statFeatures[32] = np.min(temparray)
                        statFeatures[33] = np.max(temparray)
                        statFeatures[34] = np.mean(temparray)
                        statFeatures[35] = np.std(temparray)  
                    
                    
                FileLenght = len(EntireFile)
                
#                pdb.set_trace()
                
                temp_label = []
                SubflowFromAFile = 0
                #To detect TCP traffic and skip first 3 packets
                # isFirstTCP=1;
    
                #Skip the fist few packets in the file
                if(Starting_point!=0):
                    for jjj in range(Starting_point):
                        line = f.readline()
                        if not line:
                            break
    
                # 33 features for one line
                # for i in range(10):
                # Changed by Shahbaz
                for subflow in range(extractedFlows):
    
                    startingPoint = Starting_point + subflow*StartingPointMultiply
                    
                    linedata = []
                    Prev_time = 0;  #Time of the first packet in the subflow
    
                    indexes = custom_features[:]
                    numOfSamples = 0
                    i = startingPoint
                    SkipSamples = SkipPacketsForSampling
                    while(numOfSamples < timestep):    
    #                for i in range(startingPoint, startingPoint+timestep*SkipPacketsForSampling,SkipPacketsForSampling):
    
                        if i>=FileLenght:
                            break
                        
                        
                        if RandomSampling and i!=0:
                            if random.uniform(0,1) > SamplingProbability:
                                i += 1
                                continue

                        data = list(EntireFile[i])  #To clone the list, not refering to the same list
    
#                        if data[0]=="-1" :   #it is the last line containing statistical data
#    #                        if float(data[22]) < 200:
#    #                            print(filename)
#                            try:
#                                temp_label = [float(data[j]) for j in custom_labels]
#                            except (IndexError, ValueError) as e:
#                                pass
#                            print("label failed: " + filename)
#                            subflow = extractedFlows
#                            break
#                        
#                        if(len(data)!=4):
##                            print(filename)
#                            break
    
                        
                        #shahbaz: To descretesize the the length
                        if DescretesizeLength:
                            data[2] = str(int(int(data[2])/100))

                        if DirectionLengthCombined:
                            if data[3]=="0":
                                if float(data[11])>0:
                                    data[2] = str(-1 * float(data[2]))
                                    
                        if NormalizeLength:
                            data[2] = str(float(data[2])/MaxLength)
                            
                        if ComputeInterArrival:
                            if i==startingPoint:
                                Prev_time = float(data[1])
                                data[1] = str(0)
                            else:
                                temporary = str(float(data[1]) - Prev_time)
                                Prev_time = float(data[1])                                
                                data[1] = temporary
                        if NormalizeInterArrival:
                            ttt = float(data[1]) / MaxInterArrival
                            if ttt > 1:
                                ttt=1
                            data[1]=(ttt-0.5)*2
                            
    
                        try:
                            data2 = [float(data[j]) for j in indexes]
                        except (IndexError, ValueError) as e:
                            pass
#                            print("Couldn't retrieve all data",filename)
                        else:
                            linedata += data2
                
                        numOfSamples += 1
                        i += SkipSamples
                        if IncrementalSampling:
                            if numOfSamples % NumberOfSamplesUntiIncrement == 0:
                                SkipSamples = int(SkipSamples*IncrementalStepMultiplier)
    #                            print(SkipSamples)
    
                    if (len(linedata) < len(indexes) * timestep):
                        if (PaddingThreshold > len(linedata)/len(indexes) ):
                            continue
                        #print(linedata)
                        if (PaddingEnable):
                            while(len(linedata) < len(indexes) * timestep):
                                pad = []
                                pad.extend(np.ones(len(indexes)) * 0)
                                if PadAtTheBegining:
                                    pad.extend(linedata)
                                    linedata = pad
                                else:
                                    linedata.extend(pad)
                            #print(linedata)
                        else:
                            continue
                    np.nan_to_num(linedata)
                    #print(linedata)
                    if FileCounter <= TrainingSetSize:
                        train_datalist.append(linedata)
                    else:
                        test_datalist.append(linedata)
    
    #                temp = filename.split("-")
    #                temp = int(temp[8].split(".")[0])
    #                #print(filename)
    #                grouplabel.append(temp)
    #                namelist.append(filename)
                    SubflowCounter+=1
                    SubflowFromAFile+=1
    
                    try:
                        temp_label = [float(statFeatures[j]) for j in custom_labels]
                    except (IndexError, ValueError) as e:
                        pass
                        print("label failed: " + filename)   

                #print(temp_label,SubflowFromAFile)
                total_labels = [temp_label] * SubflowFromAFile
                if FileCounter < TrainingSetSize:
                    train_labellist.extend(total_labels)
                else:
                    test_labellist.extend(total_labels)
                FlowCounter+=1

    ratio = SubflowCounter/FlowCounter
    print(dirPath + ":" + str(FlowCounter) + "/" + str(len(pathDir)) + " - Subflows:" + str(SubflowCounter) + " - Ratio:", str(ratio))
    return (np.array(train_datalist), train_labellist, np.array(test_datalist), test_labellist)


def norm(data):
    data = data - np.amin(data, axis=0, keepdims=True)
    data = data / (np.amax(data, axis=0, keepdims=True) - np.amin(data, axis=0, keepdims=True))
    return data

def norm2(data1, data2):
    temp = np.append(data1, data2, axis=0)
    data1 = data1 - np.amin(temp, axis=0, keepdims=True)
    data1 = data1 / (np.amax(temp, axis=0, keepdims=True) - np.amin(temp, axis=0, keepdims=True))
    data2 = data2 - np.amin(temp, axis=0, keepdims=True)
    data2 = data2 / (np.amax(temp, axis=0, keepdims=True) - np.amin(temp, axis=0, keepdims=True))
    data1 = (data1 - 0.5)*2
    data2 = (data2 - 0.5)*2
    return (data1, data2)
if __name__ == "__main__":


    dataPath = "Data-simple-45-22"
#    dataPath = "Data-random-45-22"
#    dataPath = "Data-incremental-45-22-1.6"
#    dataPath = "Data-simple-75"
#    dataPath = "Data-incremental-45-22-1.6(Human)"

    #### For classification
    BaseDirectory = "Data (unprocessed)/pretraining"
    (data, statlabel, data1, statlabel1) = loadData(BaseDirectory)
    data = np.concatenate((data, data1), axis=0)
    statlabel = np.concatenate((statlabel, statlabel1), axis=0)
    label = np.ones(data.shape[0])
    (data, statlabel) = convertDataUnsupervised(data, label, statlabel)
    np.save(dataPath + "/pretraining_trainData.npy", data)
    np.save(dataPath + "/pretraining_trainLabel.npy", label) #We actually do not need it
    np.save(dataPath + "/StatLabel.npy", statlabel)


    for i in range(NumOfCrossValidationFolds):
        BaseDirectory = "Data (unprocessed)/Retraining(script-triggered)"
#        BaseDirectory = "Data (unprocessed)/Retraining(human-triggered)"
        (superdata1, superstatlabel11, test_data1, test_statlabel1) = loadData(BaseDirectory + "/Google Drive/", extractedFlows=100)
        (superdata2, superstatlabel22, test_data2, test_statlabel2) = loadData(BaseDirectory + "/Youtube/", extractedFlows=100)
        (superdata3, superstatlabel33, test_data3, test_statlabel3) = loadData(BaseDirectory + "/Google Doc/", extractedFlows=100)
        (superdata4, superstatlabel44, test_data4, test_statlabel4) = loadData(BaseDirectory + "/Google Search/", extractedFlows=100)
        (superdata5, superstatlabel55, test_data5, test_statlabel5) = loadData(BaseDirectory + "/Google Music/", extractedFlows=100)
        

        
        superlabel1 = np.ones(superdata1.shape[0])
        superlabel2 = np.ones(superdata2.shape[0]) * 2
        superlabel3 = np.ones(superdata3.shape[0]) * 3
        superlabel4 = np.ones(superdata4.shape[0]) * 4
        superlabel5 = np.ones(superdata5.shape[0]) * 5
        superdata = np.concatenate((superdata1, superdata2, superdata3, superdata4, superdata5), axis=0)
        superlabel = np.concatenate((superlabel1, superlabel2, superlabel3, superlabel4, superlabel5), axis=0)
        superstatlabel = np.concatenate((superstatlabel11, superstatlabel22, superstatlabel33, superstatlabel44, superstatlabel55), axis=0)

        superlabel1 = np.ones(test_data1.shape[0])
        superlabel2 = np.ones(test_data2.shape[0]) * 2
        superlabel3 = np.ones(test_data3.shape[0]) * 3
        superlabel4 = np.ones(test_data4.shape[0]) * 4
        superlabel5 = np.ones(test_data5.shape[0]) * 5
        testdata = np.concatenate((test_data1, test_data2, test_data3, test_data4, test_data5), axis=0)
        testlabel = np.concatenate((superlabel1, superlabel2, superlabel3, superlabel4, superlabel5), axis=0)
        teststatlabel = np.concatenate((test_statlabel1, test_statlabel2, test_statlabel3, test_statlabel4, test_statlabel5), axis=0)

        (super_trainData, super_trainLabel,
         super_testdata, super_testlabels) = convertDataSupervised(superdata, superlabel, testdata, testlabel)
        
        np.save(dataPath + "/re-training_trainData-" + str(i) + ".npy", super_trainData)
        np.save(dataPath + "/re-training_trainLabel-" + str(i) + ".npy", super_trainLabel)
        np.save(dataPath + "/SuperTrainStatLabel-" + str(i) + ".npy", superstatlabel)

        np.save(dataPath + "/re-training_testData-" + str(i) + ".npy", super_testdata)
        np.save(dataPath + "/re-training_testLabel-" + str(i) + ".npy", super_testlabels)
        np.save(dataPath + "/SuperTestStatLabel-" + str(i) + ".npy", teststatlabel)

        print(super_trainData.shape, super_testdata.shape)
