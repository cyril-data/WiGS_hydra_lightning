import pandas as pd

def FindMissingSimulations(Abbrv, Threshold, text):
    ### Combined Analysis for Unique and Duplicate ###
    words = text.split()
    TotalCountUnique = []
    TotalCountDuplicate = []
    TotalCountDuplicateUNQ = []
    TotalCountDuplicateDPL = []

    for i in range(0, 50):
        substringUNQ = str(i) + Abbrv + "_UA" +str(Threshold)
        substringDPL = str(i) + Abbrv + "_DA" +str(Threshold)

        # Unique occurrences
        unique_occurrencesUNQ = {word for word in words if word.startswith(substringUNQ)}
        unique_occurrencesDPL = {word for word in words if word.startswith(substringDPL)}

        countUNQ = len(unique_occurrencesUNQ)
        countDPL = len(unique_occurrencesDPL)

        TotalCountUnique.append(countUNQ)
        TotalCountDuplicate.append(countDPL)

        # Duplicate occurrences
        occurrencesUNQ = [word for word in words if word.startswith(substringUNQ)]
        occurrencesDPL = [word for word in words if word.startswith(substringDPL)]

        countDuplicatesUNQ = len(occurrencesUNQ)
        countDuplicatesDPL = len(occurrencesDPL)

        TotalCountDuplicateUNQ.append(countDuplicatesUNQ)
        TotalCountDuplicateDPL.append(countDuplicatesDPL)

    ### Convert to DataFrames ###
    TotalCountUnique = pd.DataFrame(TotalCountUnique, columns=["Unique"])
    TotalCountDuplicate = pd.DataFrame(TotalCountDuplicate, columns=["Duplicate"])
    TotalCountDuplicateUNQ = pd.DataFrame(TotalCountDuplicateUNQ, columns=["Duplicate_UNQ"])
    TotalCountDuplicateDPL = pd.DataFrame(TotalCountDuplicateDPL, columns=["Duplicate_DPL"])

    ### Identify Missing ###
    MissingUNQ = list(TotalCountUnique[TotalCountUnique["Unique"] < 1].index)
    MissingDPL = list(TotalCountDuplicate[TotalCountDuplicate["Duplicate"] < 1].index)

    ### Identify Duplicates ###
    DuplicateIndicesUNQ = list(TotalCountDuplicateUNQ[TotalCountDuplicateUNQ["Duplicate_UNQ"] > 1].index)
    DuplicateIndicesDPL = list(TotalCountDuplicateDPL[TotalCountDuplicateDPL["Duplicate_DPL"] > 1].index)

    ### Have Both ###
    InBothBoolean = (TotalCountUnique["Unique"] == 1) & (TotalCountDuplicate["Duplicate"] == 1)
    HaveBoth = list(InBothBoolean[InBothBoolean == True].index)

    ### Missing List ###
    MissingDPL_List = [str(item) + Abbrv + "_DA" + str(str(Threshold)) for item in MissingDPL]
    MissingUNQ_List = [str(item) + Abbrv + "_UA" + str(str(Threshold)) for item in MissingUNQ]

    HaveBothDPL_List = [str(item) + Abbrv + "_DA" + str(str(Threshold)) for item in HaveBoth]
    HaveBothUNQ_List = [str(item) + Abbrv + "_UA" + str(str(Threshold)) for item in HaveBoth]

    ### Outputs ###
    print("Duplicate Indices for UNQ:", DuplicateIndicesUNQ)
    print("Duplicate Indices for DPL:", DuplicateIndicesDPL)
    print("Missing Indices for UNQ:", MissingUNQ)
    print("Missing Indices for DPL:", MissingDPL)
    print("Missing (combined) JobNames:")
    print(MissingDPL_List + MissingUNQ_List)

    print("Have both JobNames:")
    print(HaveBothDPL_List + HaveBothUNQ_List)
