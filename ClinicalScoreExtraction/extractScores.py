import os
import pandas as pd
import numpy as np

def getClinicalScores(rootdir):

    scoreDataList = []

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            # TODO: CHANGE THE NAME BELOW  TO ClinicalAssessment
            if file.find("ClinicalAssessment") != -1 and file.endswith(".xlsx"):
                filename = os.path.join(subdir, file)
                data = pd.read_excel(filename)
                df = pd.DataFrame(data)
                scoreDataList.append(df)
                # print(filename)


    result = pd.concat(scoreDataList)
    # remove duplicated rows
    result = result.drop_duplicates()
    # set Subject ID as index
    result = result.set_index('Subject ID')

    scoreSum = np.cumsum(result, axis=0)[1:]
    # print(np.cumsum(result,axis=0)[-1:])
    scoreMean = result.mean()
    print(scoreMean)

    # print(result)


if __name__ == '__main__':
    directoryPath = '/Users/Clara/Desktop/KiMoRe/Full'
    getClinicalScores(directoryPath)