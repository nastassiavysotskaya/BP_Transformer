## ===========================================================================
## Copyright (C) 2024 Infineon Technologies AG
##
## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are met:
##
## 1. Redistributions of source code must retain the above copyright notice,
##    this list of conditions and the following disclaimer.
## 2. Redistributions in binary form must reproduce the above copyright
##    notice, this list of conditions and the following disclaimer in the
##    documentation and/or other materials provided with the distribution.
## 3. Neither the name of the copyright holder nor the names of its
##    contributors may be used to endorse or promote products derived from
##    this software without specific prior written permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
## AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
## IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
## ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
## LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
## CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
## SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
## INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
## CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
## ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
## POSSIBILITY OF SUCH DAMAGE.
## ===========================================================================

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os

SMALL_SIZE = 6

plt.rc('axes', labelsize=SMALL_SIZE+4)     # fontsize of the axes title
#.rc('title', labelsize=8)     # fontsize of the axes title

plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.tight_layout()


def blandAltmanPlot(groundTruth, predicted, filename, title='Bland-Altman Plot'):
    # Calculate the mean of the ground truth and predicted values
    mean = np.mean([groundTruth, predicted], axis=0)

    # Calculate the difference between the ground truth and predicted values
    diff = groundTruth - predicted
    md = np.mean(diff)

    # Calculate the limits of agreement (1.96 times the standard deviation of the differences)
    limitsOfAgreement = 1.96 * np.std(diff)

    # Create the Bland-Altman plot
    plt.figure(figsize=(6,4), dpi=300)
    plt.scatter(mean, diff, c="#a7bbd4", s=5, alpha=0.3)
    plt.axhline(md, color='#273b55', linestyle='-', linewidth=2)
    plt.axhline(md + limitsOfAgreement, color="#4f77aa", linestyle='--', linewidth=2)
    plt.axhline(md - limitsOfAgreement, color="#4f77aa", linestyle='--', linewidth=2)

    # Add labels and title
    plt.xlabel('Mean of Ground Truth and Predicted Values')
    plt.ylabel('Difference (Ground Truth - Predicted)')
    plt.title(title, fontsize=SMALL_SIZE+4)

    xOutPlot = np.min(mean) + (np.max(mean) - np.min(mean)) * 1.14

    plt.text(xOutPlot, md - limitsOfAgreement,
             r'-1.96 std:'  + "  %.2f" % (md - limitsOfAgreement),
             ha="center",
             va="center",
             fontsize=6
             )
    plt.text(xOutPlot, md + limitsOfAgreement,
             r'+1.96 std:' + "  %.2f" % (md + limitsOfAgreement),
             ha="center",
             va="center",
             fontsize=6
             )
    plt.text(xOutPlot, md,
             'Mean:' + "  %.2f" % md,
             ha="center",
             va="center",
             fontsize=6
             )
    plt.subplots_adjust(right=0.85)

    plt.savefig(filename, transparent=True)
    plt.close()

def scatterPlot(groundTruth, predicted, filename, title='Scatter Plot'):
    # Fit a linear regression line through the predicted points
    regression = LinearRegression()
    regression.fit(np.array(groundTruth).reshape(-1, 1), predicted)
    predictionLine = regression.predict(np.array(groundTruth).reshape(-1, 1))

    r2 = r2_score(groundTruth, predicted)

    # Create the scatter plot
    fig= plt.figure(figsize=(4,3), dpi=300)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.16)
    plt.scatter(groundTruth, predicted, c="#a7bbd4", s=5, alpha=0.2)
    plt.plot(groundTruth, groundTruth, c="#273b55", label="optimal predictions")
    plt.plot(groundTruth, predictionLine, color='red', label='regression line: [%.2f]' % (r2))

    # Add labels and title
    plt.xlabel('Ground Truth')
    plt.ylabel('Predicted Values')
    plt.legend(fontsize=SMALL_SIZE)
    plt.title(title, fontsize=SMALL_SIZE+6)


    plt.savefig(filename, transparent=True)
    plt.close()


def errorHistPlot(errors, mae, std, filename, title='Error Histogram Plot'):
    absErrors = np.abs(errors)
    absBins = int(np.round(np.max(absErrors))) + int(np.round(np.min(absErrors)))
    bins = int(np.round(np.max(errors))) + int(np.round(np.abs(np.min(errors))))

    plt.figure(dpi=300, figsize=(4,3))
    #plt.hist(absErrors, bins=absBins, color='#4f77aa', edgecolor='white')
    plt.hist(errors, bins=bins, color='#a7bbd4', edgecolor='white')
    plt.axvline(mae, color='#273b55', linestyle='-', linewidth=1, label="MAE \u00B1 std: \n[%.2f \u00B1 %.2f]" % (mae, std))
    plt.axvline(mae + std, color="#4f77aa", linestyle='--', linewidth=1)
    plt.axvline(mae - std, color="#4f77aa", linestyle='--', linewidth=1)

    # Add labels and title
    plt.xlabel('Absolute Error Bins')
    plt.ylabel('Number of occurences')
    plt.xlim(int(np.round((np.min(errors)))), int(np.round(np.max(errors))))
    plt.title(title, fontsize=SMALL_SIZE+6)
    plt.legend(fontsize=SMALL_SIZE+2, handlelength=0 )

    # Show the plot
    #plt.grid()
    plt.savefig(filename, transparent=True)
    plt.close()






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_model_path", default='todo/', type=str, required=True)

    args = parser.parse_args()
    trainedModelPath = args.trained_model_path

    resultspath = trainedModelPath + "testing_results.csv"
    results = pd.read_csv(resultspath)

    sbpGroundTruth = results['SBP GT']
    dbpGroundTruth = results['DBP GT']

    sbpPredicted = results['SBP Pred']
    dbpPredicted = results['DBP Pred']

    sbpDifference = results['SBP Diff']
    dbpDifference = results['DBP Diff']

    sbpMAE = np.mean(np.abs(sbpDifference))
    dbpMAE = np.mean(np.abs(dbpDifference))

    sbpME = np.mean(sbpDifference)
    dbpME = np.mean(dbpDifference)

    sbpSTD = np.std(np.abs(sbpDifference))
    dbpSTD = np.std(np.abs(dbpDifference))

    print("\n\n")
    print("_____________________________________________________________")
    print("Let's compare to the AAMI standard (MAE+-STD):")
    print("SBP: %.2f +- %.2f" % (sbpMAE, sbpSTD))
    print("DBP: %.2f +- %.2f" % (dbpMAE, dbpSTD))

    sbpErrorSmallerThan5  = np.sum(np.abs(sbpDifference) < 5) / len(results)
    sbpErrorSmallerThan10 = np.sum(np.abs(sbpDifference) < 10) / len(results)
    sbpErrorSmallerThan15 = np.sum(np.abs(sbpDifference) < 15) / len(results)

    dbpErrorSmallerThan5  = np.sum(np.abs(dbpDifference) < 5) / len(results)
    dbpErrorSmallerThan10 = np.sum(np.abs(dbpDifference) < 10) / len(results)
    dbpErrorSmallerThan15 = np.sum(np.abs(dbpDifference) < 15) / len(results)

    print("\n\n")
    print("_____________________________________________________________")
    print("Let's compare to the BHS standard (Error <5 | <10 | <15):")
    print("SBP: [%.2f, %.2f, %.2f]" % (sbpErrorSmallerThan5, sbpErrorSmallerThan10, sbpErrorSmallerThan15))
    print("DBP: [%.2f, %.2f, %.2f]" % (dbpErrorSmallerThan5, dbpErrorSmallerThan10, dbpErrorSmallerThan15))



    resultPicturePath = trainedModelPath + "result_plots/"
    if not os.path.exists(resultPicturePath):
        os.mkdir(resultPicturePath)

    blandAltmanPlot(groundTruth=sbpGroundTruth, predicted=sbpPredicted, filename=resultPicturePath + "blandaltman_SBP.pdf", title="Bland-Altman plot for SBP")
    blandAltmanPlot(groundTruth=dbpGroundTruth, predicted=dbpPredicted, filename=resultPicturePath + "blandaltman_DBP.pdf", title="Bland-Altman plot for DBP")

    scatterPlot(groundTruth=sbpGroundTruth, predicted=sbpPredicted, filename=resultPicturePath + "scatter_SBP.pdf", title="Scatter plot for SBP")
    scatterPlot(groundTruth=dbpGroundTruth, predicted=dbpPredicted, filename=resultPicturePath + "scatter_DBP.pdf", title="Scatter plot for DBP")

    errorHistPlot(errors=(sbpDifference), mae= sbpMAE, std=sbpSTD, filename=resultPicturePath + "error_histogram_SBP.pdf", title="Histogram plot for SBP")
    errorHistPlot(errors=(dbpDifference), mae= dbpMAE, std=dbpSTD, filename=resultPicturePath + "error_histogram_DBP.pdf", title="Histogram plot for DBP")