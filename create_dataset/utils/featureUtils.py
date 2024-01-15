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

import matplotlib.pyplot as plt
from csv import DictWriter
import numpy as np
import scipy
import os


def bubblesort(sample_array):
    for j in range(len(sample_array)-1,0,-1):
        for i in range(j):
            if sample_array[i][1]< sample_array[i+1][1]:
                temp = sample_array[i]
                sample_array[i] = sample_array[i+1]
                sample_array[i+1] = temp
                bubblesort(sample_array)
    return sample_array

def findIntersectionsForSwDw(pulsewave, sbpLoc, height):
    potentialSwArray = np.argwhere(np.diff(np.sign(height - pulsewave[:sbpLoc]))).flatten()
    if len(potentialSwArray)>0:
        SW = potentialSwArray[-1]
    else:
        SW = 0
    potentialDwArray = np.argwhere(np.diff(np.sign(pulsewave[sbpLoc:] - height))).flatten()
    potentialDwArray+=sbpLoc
    if len(potentialDwArray)>0:
        DW = potentialDwArray[0]
    else:
        DW = len(pulsewave)-2



    return [SW, DW]

def find_intersections_between_1st_and_2nd_deriv(first_deriv, second_deriv):
    intersections = np.argwhere(np.diff(np.sign(np.append(second_deriv,second_deriv[-1])- first_deriv))).flatten()

    return intersections

def findZeroCrossing(pulsewave):
    zeroCrossings = np.argwhere(np.diff(np.sign(pulsewave - 0))).flatten()
    return zeroCrossings
def plotFeaturesForTestingPurposes(iPulse, newPulsewave, samplesPerSecond, importantPoints, sbpLoc, prLoc, dicroticNotchLoc, SBP, DBP, PR, TR, dicroticNotch, h10, h10Indices, h25, h25Indices, h33, h33Indices, h50, h50Indices, h66, h66Indices, h75, h75Indices):
    dirname = 'test' + '/'
    subject = 'test'
    plotdir = 'plotdirfortesting/'
    if not os.path.exists(plotdir):
        os.mkdir(plotdir)
    if not os.path.exists(plotdir + 'ImportantPointsPlots/' + dirname):
        if not os.path.exists(plotdir + 'ImportantPointsPlots/'):
            os.mkdir(plotdir + 'ImportantPointsPlots/')
        os.mkdir(plotdir + 'ImportantPointsPlots/' + dirname)

    if not os.path.exists(plotdir + 'SBP Plot/' + dirname):
        if not os.path.exists(plotdir + 'SBP Plot/'):
            os.mkdir(plotdir + 'SBP Plot/')
        os.mkdir(plotdir + 'SBP Plot/' + dirname)

    if not os.path.exists(plotdir + 'PR Plot/' + dirname):
        if not os.path.exists(plotdir + 'PR Plot/'):
            os.mkdir(plotdir + 'PR Plot/')
        os.mkdir(plotdir + 'PR Plot/' + dirname)

    if not os.path.exists(plotdir + 'Dicrotic Notch Plot/' + dirname):
        if not os.path.exists(plotdir + 'Dicrotic Notch Plot/'):
            os.mkdir(plotdir + 'Dicrotic Notch Plot/')
        os.mkdir(plotdir + 'Dicrotic Notch Plot/' + dirname)

    if not os.path.exists(plotdir + 'Augmented Pressure Plots/' + dirname):
        if not os.path.exists(plotdir + 'Augmented Pressure Plots/'):
            os.mkdir(plotdir + 'Augmented Pressure Plots/')
        os.mkdir(plotdir + 'Augmented Pressure Plots/' + dirname)

    if not os.path.exists(plotdir + 'Pulse Pressure Plots/' + dirname):
        if not os.path.exists(plotdir + 'Pulse Pressure Plots/'):
            os.mkdir(plotdir + 'Pulse Pressure Plots/')
        os.mkdir(plotdir + 'Pulse Pressure Plots/' + dirname)

    if not os.path.exists(plotdir + 'TR Plots/' + dirname):
        if not os.path.exists(plotdir + 'TR Plots/'):
            os.mkdir(plotdir + 'TR Plots/')
        os.mkdir(plotdir + 'TR Plots/' + dirname)

    if not os.path.exists(plotdir + 'Systolic Upstroke Time Plots/' + dirname):
        if not os.path.exists(plotdir + 'Systolic Upstroke Time Plots/'):
            os.mkdir(plotdir + 'Systolic Upstroke Time Plots/')
        os.mkdir(plotdir + 'Systolic Upstroke Time Plots/' + dirname)

    if not os.path.exists(plotdir + 'Diastolic Time Plots/' + dirname):
        if not os.path.exists(plotdir + 'Diastolic Time Plots/'):
            os.mkdir(plotdir + 'Diastolic Time Plots/')
        os.mkdir(plotdir + 'Diastolic Time Plots/' + dirname)

    if not os.path.exists(plotdir + 'SW-DW/' + dirname):
        if not os.path.exists(plotdir + 'SW-DW/'):
            os.mkdir(plotdir + 'SW-DW/')
        os.mkdir(plotdir + 'SW-DW/' + dirname)

    if not os.path.exists(plotdir + 'all/' + dirname):
        if not os.path.exists(plotdir + 'all/'):
            os.mkdir(plotdir + 'all/')
        os.mkdir(plotdir + 'all/' + dirname)

    ###################################################################################################################
    x = np.linspace(0, len(newPulsewave) / samplesPerSecond, len(newPulsewave))


    ###################################################################################################################
    plt.figure()
    plt.plot(x, newPulsewave)
    plt.plot(importantPoints / samplesPerSecond, newPulsewave[importantPoints], 'or')
    plt.ylabel('calibrated radar pulse wave, unitless')
    plt.xlabel('time in s')
    plt.title('Important Points\n' + 'Pulse wave: ' + str(iPulse))
    plt.savefig(plotdir + 'ImportantPointsPlots/' + dirname + str(iPulse) + '.jpg')
    plt.close()

    plt.figure()
    plt.plot(x, newPulsewave)
    plt.plot(sbpLoc / samplesPerSecond, SBP, 'o')
    plt.xlabel('time in s')
    plt.ylabel('calibrated radar pulse wave, unitless')
    plt.title('Systolic BP location\nPulse wave: ' + str(iPulse))
    plt.savefig(plotdir + 'SBP Plot/' + dirname + str(iPulse) + '.jpg')
    plt.close()

    plt.figure()
    plt.plot(x, newPulsewave)
    plt.plot(prLoc / samplesPerSecond, PR, 'o')
    plt.xlabel('time in s')
    plt.ylabel('calibrated radar pulse wave, unitless')
    plt.title('Inflection point P_r induced by the the \nonset of the reflected wave\nSubject: ' + str(
        subject) + ', Pulse wave: ' + str(iPulse))
    plt.savefig(plotdir + 'PR Plot/' + dirname + str(iPulse) + '.jpg')
    plt.close()

    plt.figure()
    plt.plot(x, newPulsewave)
    plt.plot(dicroticNotchLoc / samplesPerSecond, dicroticNotch, 'o')
    plt.xlabel('time in s')
    plt.ylabel('calibrated radar pulse wave, unitless')
    plt.title('Dicrotic Notch location\nSubject: ' + str(subject) + ', Pulse wave: ' + str(iPulse))
    plt.savefig(plotdir + 'Dicrotic Notch Plot/' + dirname + str(iPulse) + '.jpg')
    plt.close()

    plt.figure()
    plt.plot(x, newPulsewave)
    plt.plot(x, np.repeat(PR, len(x)), linestyle='dotted', color='#808080')
    plt.arrow(sbpLoc / samplesPerSecond, PR, 0, SBP - PR, linewidth=3, color='#808080')
    plt.xlabel('time in s')
    plt.ylabel('calibrated radar pulse wave, unitless')
    plt.title(
        'Augmented Pressure: increase in blood flow \ninduced by early arrival of reflected waves\nSubject: ' + str(
            subject) + ', Pulse wave: ' + str(iPulse))
    plt.savefig(plotdir + 'Augmented Pressure Plots/' + dirname + str(iPulse) + '.jpg')
    plt.close()

    plt.figure()
    plt.plot(x, newPulsewave)
    plt.plot(x, np.repeat(DBP, len(x)), linestyle='dotted', color='#808080')
    plt.plot(x, np.repeat(SBP, len(x)), linestyle='dotted', color='#808080')
    plt.arrow(sbpLoc / samplesPerSecond, DBP, 0, SBP - DBP, linewidth=3, color='#808080')
    plt.xlabel('time in s')
    plt.ylabel('calibrated radar pulse wave, unitless')
    plt.title('Pulse Pressure: The maximal peak-to-peak \namplitude of the pressure wave\nSubject: ' + str(
        subject) + ', Pulse wave: ' + str(iPulse))
    plt.savefig(plotdir + 'Pulse Pressure Plots/' + dirname + str(iPulse) + '.jpg')
    plt.close()

    plt.figure()
    plt.plot(x, newPulsewave)
    plt.plot(x, np.repeat(DBP, len(x)), linestyle='dotted', color='#808080')
    plt.plot(np.repeat(prLoc / samplesPerSecond, 15), np.linspace(DBP, PR, 15), linestyle='dotted', color='#808080')
    plt.arrow(0, DBP, TR, 0, linewidth=3, color='#808080')
    plt.xlabel('time in s')
    plt.ylabel('calibrated radar pulse wave, unitless')
    plt.title('T_r: Travel time of reflected wave\nSubject: ' + str(subject) + ', Pulse wave: ' + str(iPulse))
    plt.savefig(plotdir + 'TR Plots/' + dirname + str(iPulse) + '.jpg')
    plt.close()

    plt.figure()
    plt.plot(x, newPulsewave)
    plt.plot(x, np.repeat(DBP, len(x)), linestyle='dotted', color='#808080')
    plt.plot(np.repeat(sbpLoc / samplesPerSecond, 15), np.linspace(DBP, SBP, 15), linestyle='dotted',
             color='#808080')
    plt.arrow(0, DBP, sbpLoc / samplesPerSecond, 0, linewidth=3, color='#808080')
    plt.xlabel('time in s')
    plt.ylabel('calibrated radar pulse wave, unitless')
    plt.title('Systolic Upstroke Time\nSubject: ' + str(subject) + ', Pulse wave: ' + str(iPulse))
    plt.savefig(plotdir + 'Systolic Upstroke Time Plots/' + dirname + str(iPulse) + '.jpg')
    plt.close()

    plt.figure()
    plt.plot(x, newPulsewave)
    plt.plot(x, np.repeat(DBP, len(x)), linestyle='dotted', color='#808080')
    plt.plot(np.repeat(sbpLoc / samplesPerSecond, 15), np.linspace(DBP, SBP, 15), linestyle='dotted',
             color='#808080')
    plt.arrow(sbpLoc / samplesPerSecond, DBP, (len(newPulsewave) - sbpLoc) / samplesPerSecond, 0, linewidth=3,
              color='#808080')
    plt.xlabel('time in s')
    plt.ylabel('calibrated radar pulse wave, unitless')
    plt.title('Diastolic Time\nSubject: ' + str(subject) + ', Pulse wave: ' + str(iPulse))
    plt.savefig(plotdir + 'Diastolic Time Plots/' + dirname + str(iPulse) + '.jpg')
    plt.close()

    plt.figure()
    plt.plot(x, newPulsewave)
    plt.plot(np.repeat(sbpLoc / samplesPerSecond, 15), np.linspace(DBP, SBP, 15), color='#808080')
    plt.plot(x[h10Indices[0]:h10Indices[1]], np.repeat(h10, len(x[h10Indices[0]:h10Indices[1]])),
             linestyle='dotted', color='#808080')
    plt.plot(x[h25Indices[0]:h25Indices[1]], np.repeat(h25, len(x[h25Indices[0]:h25Indices[1]])),
             linestyle='dotted', color='#808080')
    plt.plot(x[h33Indices[0]:h33Indices[1]], np.repeat(h33, len(x[h33Indices[0]:h33Indices[1]])),
             linestyle='dotted', color='#808080')
    plt.plot(x[h50Indices[0]:h50Indices[1]], np.repeat(h50, len(x[h50Indices[0]:h50Indices[1]])),
             linestyle='dotted', color='#808080')
    plt.plot(x[h66Indices[0]:h66Indices[1]], np.repeat(h66, len(x[h66Indices[0]:h66Indices[1]])),
             linestyle='dotted', color='#808080')
    plt.plot(x[h75Indices[0]:h75Indices[1]], np.repeat(h75, len(x[h75Indices[0]:h75Indices[1]])),
             linestyle='dotted', color='#808080')
    plt.xlabel('time in s')
    plt.ylabel('calibrated radar pulse wave, unitless')
    plt.title('SW10, DW10,SW25, DW25, SW33, DW33,\nSW50, DW50, SW66, DW66, SW75, DW75\nSubject: ' + str(
        subject) + ', Pulse wave: ' + str(iPulse))
    plt.savefig(plotdir + 'SW-DW/' + dirname + str(iPulse) + '.jpg')
    plt.close()





    plt.figure()
    plt.plot(x, newPulsewave)

    plt.plot(sbpLoc / samplesPerSecond, SBP, 'o')
    plt.plot(prLoc / samplesPerSecond, PR, 'o')
    plt.plot(dicroticNotchLoc / samplesPerSecond, dicroticNotch, 'o')
    plt.plot(x, np.repeat(PR, len(x)), linestyle='dotted', color='#808080')
    #plt.arrow(SBP_loc / samples_per_second, PR, 0, SBP - PR, linewidth=3, color='#808080')
    #plt.arrow(SBP_loc / samples_per_second, DBP, 0, SBP - DBP, linewidth=3, color='#808080')
    plt.plot(np.repeat(prLoc / samplesPerSecond, 15), np.linspace(DBP, PR, 15), linestyle='dotted', color='#808080')
    #plt.arrow(0, DBP, TR, 0, linewidth=3, color='#808080')
    plt.plot(np.repeat(sbpLoc / samplesPerSecond, 15), np.linspace(DBP, SBP, 15), linestyle='dotted', color='#808080')
    #plt.arrow(0, DBP, SBP_loc / samples_per_second, 0, linewidth=3, color='#808080')

    plt.arrow(sbpLoc / samplesPerSecond, DBP, (len(newPulsewave) - sbpLoc) / samplesPerSecond, 0, head_width=0.5, head_length=0.5, linewidth=3, color='#808080')

    plt.plot(x[h10Indices[0]:h10Indices[1]], np.repeat(h10, len(x[h10Indices[0]:h10Indices[1]])),
             linestyle='dotted', color='#808080')
    plt.plot(x[h25Indices[0]:h25Indices[1]], np.repeat(h25, len(x[h25Indices[0]:h25Indices[1]])),
             linestyle='dotted', color='#808080')
    plt.plot(x[h33Indices[0]:h33Indices[1]], np.repeat(h33, len(x[h33Indices[0]:h33Indices[1]])),
             linestyle='dotted', color='#808080')
    plt.plot(x[h50Indices[0]:h50Indices[1]], np.repeat(h50, len(x[h50Indices[0]:h50Indices[1]])),
             linestyle='dotted', color='#808080')
    plt.plot(x[h66Indices[0]:h66Indices[1]], np.repeat(h66, len(x[h66Indices[0]:h66Indices[1]])),
             linestyle='dotted', color='#808080')
    plt.plot(x[h75Indices[0]:h75Indices[1]], np.repeat(h75, len(x[h75Indices[0]:h75Indices[1]])),
             linestyle='dotted', color='#808080')

    plt.savefig(plotdir + 'all/' + dirname + str(iPulse) + '.jpg')
    plt.close()

def featureDetection(pulsewave, iPulse, samplesPerSecond=125, plotForTest=True):

    pulsewaveBeginning, _ = scipy.signal.find_peaks((-1) * pulsewave[0:int(0.3 * len(pulsewave))])

    indicesMinval = np.where(pulsewave[0:int(len(pulsewave) / 2)] == np.min(pulsewave[0:int(len(pulsewave) / 2)]))

    if len(pulsewaveBeginning) == 0:
        max_loc = np.where(pulsewave == np.max(pulsewave))
        if len(indicesMinval) > 0 and len(max_loc) > 0:
            if indicesMinval[0][0] < max_loc[0][0]:
                pulsewaveBeginning = indicesMinval[0][0]
            else:
                pulsewaveBeginning = 0
        else:
            pulsewaveBeginning = 0
    else:
        pulsewaveBeginning = pulsewaveBeginning[-1]


    pulsewave = np.array(pulsewave)
    newPulsewave = pulsewave[int(pulsewaveBeginning):len(pulsewave)]

    pulsewavePeakLocs, _ = scipy.signal.find_peaks(newPulsewave)
    pulsewavePeakLocs = pulsewavePeakLocs[pulsewavePeakLocs < 0.8*len(newPulsewave)]

    if len(pulsewavePeakLocs) > 0:
        startValleySearch = pulsewavePeakLocs[0] + 2
    else:
        startValleySearch = 10
    endValleySearch = int(0.6 * len(newPulsewave))
    pulsewaveValleyLocs, _ = scipy.signal.find_peaks((-1) * newPulsewave[startValleySearch:endValleySearch])
    pulsewaveValleyLocs += startValleySearch

    SBP = np.max(newPulsewave)
    SBPLoc = np.where(newPulsewave == SBP)
    SBPLoc = SBPLoc[0][0]


    SBPTime = SBPLoc / samplesPerSecond

    AP_x, PRLoc, PR = 0, 0, 0

    diastolicPeakLoc = 0


    sortedPeaks =  sorted(([value, index] for index, value in enumerate(newPulsewave[pulsewavePeakLocs])), reverse=True)
    sortedPeakIndices = [row[1] for row in sortedPeaks]
    sortedPeakLocs = pulsewavePeakLocs[sortedPeakIndices]


    if len(sortedPeakLocs) > 1 and sortedPeakLocs[1] < len(newPulsewave):
        for i in range(1, len(sortedPeakLocs)):
            if sortedPeakLocs[i] > 0.2*len(newPulsewave):
                diastolicPeakLoc = sortedPeakLocs[i]
                break
            if i == len(sortedPeakLocs) - 1:
                diastolicPeakLoc = SBPLoc + int(0.3*len(newPulsewave)) if SBPLoc + int(0.3*len(newPulsewave)) < len(newPulsewave) else len(
                    newPulsewave) - 2
    else:
        diastolicPeakLoc = SBPLoc + int(0.3*len(newPulsewave)) if SBPLoc + int(0.3*len(newPulsewave)) < len(newPulsewave) else len(
            newPulsewave) - 2

    diastolicPeakTime = diastolicPeakLoc / samplesPerSecond

    # print(SBPLoc, diastolicPeakLoc)
    if diastolicPeakLoc <= SBPLoc:
        diastolicPeakLoc = SBPLoc + int(0.3*len(newPulsewave)) if SBPLoc + int(0.3*len(newPulsewave)) < len(newPulsewave) else len(
            newPulsewave) - 2

    if SBPLoc + 1 == diastolicPeakLoc - 1:
        diastolicPeakLoc = SBPLoc + int(0.3*len(newPulsewave)) if SBPLoc + int(0.3*len(newPulsewave)) < len(newPulsewave) else len(
            newPulsewave) - 2

    if diastolicPeakLoc > SBPLoc+2:
        dicroticNotchLoc = np.where(newPulsewave[SBPLoc + 1:diastolicPeakLoc - 1] == np.min(newPulsewave[SBPLoc + 1:diastolicPeakLoc - 1]))
        dicroticNotchLoc = dicroticNotchLoc[0][0] + SBPLoc + 1
    else:
        dicroticNotchLoc = np.where(newPulsewave[SBPLoc:] == np.min(newPulsewave[SBPLoc:]))
        dicroticNotchLoc = dicroticNotchLoc[0][0] + SBPLoc


    dicroticNotchTime = dicroticNotchLoc / samplesPerSecond
    dicroticNotch = newPulsewave[dicroticNotchLoc]


    startDeriv = 0
    endDeriv = dicroticNotchLoc
    firstDeriv = np.diff(newPulsewave[startDeriv:endDeriv])
    secondDeriv = np.diff(firstDeriv)

    zeroCrossingsFirstDeriv = findZeroCrossing(firstDeriv)
    localMaximaFirstDeriv = scipy.signal.argrelextrema(firstDeriv, np.greater)
    localMinimaFirstDeriv = scipy.signal.argrelextrema(firstDeriv, np.less)


    zeroCrossingsSecondDeriv = findZeroCrossing(secondDeriv)

    PRLoc = 0
    for zcSecondDeriv in zeroCrossingsSecondDeriv:

        if ((dicroticNotchLoc-SBPLoc)/2 + SBPLoc) < zcSecondDeriv and PRLoc != 0:
            break
        if zcSecondDeriv < SBPLoc:
            if np.any(localMinimaFirstDeriv == zcSecondDeriv+1):
                PRLoc = (zcSecondDeriv)
        else:
            if np.any(localMaximaFirstDeriv == zcSecondDeriv+1):
                PRLoc=(zcSecondDeriv)



    # young subject
    if PRLoc < SBPLoc:
        AP_x = -1
        PR = newPulsewave[PRLoc]

    # older subject
    else:
        AP_x = 1
        PR = newPulsewave[PRLoc]

    AP = (SBP - PR) * AP_x
    DBP = np.min(newPulsewave)

    PP = SBP - DBP

    TR = (PRLoc) / samplesPerSecond





    ###################################################################################################################

    systolicUpstrokeTime = (SBPLoc) / samplesPerSecond
    diastolicTime = (len(newPulsewave) - SBPLoc) / samplesPerSecond

    # SWx, DWx is the systolic/diastolic value  x% of its width
    # hx is the height at x%
    h10 = (DBP + 0.1 * PP)
    h25 = (DBP + 0.25 * PP)
    h33 = (DBP + 0.33 * PP)
    h50 = (DBP + 0.5 * PP)
    h66 = (DBP + 0.66 * PP)
    h75 = (DBP + 0.75 * PP)

    h10Indices = findIntersectionsForSwDw(newPulsewave, SBPLoc, h10)
    h25Indices = findIntersectionsForSwDw(newPulsewave, SBPLoc, h25)
    h33Indices = findIntersectionsForSwDw(newPulsewave, SBPLoc, h33)
    h50Indices = findIntersectionsForSwDw(newPulsewave, SBPLoc, h50)
    h66Indices = findIntersectionsForSwDw(newPulsewave, SBPLoc, h66)
    h75Indices = findIntersectionsForSwDw(newPulsewave, SBPLoc, h75)




    SW10, DW10 = (SBPLoc - h10Indices[0]) / samplesPerSecond, (h10Indices[1] - SBPLoc) / samplesPerSecond
    SW25, DW25 = (SBPLoc - h25Indices[0]) / samplesPerSecond, (h25Indices[1] - SBPLoc) / samplesPerSecond
    SW33, DW33 = (SBPLoc - h33Indices[0]) / samplesPerSecond, (h33Indices[1] - SBPLoc) / samplesPerSecond
    SW50, DW50 = (SBPLoc - h50Indices[0]) / samplesPerSecond, (h50Indices[1] - SBPLoc) / samplesPerSecond
    SW66, DW66 = (SBPLoc - h66Indices[0]) / samplesPerSecond, (h66Indices[1] - SBPLoc) / samplesPerSecond
    SW75, DW75 = (SBPLoc - h75Indices[0]) / samplesPerSecond, (h75Indices[1] - SBPLoc) / samplesPerSecond


    if plotForTest:
        importantPointsForPlotting = np.hstack(([SBPLoc], [PRLoc], [dicroticNotchLoc], [diastolicPeakLoc]))
        plotFeaturesForTestingPurposes(iPulse, newPulsewave, samplesPerSecond, importantPointsForPlotting, SBPLoc, PRLoc, dicroticNotchLoc, SBP, DBP, PR, TR, dicroticNotch, h10, h10Indices, h25, h25Indices, h33, h33Indices, h50, h50Indices, h66, h66Indices, h75, h75Indices)

    return pulsewaveBeginning, SBP, DBP, PR, dicroticNotch, AP, PP, TR, systolicUpstrokeTime, diastolicTime, SW10, DW10, SW25, DW25, SW33, DW33, SW50, DW50, SW66, DW66, SW75, DW75

def scaleDataTo01Range(arrayToScale):
    return (arrayToScale - np.min(arrayToScale)) / (np.max(arrayToScale) - np.min(arrayToScale))

def scaleCurveUsingGroundTruthBpAndRatioToFirstCurve(arrayToScale, referencePpForScaling, ppOfFirstCurveForRatioComparison):
    
    ppOfGivenCurve = np.max(arrayToScale) - np.min(arrayToScale)
    scaled01Array = scaleDataTo01Range(arrayToScale)
    
    #will be smaller than 1 if the pp is smaller than for the first curve and bigger otherwise
    #helps to track pp variations
    ratioBetweenCurves = ppOfGivenCurve / ppOfFirstCurveForRatioComparison
    
    scaledArrayForFeatureExtraction = scaled01Array * referencePpForScaling * ratioBetweenCurves
    
    return scaledArrayForFeatureExtraction

def findValleyIndices(data):
    peakIndices, _ = scipy.signal.find_peaks(np.array(data))
    valleyIndices, _ = scipy.signal.find_peaks((-1) * np.array(data))
    return peakIndices, valleyIndices

def findPeaksAndValleysWithMinPeakProminenceFilter(data, samplesPerSecond = 125, maxHeartRate = 180):
    peakIndicesUnoptimized, valleyIndicesUnoptimized = findValleyIndices(data)
    minPeakProm = 0.5 * (np.max(data) - np.min(data))
    minHeartbeatDurationInSeconds = 60 / maxHeartRate
    samplesForMinHeartrate = minHeartbeatDurationInSeconds * samplesPerSecond

    peakIndices, _ = scipy.signal.find_peaks(data, prominence=minPeakProm, distance=samplesForMinHeartrate) #prominence=minPeakProm,
    if len(peakIndices) < 2:
        peakIndices = peakIndicesUnoptimized
    valleyIndices = []

    for i in range(0, len(peakIndices)):

        if i == 0:
            minBeforePeak = np.argwhere(data[0:peakIndices[i]] == np.min(data[0:peakIndices[i]]))
            minBeforePeak = minBeforePeak[0][0]
        else:
            if peakIndices[i] - 20 >0:
                dataSegmentToSearchForMin = data[peakIndices[i] - 20:peakIndices[i]]
            else:
                dataSegmentToSearchForMin = data[0:peakIndices[i]]
            minBeforePeak = np.argwhere( dataSegmentToSearchForMin == np.min(dataSegmentToSearchForMin))
            minBeforePeak = minBeforePeak[0][0]
            minBeforePeak += peakIndices[i] - 20

        valleyIndices.append(minBeforePeak)



    lastSecond = data[len(data)-samplesPerSecond:len(data)]
    lastValley = np.argmax(-1*np.array(lastSecond))
    if lastValley not in valleyIndices:
        valleyIndices.append(lastValley)


    valleys = data[valleyIndices]
    peaks = data[peakIndices]
    return peaks, peakIndices, valleys, valleyIndices

def appendDictAsRow(fileName, elementDict, fieldNames):
    # Open file in append mode
    with open(fileName, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        dictWriter = DictWriter(write_obj, fieldnames=fieldNames)
        # Add dictionary as wor in the csv
        dictWriter.writerow(elementDict)
