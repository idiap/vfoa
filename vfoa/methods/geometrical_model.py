"""
Copyright (c) 2018 Idiap Research Institute, http://www.idiap.ch/
Written by Remy Siegfried <remy.siegfried@idiap.ch>

This file contains a vfoa estimation method based on geometrical features.
Estimate vfoa based on a geometric model: if the angular distance between the gaze vector and the target
direction (i.e. line frome eye to target) is below a given threshold, the gaze is allocated to this target.
If several targets fill this condition, the nearest to the gaze vector wins.
"""
from copy import deepcopy
import numpy as np

from vfoa.utils.geometry import angleBetweenVectors_deg, yawElevationToVector


class GeometricModel:
    def __init__(self, thresh=10):
        self.name = 'geometricalModel'
        self.thresh = thresh  # Threshold applied on the angular distance between gaze and target direction
        self.vfoa_prob = None  # Unnormalized vfoa
        self.vfoa = None  # Normalized distribution

    def set_parameters(self, parameterList):
        if parameterList is None or len(parameterList) != 1:
            raise Exception('given parameters ({}) do not fit the model. Need 1 parameter'.format(parameterList))
        threshold = parameterList[0]

        if threshold is not None:
            self.thresh = threshold

    def compute_vfoa(self, person, targetDict, timestamp):
        """ <gaze>: [x, y, z, yaw, pitch, roll]
            <targetDict>: {'targetname': Target}
            <gazeFormat>: should be 'vector' or 'angles' """
        # Check coordinate systems and units
        person.convert_to(positionCS='CCS', poseCS='CCS', poseUnit='deg')
        for key in targetDict.keys():
            targetDict[key].convert_to(positionCS='CCS')

        # Get nearest Target
        nearestTarget, minDist = 'aversion', np.inf
        for targetName, target in targetDict.items():
            if targetName != person.name:
                if target.position is not None:
                    # Compute 3D anglular error
                    gazeVec = yawElevationToVector(person.gaze[3:5])
                    angDist = angleBetweenVectors_deg(gazeVec, target.position - person.gaze[0:3])  # Error in 3D

                    # Compute error in 2D angles (yaw, elevation)
                    # targetAngles = np.array(vectorToYawElevation(target.position - person.gaze[0:3]))
                    # yawEleDist = person.gaze[3:5] - targetAngles

                    if angDist < self.thresh and angDist < minDist:
                        minDist = angDist
                        nearestTarget = targetName

        # Build vfoa
        self.vfoa = {'aversion': int(nearestTarget is 'aversion')}
        for targetName, target in targetDict.items():
            if targetName != person.name:
                self.vfoa[targetName] = int(nearestTarget is targetName)
        self.vfoa_prob = self.vfoa

    def get_vfoa_prob(self, targetName=None):
        """ Return probability that the person looks to the target (dict of {'targetName': probability} or this
            probability for all targets if <targetName> is None.)
            Note that those probabilites does not sum to 1 (not like <self.vfoa>, which is a distribution)."""
        if targetName:
            return deepcopy(self.vfoa_prob[targetName])
        else:
            return deepcopy(self.vfoa_prob)

    def get_vfoa_distribution(self, targetName=None):
        """ Return distribution that the person looks to the target (dict of {'targetName': probability} or this
            probability for all targets if <targetName> is None.) """
        if targetName:
            return deepcopy(self.vfoa[targetName])
        else:
            return deepcopy(self.vfoa)

    def train(self, personList, targetDictList, groundTruth):
        distCollection_targets = []
        distCollection_aversion = []

        # Gather decision value (i.e. angular distance between target direction and gaze)
        nbSamples = 0
        for person, targetDict, gt in zip(personList, targetDictList, groundTruth):
            nearestTarget, minDist = 'aversion', np.inf
            for targetName, target in targetDict.items():
                if targetName != person.name:
                    if target.position is not None:
                        # Compute 3D angular error
                        gazeVec = yawElevationToVector(person.gaze[3:5])
                        angDist = angleBetweenVectors_deg(gazeVec, target.position - person.gaze[0:3])
                        if angDist < minDist:
                            minDist = angDist
                            nearestTarget = targetName

            if gt == nearestTarget:
                distCollection_targets.append(minDist)
                nbSamples += 1
            elif gt == 'aversion':
                distCollection_aversion.append(minDist)
                nbSamples += 1

        # Build GMM model of classes
        mean1 = np.mean(np.array(distCollection_targets))
        var1 = np.var(np.array(distCollection_targets))

        mean2 = np.mean(np.array(distCollection_aversion))
        var2 = np.var(np.array(distCollection_aversion))

        # Compute gaussian intersections
        a = 1/var1 - 1/var2
        b = 2*(mean2/var2 - mean1/var1)
        c = mean1**2/var1 - mean2**2/var2 - np.log(var2/var1)
        intersections = np.roots([a, b, c])

        # Select the intersection representing a highest probability (intuition)
        def _gaussian(x, mu, var):
            return 1 / (2 * np.pi * var) ** .5 * np.exp(-.5 * (x - mu) ** 2 / var)
        if _gaussian(intersections[0], mean1, var1) > _gaussian(intersections[1], mean1, var1):
            self.thresh = intersections[0]
        else:
            self.thresh = intersections[1]
        print('Training {} ({} samples), threshold was updated to {}'.format(self.name, nbSamples, self.thresh))
