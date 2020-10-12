"""
Copyright (c) 2018 Idiap Research Institute, http://www.idiap.ch/
Written by Remy Siegfried <remy.siegfried@idiap.ch>

This file contains a vfoa estimation method based on a GMM.
Estimate vfoa based on a gaussian model: each target is modelled as a gaussian centered on the target and with
a given variance. The aversion is also modelled as a gaussian centered on the head pose of the subject, with
another given variance. Finally, measure noise is added to each gaussian, and it compute the likelihood of
the observed gaze wrt each target. The final probabilities are normalized and returned, giving the probability
that the subject looks at each target
"""
from copy import deepcopy
import numpy as np

from vfoa.utils.geometry import vectorToYawElevation


class GaussianModel:
    def __init__(self, var_aversion=400., var_target=36., var_noise=0., prob_aversion=0.01):
        self.name = 'gaussianModel'

        self.var_aversion = var_aversion * np.eye(2)  # variance of aversion gaussian (centered on subject's headpose)
        self.var_target = var_target * np.eye(2)      # variance of target gaussian (i.e. radius of target)
        self.var_noise = var_noise * np.eye(2)        # variance of gaze gaussian (i.e. extraction noise)
        self.prob_aversion = prob_aversion            # used if headpose is not given, aversion density is then uniform

        self.vfoa_prob = None  # Unnormalized vfoa
        self.vfoa = None  # Normalized distribution

    def set_parameters(self, parameterList):
        if parameterList is None or len(parameterList) != 4:
            raise Exception('given parameters ({}) do not fit the model. Need 4 parameters'.format(parameterList))
        var_aversion, var_target, var_noise, prob_aversion = parameterList

        if var_aversion is not None:
            self.var_aversion = var_aversion
        if var_target is not None:
            self.var_target = var_target
        if var_noise is not None:
            self.var_noise = var_noise
        if prob_aversion is not None:
            self.prob_aversion = prob_aversion

    def _gaussian(self, x, mu, covar):
        """ Take has input the evaluated variable <x>, the mean <mu> and the covariance matrix <covar>.
            Note that <x> and <mu> must be lists and <covar> a NxN matrix.
            For the 1D case, <covar> must be the variance i.e. sigma**2"""
        if isinstance(mu, np.ndarray) or isinstance(mu, list):
            n = len(mu)
            x = np.reshape(x, (-1))
            mu = np.reshape(mu, (-1))
            covar_inv = np.linalg.inv(covar)

            cst = 1 / ((2*np.pi)**n * np.linalg.norm(covar)) ** .5
            exponential = np.exp(-.5 * (x - mu).dot(covar_inv).dot((x - mu).transpose()))
            return cst * exponential

        else:
            return 1 / (2 * np.pi * covar) ** .5 * np.exp(-.5 * (x - mu)**2 / covar)

    def compute_vfoa(self, person, targetDict, timestamp):
        """ <gaze>: [x, y, z, yaw, pitch, roll]
            <targetDict>: {'targetname': Target}
            <gazeFormat>: should be 'vector' or 'angles' """
        # Check coordinate systems and units
        person.convert_to(positionCS='CCS', poseCS='CCS', poseUnit='deg')
        for key in targetDict.keys():
            targetDict[key].convert_to(positionCS='CCS')

        probLabels = []
        probList = []

        # Compute probability of aversion
        probLabels.append('aversion')
        if person.headpose is not None:
            probList.append(self._gaussian(person.gaze[3:5], person.headpose[3:5], self.var_noise + self.var_aversion))
        else:
            probList.append(self.prob_aversion)  # uniform prob for aversion

        # Compute probability for each target
        for targetName, target in targetDict.items():
            if targetName != person.name:
                if target.position is not None:
                    target = vectorToYawElevation(target.position - person.gaze[0:3])
                    probLabels.append(targetName)
                    probList.append(self._gaussian(person.gaze[3:5], target, self.var_noise + self.var_target))
                else:
                    probLabels.append(targetName)
                    probList.append(0)

        # Normalization and log
        distribution = np.array(probList) / np.sum([probList])

        # Build vfoa
        self.vfoa, self.vfoa_prob = {}, {}
        for label, prob, distr in zip(probLabels, probList, distribution):
            self.vfoa[label] = distr
            self.vfoa_prob[label] = prob

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
        print('[vfoa_module - {}] Training is not yet implemented'.format(self.name))
