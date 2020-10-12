"""
Copyright (c) 2018 Idiap Research Institute, http://www.idiap.ch/
Written by Remy Siegfried <remy.siegfried@idiap.ch>

This file contains a vfoa estimation method based on gaze posterior.
Estimate vfoa based on the probability that the subject is looking in the target direction. It means that
we want to compute the posterior probability of the gaze and evaluate it at each target position.
Thus, output probabilities does not sum to 1, as they are only point-wise evaluation of the posterior.
"""

from copy import deepcopy
import numpy as np

from vfoa.utils.geometry import vectorToYawElevation


class GazeProbability:
    def __init__(self, var_noise=36., var_obsPrior=None, var_estPrior=None, prob_aversion=0.0001):
        # By default, var_obsPrior and var_estPrior are not used as they seem to decrease the results
        self.name = 'gazeProbability'
        self.var_noise = var_noise * np.eye(2)  # variance of gaze gaussian (i.e. extraction noise)
        # variance of gaze priors (centered on subject's headpose)
        self.var_obsPrior = var_obsPrior * np.eye(2) if var_obsPrior is not None else None
        self.var_estPrior = var_estPrior * np.eye(2) if var_estPrior is not None else None
        self.prob_aversion = prob_aversion      # Aversion fixed probability (i.e. a threshold)

        self.vfoa_prob = None  # Unnormalized vfoa
        self.vfoa = None  # Normalized distribution

    def set_parameters(self, parameterList):
        if parameterList is None or len(parameterList) != 4:
            raise Exception('given parameters ({}) do not fit the model. Need 4 parameters'.format(parameterList))
        var_noise, var_obsPrior, var_estPrior, prob_aversion = parameterList

        if var_noise is not None:
            self.var_noise = var_noise
        if var_obsPrior is not None:
            self.var_obsPrior = var_obsPrior
        if var_estPrior is not None:
            self.var_estPrior = var_estPrior
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

        # Compute probability for each target
        for targetName, target in targetDict.items():
            if targetName != person.name:
                if target.position is not None:
                    targetDirection = vectorToYawElevation(target.position - person.gaze[0:3])
                    probLabels.append(targetName)
                    if self.var_estPrior is not None and self.var_obsPrior is not None:
                        likelihood = self._gaussian(person.gaze[3:5], targetDirection, self.var_noise)       # p(g*|g=t)
                        prior = self._gaussian(targetDirection, person.headpose[3:5], self.var_estPrior)      # p(g=t)
                        norm_cst = self._gaussian(person.gaze[3:5], person.headpose[3:5], self.var_obsPrior)  # p(g*)
                        probList.append(likelihood * prior / norm_cst)
                    elif self.var_estPrior is not None:
                        # WARNING: This option return a probability that is only proportional to the real probability
                        likelihood = self._gaussian(person.gaze[3:5], targetDirection, self.var_noise)       # p(g*|g=t)
                        prior = self._gaussian(targetDirection, person.headpose[3:5], self.var_estPrior)      # p(g=t)
                        probList.append(likelihood * prior)
                    else:
                        # Truncated probability
                        probList.append(self._gaussian(person.gaze[3:5], targetDirection, self.var_noise))
                else:
                    probLabels.append(targetName)
                    probList.append(0)

        # Compute aversion given the probability to look at any target and threshold
        if self.prob_aversion is not None:
            probLabels.append('aversion')
            probList.append(self.prob_aversion)

        # Build vfoa
        self.vfoa_prob = {}
        for label, prob in zip(probLabels, probList):
            self.vfoa_prob[label] = prob

        sum_prob = np.sum(self.vfoa_prob.values())
        self.vfoa = deepcopy(self.vfoa_prob)
        for key, val in self.vfoa.items():
            self.vfoa[key] = val / sum_prob

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
        # Gather data
        nbSamples_obs, nbSamples_gt, nbSamples_aversion = 0, 0, 0
        observationAroundGaze, observationAroundHead, gazeAroundHead = [], [], []
        for person, targetDict, gt in zip(personList, targetDictList, groundTruth):
            gaze_obs = person.gaze[3:5]
            head_obs = person.headpose[3:5]
            observationAroundHead.append(gaze_obs - head_obs)     # gaze_obs prior is gaussian around head
            nbSamples_obs += 1
            if gt in targetDict.keys():
                gaze_gt = vectorToYawElevation(targetDict[gt].position - person.gaze[0:3])
                observationAroundGaze.append(gaze_obs - gaze_gt)  # gaze_obs density is gaussian around gaze_gt
                gazeAroundHead.append(gaze_gt - head_obs)         # gaze_gt prior is gaussian around head
                nbSamples_gt += 1
        observationAroundGaze = np.array(observationAroundGaze)
        observationAroundHead = np.array(observationAroundHead)
        gazeAroundHead = np.array(gazeAroundHead)

        # Step1: learn observed gaze, estimated gaze densities (gaussian models)
        # Hypothesis here is that there is no bias (i.e. the mean is equal to 0 for all distributions)
        def get_covarianceMat(data2D):
            data2D = np.array(data2D)
            N = len(data2D)
            covMat = np.diag(np.sum(data2D**2, axis=0) / N)
            covMat[0, 1] = np.sum([x*y for x, y in data2D]) / N
            covMat[1, 0] = np.sum([x*y for x, y in data2D]) / N
            return covMat

        self.var_noise = get_covarianceMat(observationAroundGaze)
        # self.var_obsPrior = get_covarianceMat(observationAroundHead)
        # self.var_estPrior = get_covarianceMat(gazeAroundHead)
        self.var_obsPrior = None  # Get better results when normalization constant and prior are ignored
        self.var_estPrior = None

        # Step2: optimize aversion threshold
        # Gather highest probability among targets for each sample
        self.prob_aversion = 0
        aversionProbList, targetProbList = [], []
        for person, targetDict, gt in zip(personList, targetDictList, groundTruth):
            vfoa = self.compute_vfoa(person, targetDict, None)
            maxProb, bestTarget = 0., None
            for targetName, prob in vfoa.items():
                if prob > maxProb:
                    maxProb, bestTarget = prob, targetName
            if gt in targetDict.keys():
                targetProbList.append(maxProb)
                nbSamples_aversion += 1
            elif gt == 'aversion':
                aversionProbList.append(maxProb)
                nbSamples_aversion += 1
        aversionProbList, targetProbList = np.array(aversionProbList), np.array(targetProbList)

        # Search a threshold that optimize the classification accuracy
        bestThresh, score = 0, 0
        for prob in np.concatenate([aversionProbList, targetProbList]):
            thresh_prop = prob
            correct = sum(aversionProbList <= thresh_prop) + sum(targetProbList > thresh_prop)
            score_prop = float(correct) / nbSamples_aversion
            if score_prop > score:
                bestThresh, score = thresh_prop, score_prop
        self.prob_aversion = bestThresh

        print('Training {}, parameters were updated:'.format(self.name))
        print('\tNoise variance: {} ({} samples)'.format(self.var_noise, nbSamples_gt))
        print('\tObservation prior variance: {} ({} samples)'.format(self.var_obsPrior, nbSamples_obs))
        print('\tEstimation prior variance: {} ({} samples)'.format(self.var_estPrior, nbSamples_gt))
        print('\tAversion probability: {} ({} samples)'.format(self.prob_aversion, nbSamples_aversion))
