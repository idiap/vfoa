"""
Copyright (c) 2018 Idiap Research Institute, http://www.idiap.ch/
Written by Remy Siegfried <remy.siegfried@idiap.ch>

This file contains a vfoa estimation method based on gaze posterior.
A Kalman Filter is used to refine the extracted gaze and the posterior of the gaze is used to assess the probability
that the subject is looking in the direction of a given target.
"""

from copy import deepcopy
import numpy as np

from vfoa.utils.geometry import vectorToYawElevation


class KalmanFilter:
    def __init__(self, var_gaze=2., var_noise=36., var_prior=625, prob_aversion=0.0001):
        self.name = 'kalmanFilter'

        # Parameters
        self.var_gaze = var_gaze*np.eye(2)                                 # variance of gaze dynamic
        self.var_noise = var_noise*np.eye(2)                               # variance of gaze observation (noise)
        self.var_prior = None if var_prior is None else var_prior*np.eye(2)  # variance of gaze prior (around headpose)
        self.prob_aversion = prob_aversion

        # Gaze posterior parameters (normal distribution : params = (pi, mu, sigma))
        self.timestamp = None
        self.post_pi = None
        self.post_mu = None
        self.post_sigma = None

        self.vfoa_prob = None  # Unnormalized vfoa
        self.vfoa = None  # Normalized distribution

    def set_parameters(self, parameterList):
        if parameterList is None or len(parameterList) != 4:
            raise Exception('given parameters ({}) do not fit the model. Need 4 parameters'.format(parameterList))
        var_gaze, var_noise, var_prior, prob_aversion = parameterList

        if var_gaze is not None:
            self.var_gaze = var_gaze
        if var_noise is not None:
            self.var_noise = var_noise
        if var_prior is not None:
            self.var_prior = var_prior
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

            cst = 1 / ((2 * np.pi) ** n * np.linalg.norm(covar)) ** .5
            exponential = np.exp(-.5 * (x - mu).dot(covar_inv).dot((x - mu).transpose()))
            return cst * exponential

        else:
            return 1 / (2 * np.pi * covar) ** .5 * np.exp(-.5 * (x - mu) ** 2 / covar)

    def compute_vfoa(self, person, targetDict, timestamp=None):
        """ <person>: instance of Person class
            <targetDict>: {'targetname': Target}
            <timestamp>: float """
        # Check coordinate systems and units
        person.convert_to(positionCS='CCS', poseCS='CCS', poseUnit='deg')
        for key in targetDict.keys():
            targetDict[key].convert_to(positionCS='CCS')

        probLabels = []
        probList = []

        # Adapt dynamic to elapsed time since last data
        if timestamp is not None and self.timestamp is not None:
            dt = timestamp - self.timestamp
            self.timestamp = timestamp
        else:
            dt = 1. / 30  # 30 fps by default
            self.timestamp = timestamp

        # Initialize parameters based on prior
        if self.post_mu is None:
            self.post_mu = person.headpose[3:5]
            self.post_sigma = self.var_prior - dt * self.var_gaze

        # Compute new gaze posterior parameters
        var_dyn = self.post_sigma + dt * self.var_gaze
        totalVar = self.post_sigma + dt * self.var_gaze + self.var_noise

        self.post_pi = self._gaussian(person.gaze[3:5], self.post_mu, totalVar)
        self.post_pi /= self._gaussian(person.gaze[3:5], person.headpose[3:5], self.var_prior)
        self.post_sigma = np.linalg.inv(np.linalg.inv(self.var_noise) + np.linalg.inv(var_dyn))
        self.post_mu = np.linalg.inv(self.var_noise).dot(person.gaze[3:5]) + np.linalg.inv(var_dyn).dot(self.post_mu)
        self.post_mu = self.post_sigma.dot(self.post_mu)

        # Compute probability for each target
        for targetName, target in targetDict.items():
            if targetName != person.name:
                if target.position is not None:
                    targetDirection = vectorToYawElevation(target.position - person.gaze[0:3])
                    probLabels.append(targetName)
                    probList.append(self.post_pi * self._gaussian(targetDirection, self.post_mu, self.post_sigma))
                else:
                    probLabels.append(targetName)
                    probList.append(0)

        # Compute aversion given the probability to look at any target and threshold
        if self.prob_aversion is not None:
            probLabels.append('aversion')
            probList.append(self.prob_aversion)

        # Build vfoa
        self.vfoa, self.vfoa_prob = {}, {}
        for label, prob in zip(probLabels, probList):
            self.vfoa[label] = prob
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
