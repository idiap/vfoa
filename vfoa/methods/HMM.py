"""
Copyright (c) 2018 Idiap Research Institute, http://www.idiap.ch/
Written by Remy Siegfried <remy.siegfried@idiap.ch>

This file contains a HMM-based vfoa estimation method, from the work of Salim Kayal <salim.kayal@idiap.ch> and
Vasil Khalidov <Vasil.Khalidov@idiap.ch>.
"""

from copy import deepcopy
from collections import OrderedDict
import numpy as np

from vfoa.utils.geometry import vectorToYawElevation
from vfoa.data_structures.person import Person
from vfoa.data_structures.target import Target

DEBUG = False

PRIOR_STD = np.array([12 *np.pi/180, 13 *np.pi/180])
C_PAN, C_TILT = 1., 1.
PRIOR_UNFOCUSED = (1.0 / (2 * np.pi * PRIOR_STD[0])) * (1.0 / (2 * np.pi * PRIOR_STD[1])) * np.exp(-.5 * (C_PAN**2 + C_TILT**2))  # 0.196


class HMM:
    def __init__(self):
        # By default, var_obsPrior and var_estPrior are not used as it seems to improve the results
        self.name = 'HMM'

        self.person = None
        self.targetDict = OrderedDict({'unfocused': None})
        self.transitionMat = None
        self.prior = None
        self.prob = None  # Probability of each target against unfocused separately
        self.vfoa_prob = None  # Unnormalized vfoa
        self.vfoa = None  # Normalized distribution

        # Parameters - compute previous head pose and reference head pose from past data
        self.headpose_history_duration = 35
        self.headpose_ref_duration = 30  # Span of head poses use to compute reference head pose (in sec)
        self.headpose_ref_gap = 0  # Gap between current timestamp and head poses use to compute reference head pose (in sec)
        self.headpose_prev_duration = 5  # Span of head poses use to compute previous head pose (in sec)
        self.headpose_prev_gap = 5  # Gap between current timestamp and head poses use to compute previous head pose (in sec)
        
        # Parameters - head-gaze model
        self.headgaze_ref_factor = np.array([.4, .6])  # Contribution of the headpose in both dimension for Dynamical Head Reference model
        self.headgaze_prev_factor = np.array([0., 0.])  # Use only Dynamical Head Reference model (no impact of old headpose)
        # self.headgaze_prev_factor = np.array([1., 1.])  # Use Midline Effect model (take into account the old headpose to know where)
        
        # Parameters - probabilities
        self.prior_min = .001  # Target prior can not be smaller than this minimum
        self.prior_std = np.array([12 * np.pi / 180, 13 * np.pi / 180])  # Std of the gaussian prior distribution
        self.prior_unfocused = PRIOR_UNFOCUSED
        self.default_prob_ii = .6  # Set conservatice prob in transition matrix for independent target probability (i.e. target VS unfocused)

        if DEBUG:
            self.data = None

    def set_parameters(self, parameterList):
        if parameterList is None or len(parameterList) != 11:
            raise Exception('given parameters ({}) do not fit the model. Need 11 parameters'.format(parameterList))
        headpose_history_duration = parameterList[0]  # int
        headpose_ref_duration = parameterList[1]  # int
        headpose_ref_gap = parameterList[2]  # int
        headpose_prev_duration = parameterList[3]  # int
        headpose_prev_gap = parameterList[4]  # int
        headgaze_ref_factor = parameterList[5]  # np.array, shape=(2,)
        headgaze_prev_factor = parameterList[6]  # np.array, shape=(2,)
        prior_min = parameterList[7]  # float
        prior_std = parameterList[8]  # np.array, shape=(2,)
        prior_unfocused = parameterList[9]  # float
        default_prob_ii = parameterList[10]  # float

        if headpose_history_duration is not None:
            self.headpose_history_duration = int(headpose_history_duration)
        if headpose_ref_duration is not None:
            self.headpose_ref_duration = int(headpose_ref_duration)
        if headpose_ref_gap is not None:
            self.headpose_ref_gap = int(headpose_ref_gap)
        if headpose_prev_duration is not None:
            self.headpose_prev_duration = int(headpose_prev_duration)
        if headpose_prev_gap is not None:
            self.headpose_prev_gap = int(headpose_prev_gap)

        if headgaze_ref_factor is not None:
            self.headgaze_ref_factor = np.array(headgaze_ref_factor)
        if headgaze_prev_factor is not None:
            self.headgaze_prev_factor = np.array(headgaze_prev_factor)

        if prior_min is not None:
            self.prior_min = float(prior_min)
        if prior_std is not None:
            self.prior_std = np.array(prior_std)
        if prior_unfocused is not None:
            self.prior_unfocused = float(prior_unfocused)
        if default_prob_ii is not None:
            self.default_prob_ii = float(default_prob_ii)

    def merge_person(self, personToMerge, timestamp):
        """ <personToMerge> is a Person object, whose data will (head pose for this model) be merged to the data of the
            person concerned by this model. """
        if personToMerge.headpose.__class__ == list:  # The person to merge had already its headpose transformed to list
            personToMerge.headpose.extend(self.person.headpose)
            self.person.headpose = personToMerge.headpose
        else:
            self._update_personData(personToMerge, timestamp)

        # Make sure that data are sorted
        self.person.headpose.sort(key=lambda x: x[0], reverse=True)

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

    def _compute_transitionMatrix(self):
        transitionMat = np.zeros((len(self.vfoa.keys()), len(self.targetDict.keys())))
        default_prob_ij = (1. - self.default_prob_ii) / (len(self.targetDict.keys()) - 1)
        for i, key_i in enumerate(self.vfoa.keys()):
            if key_i not in self.targetDict.keys():  # No conservative probability
                transitionMat[i, :] = 1. / len(self.targetDict.keys())
            else:
                for j, key_j in enumerate(self.targetDict.keys()):
                    transitionMat[i, j] = self.default_prob_ii if key_i == key_j else default_prob_ij
        return transitionMat

    def _update_targetData(self, targetDict, personName, timestamp):
        self.targetDict = OrderedDict()
        for key in sorted(targetDict.keys()):
            if key != personName:
                self.targetDict[key] = targetDict[key]
        self.targetDict['unfocused'] = Target('unfocused')

    def _update_personData(self, person, timestamp):
        """ <person>: Person object """
        newHeadpose = [[timestamp, deepcopy(person.headpose)]]
        if self.person is None:
            # Initialization
            distribArray = np.ones(len(self.targetDict.keys())) / len(self.targetDict.keys())
            self.vfoa = OrderedDict(zip(self.targetDict.keys(), distribArray))
        else:
            # Fill memory
            newHeadpose.extend(self.person.headpose)

            # Update
            if person.name != self.person.name:
                raise Exception('[VFOAModule - HMM] Wrong person detected.' +
                                'To handle another person, please instantiate a new object')

        self.person = deepcopy(person)
        self.person.headpose = newHeadpose

        # Discard old data (make the hypothesis than head poses are sorted in ascending order of timestamp
        headposeList = self.person.headpose
        for i in range(len(headposeList)):
            if headposeList[0][0] - headposeList[i][0] > self.headpose_history_duration:
                self.person.headpose = headposeList[:i]
                break

    def _get_meanHeadpose(self, name, timestamp, gap=5, duration=10):
        """ Return the mean of the headposes recording of the <name> subject filling a temporal constraint:
            they were recorded at most <duration> seconds before the actual <timestamp> and at least <gap> seconds
            (leaving a gap with the present measure) """
        fullLength = self.person.headpose[0][0] - self.person.headpose[-1][0]
        gap = fullLength if gap > fullLength else gap
        duration = fullLength - gap if duration + gap > fullLength else duration

        headposeList = []
        for headpose in self.person.headpose:
            if gap < timestamp - headpose[0] < duration:
                headposeList.append(headpose[1].copy())
        return np.mean(headposeList, axis=0) if len(headposeList) > 0 else self.person.headpose[-1][1].copy()

    def _compute_headGazeModel(self, targetDirection, headpose_ref, headpose_prev):
        """ Compute the theoretical position of the head according to head-gaze model, given that the person is
            looking at a known target. """
        headpose_ref = headpose_ref[3:5]
        headpose_prev = headpose_prev[3:5]

        # Dynamical Head Reference model (head1)
        head1 = self.headgaze_ref_factor * headpose_ref + (np.array([1., 1.]) - self.headgaze_ref_factor) * targetDirection[0:2]

        # Midline Effect model (head2)
        head2 = np.zeros(2)
        for dim in [0, 1]:  # Same model for pan and tilt (but different REF, PREV factors)
            # if targetDirection[dim] - headpose_ref[dim] >= 0:  # TODO: THIS SHOULD BE THE CORRECT WAY ?!
            if targetDirection[dim] >= 0:
                # if headpose_prev[dim] < head1[dim]:  # In humavips tracker was "headpose_prev[dim] < targetDirection[dim]" TODO: THIS SHOULD BE THE CORRECT WAY ?!
                if headpose_prev[dim] < targetDirection[dim]:  # In humavips tracker was "headpose_prev[dim] < targetDirection[dim]"
                    head2[dim] = head1[dim]
                else:
                    midline = self.headgaze_prev_factor[dim] * headpose_prev[dim] + (1 - self.headgaze_prev_factor[dim]) * head1[dim]
                    head2[dim] = min(targetDirection[dim], midline)
            else:
                # if headpose_prev[dim] > head1[dim]:  # TODO: THIS SHOULD BE THE CORRECT WAY ?!
                if headpose_prev[dim] < targetDirection[dim]:
                    head2[dim] = head1[dim]
                else:
                    midline = self.headgaze_prev_factor[dim] * headpose_prev[dim] + (1 - self.headgaze_prev_factor[dim]) * head1[dim]
                    head2[dim] = max(targetDirection[dim], midline)

        return head2

    def compute_vfoa(self, person, targetDict, timestamp):
        """ <person>: Person object (Note: it should be always the same person.)
            <targetDict>: {'targetname': Target object}
            <gazeFormat>: should be 'vector' or 'angles' """
        # Check coordinate systems and units
        person.convert_to(positionCS='CCS', poseCS='FCS', poseUnit='rad')
        for key in targetDict.keys():
            targetDict[key].convert_to(positionCS='CCS')

        self._update_targetData(targetDict, person.name, timestamp)
        self._update_personData(person, timestamp)

        # Find head pose references
        if self.person.bodypose is None:
            headpose_ref = self._get_meanHeadpose(person.name, timestamp, gap=self.headpose_ref_gap,
                                                  duration=self.headpose_ref_duration)
        else:
            headpose_ref = self.person.bodypose
        headpose_prev = self._get_meanHeadpose(person.name, timestamp, gap=self.headpose_prev_gap,
                                               duration=self.headpose_prev_duration)

        # Compute prior and probability for each target separately (i.e. target VS unfocused)
        old_prob = deepcopy(self.prob)
        currentHeadPosition = self.person.headpose[0][1][0:3]
        currentHeadpose = self.person.headpose[0][1][3:5]
        self.prior = OrderedDict()
        self.prob = OrderedDict()
        for targetName, target in self.targetDict.items():
            if targetName != person.name:
                if target.position is not None:
                    ### Compute prior (based on head-gaze coordination model)

                    # Compute theoretical headpose (if the person looks straight to the target)
                    cameraDirection = np.array(vectorToYawElevation(np.array([0, 0, 0]) - currentHeadPosition, unit='rad'))
                    targetDirection = np.array(vectorToYawElevation(np.array(target.position) - currentHeadPosition, unit='rad'))
                    targetDirection = targetDirection - cameraDirection  # Classical way: remove this, but worse performances

                    theoreticalHeadpose = self._compute_headGazeModel(targetDirection, headpose_ref, headpose_prev)

                    # Compute prior
                    # prior = self._gaussian(currentHeadpose, theoreticalHeadpose, np.diag(self.prior_std))  # TODO: THIS SHOULD BE THE CORRECT WAY ?!
                    prior = self._gaussian(currentHeadpose[0], theoreticalHeadpose[0], self.prior_std[0]**2) *\
                            self._gaussian(currentHeadpose[1], theoreticalHeadpose[1], self.prior_std[1]**2)
                    self.prior[targetName] = max(prior, self.prior_min)

                    ### TEST: use gaze instead of head pose
                    # prior = self._gaussian(self.person.gaze[3:5], targetDirection, np.diag(self.prior_std))
                    # self.prior[targetName + '_gaze'] = max(prior, self.prior_min)
                    ###

                    ### Compute the probability that the person looks to the target VS unfocused

                    # Recover last target probability (first: unfocused, second: this target)
                    if old_prob is not None and targetName in old_prob.keys():
                        probVSunfocused = np.array([1 - old_prob[targetName], old_prob[targetName]])
                    else:
                        probVSunfocused = np.array([.5, .5])

                    # Compute transition probabilities (target vs aversion)
                    default_trans_mat = np.array([[self.default_prob_ii, 1 - self.default_prob_ii],
                                                  [1 - self.default_prob_ii, self.default_prob_ii]])
                    trans_prob = np.dot(probVSunfocused, default_trans_mat)

                    # Compute posterior (target vs aversion)
                    posterior = trans_prob * np.array([self.prior_unfocused, self.prior[targetName]])
                    posterior /= np.sum(posterior)
                    self.prob[targetName] = posterior[1]

        # Add unfocused prior
        self.prior['unfocused'] = self.prior_unfocused
        self.prob['unfocused'] = self.prior_unfocused

        # Compute general transition matrix
        transitionMat = self._compute_transitionMatrix()

        # Compute probability distribution
        distrib_trans = np.dot(self.vfoa.values(), transitionMat)

        # Compute posterior distribution
        posterior = distrib_trans * self.prior.values()
        self.vfoa_prob = OrderedDict(zip(self.targetDict.keys(), posterior))
        posterior /= np.sum(posterior)
        self.vfoa = OrderedDict(zip(self.targetDict.keys(), posterior))

        if DEBUG:
            self.data = np.vstack([self.data, self.vfoa.values()]) if self.data is not None else self.vfoa.values()
            # self.data = np.vstack([self.data, self.prior.values()]) if self.data is not None else self.prior.values()
            # self.data = np.vstack([self.data, self.vfoa_prob.values()]) if self.data is not None else self.vfoa_prob.values()
            if len(self.data) == 1000:
                import matplotlib.pyplot as plt
                plt.figure()
                for i in range(self.data.shape[1]):
                    plt.plot(self.data[:, i], label=self.vfoa.keys()[i])
                plt.legend(loc=1)
                plt.show()
                # quit()

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
