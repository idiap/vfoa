"""
Copyright (c) 2018 Idiap Research Institute, http://www.idiap.ch/
Written by Remy Siegfried <remy.siegfried@idiap.ch>

This file contains the main class to use in order to estimate vfoa.
"""

from copy import deepcopy
# Import methods
from vfoa.methods.geometrical_model import GeometricModel
from vfoa.methods.gaussian_model import GaussianModel
from vfoa.methods.gaze_probability import GazeProbability
from vfoa.methods.kalman_filter import KalmanFilter
from vfoa.methods.HMM import HMM
# from vfoa.methods.headHMM import HeadHMM
# from vfoa.methods.gazeHMM import GazeHMM
# Import data_structures allowing easier import for user ("from vfoa.vfoa_module import VFOAModule, Person, Target")
from vfoa.data_structures.person import Person
from vfoa.data_structures.target import Target


class VFOAModule:
    def __init__(self, model='gaussianModel', history_duration=None):
        # Method
        self.modelDict = {}
        self.model_ref = None
        self.model_parameters = None

        # Set reference model (i.e. model that will be used for a new person)
        if model == 'gaussianModel':
            self.model_ref = GaussianModel
        elif model == 'geometricalModel':
            self.model_ref = GeometricModel
        elif model == 'gazeProbability':
            self.model_ref = GazeProbability
        elif model == 'kalmanFilter':
            self.model_ref = KalmanFilter
        elif model == 'HMM':
            self.model_ref = HMM
        # elif model == 'headHMM':
        #     self.model_ref = HeadHMM
        # elif model == 'gazeHMM':
        #     self.model_ref = GazeHMM
        else:  # Default
            raise Exception('[VFOAModule] Unknown model: {}'.format(model))

        # Parameters
        self.history_duration = history_duration  # time after which unupdated person must be deleted (in seconds)

        # Scene information (last given by user)
        self.timestamp = 0
        self.personDict = {}  # List of subjects whose vfoa will be estimated
        self.targetDict = {}  # List of possible vfoa targets including persons

        # Output
        self.vfoaTimestampDict = {}
        self.vfoaProbDict = {}
        self.vfoaDict = {}

    def set_model_parameters(self, parameterList):
        self.model_parameters = parameterList

    def compute_vfoa(self, persons=None, targets=None, timestamp=None):
        """ <persons>: dict of {'name': Person}
            <targets>: dict of {'name': Target}
            <timestamp>: float (in seconds)
            Update information about the scene wrt given persons and targets and compute the vfoa for each person.
            Each Person/Target must have a unique name. """

        # Update information
        self.timestamp = timestamp
        self.personDict = persons
        self.targetDict = targets

        # Add each person as target (not always wanted)
        # for personName, person in persons.items():
        #     self.targetDict[personName] = Target(personName, person.headpose[0:3])

        # Compute vfoa
        for personName, person in self.personDict.items():
            # Create model for the person if needed
            if personName not in self.modelDict.keys():
                # Initialize variables related to this new person (model and outputs)
                self.modelDict[personName] = self.model_ref()
                if self.model_parameters is not None:
                    self.modelDict[personName].set_parameters(self.model_parameters)
                self.vfoaDict[personName] = None
                self.vfoaProbDict[personName] = None

            self.modelDict[personName].compute_vfoa(person, self.targetDict, timestamp)
            self.vfoaProbDict[personName] = self.modelDict[personName].get_vfoa_prob()
            self.vfoaDict[personName] = self.modelDict[personName].get_vfoa_distribution()
            self.vfoaTimestampDict[personName] = timestamp

        # Remove un-updated people after given duration
        if self.history_duration is not None:
            for personName in self.modelDict.keys():
                if self.vfoaTimestampDict[personName] < self.timestamp - self.history_duration:
                    self.remove_person(personName)

    def merge_people(self, personName_ref, personName_toMerge):
        if hasattr(self.modelDict[personName_ref], 'merge_person'):
            personData = self.modelDict[personName_toMerge].person
            personTimestamp = self.vfoaTimestampDict[personName_toMerge]
            self.modelDict[personName_ref].merge_person(personData, personTimestamp)
        self.remove_person(personName_toMerge)

    def remove_person(self, personName):
        del self.modelDict[personName]
        del self.vfoaProbDict[personName]
        del self.vfoaDict[personName]
        del self.vfoaTimestampDict[personName]

    def train(self, personList, targetDictList, groundTruth):
        """ <personList>: list of Person
            <targetList>: list of dict {'name': Target}
            <groundTruth>: list of str indicating which target the person is looking at. """
        for person in personList:
            self.modelDict[person.name].train([person], targetDictList, groundTruth)

    def get_vfoa(self, personName=None, normalized=False):
        """ Return vfoa (dict of {'targetName': probability} of the wanted <personName>, or the vfoa for all persons
            if <personName> is None.) """
        vfoa = self.vfoaProbDict if not normalized else self.vfoaDict
        if personName:
            return deepcopy(vfoa[personName])
        else:
            return deepcopy(vfoa)

    def get_vfoa_best(self, personName=None):
        """ Return the target with the highest probability to be the vfoa target of the wanted <personName>, or of
            all persons if <personName> is None.)"""
        if personName is not None:
            maxProb, bestTarget = 0., None
            for targetName, prob in self.vfoaProbDict[personName].items():
                if prob > maxProb:
                    maxProb, bestTarget = prob, targetName
            return bestTarget
        else:
            bestTargetsDict = {}
            for personName, vfoa in self.vfoaProbDict.items():
                maxProb, bestTarget = 0., None
                for targetName, prob in self.vfoaProbDict[personName].items():
                    if prob > maxProb:
                        maxProb, bestTarget = prob, targetName
                bestTargetsDict[personName] = bestTarget
            return bestTargetsDict
