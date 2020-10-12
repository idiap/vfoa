# coding=utf-8

import os
import sys
import logging
import argparse

import numpy as np

from vfoa.vfoa_module import VFOAModule
from vfoa.vfoa_module import Person as VFOAPerson
from vfoa.vfoa_module import Target as VFOATarget
from vfoa.utils.geometry import vectorToYawElevation

vfoa_model = "HMM" # geometricModel, gazeProbability
vfoa_module = VFOAModule(model=vfoa_model)


class Person:
    def __init__(self, name="", x=0, y=0, z=0, yaw=0, pitch=0, roll=0):
        self.name = name
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll

frameIndex = 0

# persons_list = [
#     Person("alphonse", 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
#     Person("bob", -1.0, 0.0, 1.0, 45.0, 0.0, 0.0)
# ]

persons_list = [
    Person("alphonse", 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    Person("bob", -1.0, 0.0, 1.0, 0.0, 0.0, 0.0)
]


personDict = {}
for p in persons_list:
    personDict[p.name] = VFOAPerson(p.name,
                                    headpose=np.array([p.x, p.y, p.z,
                                              p.yaw, p.pitch, p.roll]),
                                    positionCS="OCS",
                                    poseCS="FCS",
                                    poseUnit="deg")

# for _, p in personDict.items():
#     p.print_person()

targetDict = {
    "robot": VFOATarget("robot", np.array([0.0, 0.0, 0.0]), positionCS="OCS"),
    # "screen": VFOATarget("screen", np.array([-1.0, 0.0, 0.0]), positionCS="OCS"),
    # "bob": VFOATarget("bob", np.array([-1.0, 0.0, 1.0]), positionCS="OCS"),
}

for p in persons_list:
    targetDict[p.name] = VFOATarget(p.name,
                                    position=np.array([p.x, p.y, p.z]),
                                    positionCS="OCS")

# vfoa_module.compute_vfoa(personDict, targetDict, frameIndex)

# for name in personDict:
#     print("#"*20, name)
#     vfoa = vfoa_module.get_vfoa(name, normalized=False)
#     print(vfoa)
#     best = vfoa_module.get_vfoa_best(name)
#     print(best)

# vfoa = self.vfoa_module.get_vfoa()
# # print(vfoa)
# for p in vfoa:
#     best_target = None
#     best_score = 0
#     for i, (target, score) in enumerate(vfoa[p].items()):
#         if i == 0 or score > best_score:
#             best_score = score
#             best_target = target
#     print("{} is looking at {} ({})".format(p, best_target, best_score))

name = "alphonse"

for t in range(50):
    vfoa_module.compute_vfoa(personDict, targetDict, t)
    vfoa = vfoa_module.get_vfoa(name, normalized=False)
    best = vfoa_module.get_vfoa_best(name)
    print("{:06d}".format(t), vfoa, best)
    print len(vfoa_module.modelDict['alphonse'].person.headpose), vfoa_module.modelDict['alphonse'].person.headpose[0]

personDict["alphonse"] = VFOAPerson("alphonse",
                                    headpose=np.array([0., 0., 1.,
                                                       55., 0., 0.]),
                                    positionCS="OCS",
                                    poseCS="FCS",
                                    poseUnit="deg")

t = 50
vfoa_module.compute_vfoa(personDict, targetDict, t)
vfoa = vfoa_module.get_vfoa(name, normalized=False)
best = vfoa_module.get_vfoa_best(name)
print("{:06d}".format(t), vfoa, best)
print len(vfoa_module.modelDict['alphonse'].person.headpose), vfoa_module.modelDict['alphonse'].person.headpose[0]
