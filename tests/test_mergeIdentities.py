# coding=utf-8

import numpy as np

from vfoa.vfoa_module import VFOAModule
from vfoa.vfoa_module import Person as VFOAPerson
from vfoa.vfoa_module import Target as VFOATarget

vfoa_model = "HMM" # geometricModel, gazeProbability
vfoa_module = VFOAModule(model=vfoa_model)


class Person:
    def __init__(self, name="", x=0., y=0., z=0., yaw=0., pitch=0., roll=0.):
        self.name = name
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll

frameIndex = 0

persons_list = [Person("subject1", 0., 0., 1., 0., 0., 0.)]

personDict1 = {}
for p in persons_list:
    personDict1[p.name] = VFOAPerson(p.name,
                                     headpose=np.array([p.x, p.y, p.z, p.yaw, p.pitch, p.roll]),
                                     positionCS="OCS",
                                     poseCS="FCS",
                                     poseUnit="deg")

persons_list = [Person("subject2", 0., 0., 1., 45., 0., 0.)]

personDict2 = {}
for p in persons_list:
    personDict2[p.name] = VFOAPerson(p.name,
                                     headpose=np.array([p.x, p.y, p.z, p.yaw, p.pitch, p.roll]),
                                     positionCS="OCS",
                                     poseCS="FCS",
                                     poseUnit="deg")

targetDict = {
    "t1": VFOATarget("t1", np.array([-1.0, 0.0, 0.0]), positionCS="OCS"),
    "t2": VFOATarget("t2", np.array([0.0, 0.0, 0.0]), positionCS="OCS"),
    "t3": VFOATarget("t3", np.array([1.0, 0.0, 0.0]), positionCS="OCS"),
}

for t in range(60):
    if t < 10:  # Subject 1 is detected, vfoa is computed
        pDict = personDict1
    elif t < 20:  # Subject 1 disappear, subject 2 is detected
        pDict = personDict2
    elif t == 20:  # Subject 2 is re-identified as subject 1
        print('MERGE')
        vfoa_module.merge_people("subject2", "subject1")
        pDict = personDict2
    else:  # Continue with subject 2
        pDict = personDict2

    vfoa_module.compute_vfoa(pDict, targetDict, t)
    for key in vfoa_module.personDict:
        vfoa = vfoa_module.get_vfoa(key, normalized=False)
        best = vfoa_module.get_vfoa_best(key)
        print(key, "{:06d}".format(t), vfoa, best)
        print('Tracked people: {}'.format(vfoa_module.vfoaDict.keys()))
        # for t, hp in vfoa_module.modelDict[key].person.headpose:
        #     print '\t', t, hp
        print
