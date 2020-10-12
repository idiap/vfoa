"""
Copyright (c) 2018 Idiap Research Institute, http://www.idiap.ch/
Written by Remy Siegfried <remy.siegfried@idiap.ch>

This file contains the a class to gather data relative to a person.
"""

import numpy as np

from vfoa.utils.geometry import vectorToYawElevation, yawElevationToVector


class Person:
    def __init__(self, name, headpose=None, gaze=None, bodypose=None, speaking=None,
                 positionCS='CCS', poseCS='CCS', poseUnit='deg'):
        """ <name> is a string identifying the person. It should be unique among the given people and targets
            <headpose>, <gaze> and <bodypose> are 1 dimensional lists giving position and orientation (in
                angle) [x, y, z, yaw, pitch, roll]
            <positionCS> is a string in ['CCS', 'OCS'] indicating the coordinate system in which the positions are defined
            <poseCS> is a string in ['CCS', 'FCS'] indicating the coordinate system in which the orientations are defined
            <poseUnit> is a string in ['deg', 'rad'] indicating the unit of the given angles """
        # Input checks
        if poseCS == 'FCS' and headpose is None:
            raise Exception('[vfoa_module] FCS coordinate system has no sense if headpose is not set')

        self.name = name
        # Coordinate systems and units (string)
        self.positionCS = positionCS
        self.poseCS = poseCS
        self.poseUnit = poseUnit
        # Tracking data [x, y, z, yaw, pitch, roll]
        self.headpose = np.array(headpose, dtype=np.float32) if headpose is not None else None
        self.gaze = np.array(gaze, dtype=np.float32) if gaze is not None else None
        self.bodypose = np.array(bodypose, dtype=np.float32) if bodypose is not None else None
        # Binary states
        self.speaking = speaking

        # Ensure that angles are defined as [yaw, elevation] (i.e. [yaw, -pitch])
        if poseCS == 'FCS':
            for cue in [self.headpose, self.gaze, self.bodypose]:
                if cue is not None:
                    cue[3] *= -1.

    def convert_to(self, positionCS='CCS', poseCS='CCS', poseUnit='deg'):
        self._transform_positionCS(self.positionCS, positionCS)
        self._transform_poseCS(self.poseCS, poseCS)
        self._transform_poseUnit(self.poseUnit, poseUnit)

    def _transform_positionCS(self, positionCS_old, positionCS_new):
        if positionCS_old != positionCS_new:
            success = False
            if positionCS_old == 'OCS' and positionCS_new == 'CCS':
                success = True
                self.positionCS = positionCS_new
                for cue in [self.headpose, self.gaze, self.bodypose]:
                    if cue is not None:
                        cue[0:3] *= np.array([1, -1, -1])
            elif positionCS_old == 'CCS' and positionCS_new == 'OCS':
                success = True
                self.positionCS = positionCS_new
                for cue in [self.headpose, self.gaze, self.bodypose]:
                    if cue is not None:
                        cue[0:3] *= np.array([1, -1, -1])

            if not success:
                raise Exception('[vfoa_module] Unable to convert position coordinate system from {} to {}'.format(positionCS_old, positionCS_new))

    def _transform_poseCS(self, poseCS_old, poseCS_new):
        if poseCS_old != poseCS_new:
            success = False
            if poseCS_old == 'FCS' and poseCS_new == 'CCS':
                success = True
                self.poseCS = poseCS_new
                for cue in [self.headpose, self.gaze, self.bodypose]:
                    if cue is not None:
                        if self.headpose is not None:
                            # Compute pose vector in FCS
                            vec_fcs = yawElevationToVector(cue[3:5])
                            transform = np.reshape([[0, 0, 1], [1, 0, 0], [0, 1, 0]], (3, 3))  # In yawElevationToVector, basis is not defined as in the FCS
                            vec_fcs = np.dot(transform, vec_fcs)

                            # Compute transform from FCS to CCS (only pose, no translation needed)
                            R_fcs_ccs = np.zeros((3, 3))
                            if self.positionCS == 'CCS':
                                head_position = self.headpose[0:3].copy()
                            elif self.positionCS == 'OCS':
                                head_position = self.headpose[0:3].copy() * np.array([1, -1, -1])
                            else:
                                raise Exception('No reference position to convert FCS to CCS (position should already be given in CCS)')
                            R_fcs_ccs[:, 0] = -head_position / np.linalg.norm(head_position)
                            R_fcs_ccs[:, 1] = np.cross([0.0, 1.0, 0.0], R_fcs_ccs[:, 0])
                            R_fcs_ccs[:, 2] = np.cross(R_fcs_ccs[:, 0], R_fcs_ccs[:, 1])

                            # Compute angles in CCS
                            vec_ccs = np.dot(R_fcs_ccs, vec_fcs)
                            cue[3:5] = vectorToYawElevation(vec_ccs)
                        else:
                            raise Exception('[vfoa_module] FCS coordinate system has no sense if headpose is not set')
            elif poseCS_old == 'CCS' and poseCS_new == 'FCS':
                success = True
                self.poseCS = poseCS_new
                for cue in [self.headpose, self.gaze, self.bodypose]:
                    if cue is not None:
                        if self.headpose is not None:
                            # Compute vector in CCS
                            vec_ccs = yawElevationToVector(cue[3:5])

                            # Compute transform from FCS to CCS (only pose, no translation needed)
                            R_fcs_ccs = np.zeros((3, 3))
                            if self.positionCS == 'CCS':
                                head_position = self.headpose[0:3].copy()
                            elif self.positionCS == 'OCS':
                                head_position = self.headpose[0:3].copy() * np.array([1, -1, -1])
                            else:
                                raise Exception('No reference position to convert FCS to CCS (position should already be given in CCS)')
                            R_fcs_ccs[:, 0] = -head_position / np.linalg.norm(head_position)
                            R_fcs_ccs[:, 1] = np.cross([0.0, 1.0, 0.0], R_fcs_ccs[:, 0])
                            R_fcs_ccs[:, 2] = np.cross(R_fcs_ccs[:, 0], R_fcs_ccs[:, 1])
                            R_ccs_fcs = np.linalg.inv(R_fcs_ccs)

                            # Compute angles in CCS
                            vec_ccs = np.dot(R_ccs_fcs, vec_ccs)
                            transform = np.reshape([[0, 1, 0], [0, 0, 1], [1, 0, 0]], (3, 3))  # In vectorToYawElevation, basis is not defined as in the FCS
                            vec_ccs = np.dot(transform, vec_ccs)
                            cue[3:5] = vectorToYawElevation(vec_ccs)
                        else:
                            raise Exception('[vfoa_module] FCS coordinate system has no sense if headpose is not set')

            if not success:
                raise Exception('[vfoa_module] Unable to convert pose coordinate system from {} to {}'.format(poseCS_old, poseCS_new))

    def _transform_poseUnit(self, poseUnit_old, poseUnit_new):
        if poseUnit_old != poseUnit_new:
            success = False
            if poseUnit_old == 'deg' and poseUnit_new == 'rad':
                success = True
                self.poseUnit = poseUnit_new
                for cue in [self.headpose, self.gaze, self.bodypose]:
                    if cue is not None:
                        cue[3:6] *= np.pi / 180
            elif poseUnit_old == 'rad' and poseUnit_new == 'deg':
                success = True
                self.poseUnit = poseUnit_new
                for cue in [self.headpose, self.gaze, self.bodypose]:
                    if cue is not None:
                        cue[3:6] *= 180 / np.pi

            if not success:
                raise Exception('[vfoa_module] Unable to convert pose unit from {} to {}'.format(poseUnit_old, poseUnit_new))

    def print_person(self):
        print('Instance of class <Person>:')
        print('\tname:\t\t{}'.format(self.name))
        print('\theadpose:\t{}'.format([round(hp, 3) for hp in self.headpose] if self.headpose is not None else None))
        print('\tgaze:\t\t{}'.format([round(g, 3) for g in self.gaze] if self.gaze is not None else None))
        print('\tbodypose:\t{}'.format([round(b, 3) for b in self.bodypose] if self.bodypose is not None else None))
        print('\tspeaking:\t{}'.format(self.speaking))
        print('\t-----')
        print('\tposition CS:\t{}'.format(self.positionCS))
        print('\tpose CS:\t{}'.format(self.poseCS))
        print('\tpose unit:\t{}'.format(self.poseUnit))
