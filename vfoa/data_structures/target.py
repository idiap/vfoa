"""
Copyright (c) 2018 Idiap Research Institute, http://www.idiap.ch/
Written by Remy Siegfried <remy.siegfried@idiap.ch>

This file contains the a class to gather data relative to a target.
"""

import numpy as np


class Target:
    def __init__(self, name, position=None, positionCS='CCS'):
        """ <name> is a string identifying the person. It should be unique among the given people and targets
            <position> is a 1 dimensional list giving position [x, y, z]
            <positionCS> is a string in ['CCS', 'OCS'] indicating the coordinate system in which the positions are
                defined """

        self.name = name
        # Coordinate systems and units (string)
        self.positionCS = positionCS
        # Tracking data [x, y, z]
        self.position = np.array(position, dtype=np.float32) if position is not None else None

    def convert_to(self, positionCS='CCS'):
        self._transform_positionCS(self.positionCS, positionCS)

    def _transform_positionCS(self, positionCS_old, positionCS_new):
        if positionCS_old != positionCS_new:
            success = False
            if positionCS_old == 'CCS' and positionCS_new == 'OCS':
                success = True
                self.positionCS = positionCS_new
                if self.position is not None:
                    self.position *= np.array([1, -1, -1])
            elif positionCS_old == 'OCS' and positionCS_new == 'CCS':
                success = True
                self.positionCS = positionCS_new
                if self.position is not None:
                    self.position *= np.array([1, -1, -1])

            if not success:
                raise Exception('[vfoa_module] Unable to convert position coordinate system from {} to {}'.format(positionCS_old, positionCS_new))

    def print_target(self):
        print('Instance of class <Target>:')
        print('\tname:\t\t{}'.format(self.name))
        print('\tposition:\t{}'.format([round(p, 3) for p in self.position] if self.position is not None else None))
        print('\t-----')
        print('\tposition CS:\t{}'.format(self.positionCS))
