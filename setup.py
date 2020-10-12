import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name='vfoa',
      version='0.2',
      description='Visual Focus of Attention Estimation Module',
      long_description=read('README.md'),
      url='https://gitlab.idiap.ch/rsiegfried/vfoa',
      author='Remy Siegfried',
      author_email='remy.siegfried@idiap.ch',
      license='BSDv3',
      packages=find_packages(),
      install_requires=['numpy'],
      include_package_data=False,
      zip_safe=False)
