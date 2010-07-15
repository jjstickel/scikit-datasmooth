#! /usr/bin/env python
# Last Change: 7/14/10

descr   = """This is a scikit intended to include numerical methods for smoothing
data. """

import os
import sys

DISTNAME            = 'scikits.datasmooth'
DESCRIPTION         = 'Scikits data smoothing package'
LONG_DESCRIPTION    = descr
MAINTAINER          = 'Jonathan Stickel',
MAINTAINER_EMAIL    = 'jjstickel@vcn.com',
URL                 = 'http://scikits.appspot.com/'
LICENSE             = 'BSD'
DOWNLOAD_URL        = URL
VERSION             = '0.5'

import setuptools
from numpy.distutils.core import setup

def configuration(parent_package='', top_path=None, package_name=DISTNAME):
    if os.path.exists('MANIFEST'): os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(package_name, parent_package, top_path,
                           version = VERSION,
                           maintainer  = MAINTAINER,
                           maintainer_email = MAINTAINER_EMAIL,
                           description = DESCRIPTION,
                           license = LICENSE,
                           url = URL,
                           download_url = DOWNLOAD_URL,
                           long_description = LONG_DESCRIPTION)

    return config

if __name__ == "__main__":
    setup(configuration = configuration,
        install_requires = 'numpy',
        namespace_packages = ['scikits'],
        packages = setuptools.find_packages(),
        include_package_data = True,
        #test_suite="tester", # for python setup.py test
        zip_safe = True, # the package can run out of an .egg file
        classifiers =
            [ 'Development Status :: 1 - Planning',
              'Environment :: Console',
              'Intended Audience :: Developers',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: BSD License',
              'Topic :: Scientific/Engineering'])
