#! /usr/bin/env python

descr = """This is a scikit intended to include numerical methods for smoothing
data. """


DISTNAME            = 'scikit.datasmooth'
DESCRIPTION         = 'Scikit data smoothing package'
LONG_DESCRIPTION    = descr
MAINTAINER          = 'Jonathan Stickel'
MAINTAINER_EMAIL    = 'jjstickel@gmail.com'
URL                 = 'https://github.com/jjstickel/scikit-datasmooth/'
LICENSE             = 'BSD'
DOWNLOAD_URL        = 'http://pypi.python.org/pypi/scikits.datasmooth/'
VERSION             = '0.7.1'

from setuptools import setup, find_packages


setup(
    name=DISTNAME,
    version=VERSION,
    url=URL,
    namespace_packages=['scikits'],
    install_requires=['numpy', 'scipy'],
    extras_require={
        'all': ['cvxopt']
    },
    packages=find_packages(),
    include_package_data=True,
    zip_safe=True,  # the package can run out of an .egg file
    author=MAINTAINER,
    author_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    download_url=DOWNLOAD_URL,
    long_description=LONG_DESCRIPTION,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ]
)
