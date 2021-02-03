from setuptools import setup, find_packages

import versioneer


setup(
    name='openpifpaf_tracker',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    license='GNU AGPLv3',
    description='OpenPifPaf Tracker',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Sven Kreiss',
    author_email='me@svenkreiss.com',
    url='https://github.com/vita-epfl/openpifpaf_tracker',

    install_requires=[
        'matplotlib',
        'pysparkling',  # for log analysis
        'python-json-logger',
        'openpifpaf>=0.9',
        'scipy',
        'torch>=1.0.0',
        'torchvision',
    ],
    extras_require={
        'test': [
            'pylint',
            'pytest',
        ],
        'eval': [
            'poseval@https://github.com/svenkreiss/poseval/archive/packaging.zip',
        ],
        'train': [
            'motmetrics',
        ],
    },
)
