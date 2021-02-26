from setuptools import setup, find_packages

import versioneer


setup(
    name='openpifpaf_posetrack',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    license='MIT',
    description='OpenPifPaf plugin for Posetrack',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Sven Kreiss',
    author_email='research@svenkreiss.com',
    url='https://github.com/vita-epfl/openpifpaf_posetrack',

    install_requires=[
        'openpifpaf>=0.12.1',
    ],
    extras_require={
        'test': [
            'pycodestyle',
            'pylint',
            'pytest',
        ],
        'eval': [
            'poseval@https://github.com/svenkreiss/poseval/archive/packaging.zip',
        ],
        'train': [
            'motmetrics',
        ],
        'video': [
            'opencv-python',
        ],
    },
)
