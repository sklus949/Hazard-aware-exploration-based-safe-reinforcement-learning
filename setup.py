from setuptools import setup


setup(
    name='crowdnav',
    version='0.0.1',
    packages=[
        'crowd_nav',
        'crowd_nav.configs',
        'crowd_nav.data',
        'crowd_nav.data.hesrl',
        'crowd_nav.data.output',
        'crowd_nav.mpc_control',
        'crowd_nav.policy',
        'crowd_nav.trajectory_prediction',
        'crowd_nav.utils',
        'crowd_sim',
        'crowd_sim.envs',
        'crowd_sim.envs.policy',
        'crowd_sim.envs.utils',
    ],
    install_requires=[
        'gitpython',
        'gym', #0.15.x
        'matplotlib',
        'numpy',
        'scipy',
        'torch', #0.10.x
        'torchvision',
        'casadi', #3.6.x

    ],
    extras_require={
        'test': [
            'pylint',
            'pytest',
        ],
    },
)
