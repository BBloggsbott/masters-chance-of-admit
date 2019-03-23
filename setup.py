from setuptools import setup

setup(
    name='masters_chance_of_admit',
    packages=['masters_chance_of_admit'],
    include_package_data=True,
    install_requires=[
        'flask','sklearn','pandas','shap','matplotlib', 'numpy'
    ],
    setup_requires=[
        'pytest-runner',
    ],
    tests_require=[
        'pytest',
    ],
)