# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()
    
setup(
    name='sample',
    version='0.1.0',
    description='Cosmic Rays at L2 as seen by Gaia Project',
    author='Asier Abreu',
    author_email='asierabreu@gmail.com',
    url='https://github.com/asierabreu/cosmics',
    license=license,
    packages=find_packages(exclude=('tests'))
)

