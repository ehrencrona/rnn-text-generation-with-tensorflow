#!/usr/bin/env python

from setuptools import setup

setup(name='Text Generator',
  version='1.0',
  description='RNN Text Generator',
  author='Andreas Ehrencrona',
  author_email='andreas.ehrencrona@velik.it',
  url='https://github.com/ehrencrona/rnn-text-generation-with-tensorflow',
  packages=['trainer'],
  install_requires=[
    'tensorflow',
  ]
)
