from setuptools import setup

setup(
	name = 'kcalc',
	packages = ['kcalc'],
	author = 'Kelvin Sherlock',
	entry_points = {
		'console_scripts': ['kc=kcalc.calc:main']

	},


)
