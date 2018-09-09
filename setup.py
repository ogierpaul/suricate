from setuptools import setup

setup(name='wookie',
      version='0.1',
      description='MAAAAAAH',
      url='http://github.com/ogierpaul/wookie',
      author='Flying Circus',
      author_email='flyingcircus@example.com',
      license='MIT',
      packages=['wookie'],
      install_requires=[
          'scikit-learn',
          'pandas',
          'numpy',
          'fuzzywuzzy',
          'pytest'
      ],
      zip_safe=False)
